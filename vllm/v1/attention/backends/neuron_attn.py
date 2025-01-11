from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

B_P_SIZE = 128
import torch.nn.functional as F

class NeuronAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "NEURON_ATTN_V1"

    @staticmethod
    def get_impl_cls() -> Type["NeuronAttentionBackendImpl"]:
        return NeuronAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["NeuronAttentionMetadata"]:
        return NeuronAttentionMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)
        # return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    @torch.compile(backend="openxla")
    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]

class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = (torch.arange(n_queries).reshape(n_queries,
                                             1).expand(n_queries, n_keys))
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens)
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size)
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens)
        return prior_mask, active_mask


@dataclass
class NeuronAttentionMetadata:

    slot_mapping: torch.Tensor
    is_prompt: Optional[bool] = False
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    active_block_table: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    # Currently, input sequences can only contain all prefills
    # or all decoding.
    # block_tables: Optional[torch.Tensor] = None
    # context_lens: Optional[torch.Tensor] = None
    # effective_query_lens: Optional[torch.Tensor] = None

    # @property
    # def prefill_metadata(self) -> Optional["NeuronAttentionMetadata"]:
    #     if self.num_prefills == 0:
    #         return None

    #     assert self.num_decode_tokens == 0
    #     return self

    # @property
    # def decode_metadata(self) -> Optional["NeuronAttentionMetadata"]:
    #     if self.num_decode_tokens == 0:
    #         return None

    #     assert self.num_prefills == 0
    #     assert self.num_prefill_tokens == 0
    #     assert self.block_tables is not None
    #     assert self.context_lens is not None
    #     return self


class NeuronAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # HACK AOYU Disable initial check in  NeuronAttentionBackendImpl(AttentionImpl)
        # if head_size % 128 != 0:
        #     raise NotImplementedError("Head size must be a multiple of 128.")
        # if alibi_slopes is not None:
        #     raise NotImplementedError("Alibi slopes is not supported.")
        # if sliding_window is not None:
        #     raise NotImplementedError("Sliding window is not supported.")
        # if kv_cache_dtype != "auto":
        #     raise NotImplementedError("FP8 KV cache dtype is not supported.")
        # if blocksparse_params is not None:
        #     raise NotImplementedError("Blocksparse is not supported.")
        # if logits_soft_cap is not None:
        #     raise NotImplementedError(
        #         "Attention logits soft-capping is not supported.")

        # if torch_xla.tpu.version() < 4:
        #     raise NotImplementedError("TPU version must be 4 or higher.")

        # self.megacore_mode = None
        # tpu_env = torch_xla.tpu.get_tpu_env()
        # tpu_type = (tpu_env.get("ACCELERATOR_TYPE", None)
        #             or tpu_env.get("TYPE", None)
        #             or tpu_env.get("TPU_ACCELERATOR_TYPE", None))
        # assert tpu_type is not None
        # tpu_type = tpu_type.lower()

        # if (("lite" not in tpu_type) and ("v6" not in tpu_type)):
        #     if self.num_kv_heads % 2 == 0:
        #         self.megacore_mode = "kv_head"
        #     else:
        #         # NOTE(woosuk): If the batch size is not a multiple of 2, the
        #         # megacore mode will be None.
        #         self.megacore_mode = "batch"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Neuron attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            [1, num_tokens_to_compute, num_heads * head_size]
            attn_mask
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            # kv_cache[0] = [num_kv_heads, num_blocks, block_size, head_size]
            # kv_cache[1] = [num_kv_heads, num_blocks, block_size, head_size]
            kv_cache[0] = [num_blocks, block_size, num_kv_heads, head_size]
            kv_cache[1] = [num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor 
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "NeuronAttentionBackendImpl")
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)
        

        if kv_cache[0].numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        pad_dims = (
            0,
            B_P_SIZE - query.shape[3],
            0,
            0,
            0,
            B_P_SIZE - query.shape[1],
            0,
            0
        )
        # output = torch.empty_like(query)
        # output = []
        # padding
        query = F.pad(query, pad_dims, "constant", 0)
        key = F.pad(key, pad_dims, "constant", 0)
        value = F.pad(value, pad_dims, "constant", 0)
        key_cache = F.pad(key_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        value_cache = F.pad(value_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        # key_cache=key_cache.permute(1,2,0,3).contiguous()
        # value_cache=value_cache.permute(1,2,0,3).contiguous()
        # from vllm.attention.ops.nki_flash_attn import context_attention_fwd, context_flash_attention_fwd
        from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_nkifunc
        attn_mask = attn_metadata.attn_mask
        q = query.permute(0,2,3,1).contiguous()
        k = key.permute(0,2,3,1).contiguous()
        v = value.permute(0,2,1,3).contiguous()
        out = flash_attn_varlen_nkifunc(
            q,
            k,
            v,
            key_cache,
            value_cache,
            block_table=attn_metadata.active_block_table,
            attn_mask=attn_mask,
            n_kv_head=self.num_kv_heads,
            head_size=self.head_size,
        )
        # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        out = out.permute(
            0, 2, 1, 3)[:, :seq_len, :, :self.head_size]
        # out = out[:,:,:seq_len,:self.head_size]
        # out = out.permute(0, 2, 1, 3).contiguous()
        # output.append(out)
        # for batch_idx in range(batch_size):
        #     q = query[batch_idx,:,:,:].unsqueeze(0)
        #     k = key[batch_idx,:,:,:].unsqueeze(0)
        #     v = value[batch_idx,:,:,:].unsqueeze(0)
        #     # debug tricks
        #     attn_mask = attn_metadata.attn_mask.permute(0,1)
        #     # FIXME aoyu : only left context_attention_fwd, same as AOYU
        #     # active_block_table: padded 0 if not active
        #     if attn_metadata.is_prompt is True:
        #         out = context_flash_attention_fwd(
        #             q.permute(0,2,3,1).contiguous(),
        #             k.permute(0,2,3,1).contiguous(),
        #             v.permute(0,2,1,3).contiguous(),
        #             attn_mask=attn_metadata.attn_mask,
        #             head_size=self.head_size,
        #         )
        #         # out = q.permute(0,2,3,1)
        #     else:
        #         q = q.permute(0,2,3,1).contiguous()
        #         k = k.permute(0,2,3,1).contiguous()
        #         v = v.permute(0,2,1,3).contiguous()
        #         out = context_attention_fwd(
        #             q,
        #             k,
        #             v,
        #             key_cache,
        #             value_cache,
        #             block_table=attn_metadata.active_block_table,
        #             attn_mask=attn_mask,
        #             n_kv_head=self.num_kv_heads,
        #             head_size=self.head_size,
        #         )
        #     out = out[:,:,:seq_len,:self.head_size]
        #     out = out.permute(0, 2, 1, 3).contiguous()
        #     output.append(out)
        # Reshape the output tensor.
        # output = torch.cat(output, dim=0)
        return out.reshape(batch_size, seq_len, hidden_size)
        # if attn_metadata.num_prefills > 0:
        if attn_metadata.is_prompt is True:
            # if attn_metadata.block_tables is None:
            #     # Prefill without paged KV cache.
            assert seq_len % 16 == 0, (
                "Pallas FlashAttention kernel requires seq_len to be a "
                f"multiple of 16 but got {seq_len}")

            # # Handle GQA/MQA.
            # # NKI attention dosen't need this
            # if self.num_kv_heads != self.num_heads:
            #     key = key.repeat_interleave(self.num_queries_per_kv,
            #                                 dim=-2)
            #     key = key.view(batch_size, seq_len, self.num_heads,
            #                    self.head_size)
            #     value = value.repeat_interleave(self.num_queries_per_kv,
            #                                     dim=-2)
            #     value = value.view(batch_size, seq_len, self.num_heads,
            #                        self.head_size)
            # FlashAttention kernel requires the input shape to be
            # [batch_size, num_heads, seq_len, d_model]
            # while the input is [batch_size, seq_len, num_heads, d_model].
            # Permute the input to match the required format.
            # NKI Flash paged  attention reuiqres the input shape to be
            # query: [1, n_heads, d, seq_q]
            # key: [1, n_kv_heads, d, seq_k]
            # value: [1, n_kv_heads, seq_v, d]
            # key_cache: (num_blocks, block_size, n_kv_heads, d)
            # value_cache: (num_blocks, block_size, n_kv_heads, d)
            # block_tables: (num_active_blocks, )
            # mask: (seq_q, num_active_blocks * block_size)
            # o: shape (1, n_heads, seq_q, d)
            # l_m: shape (1, n_heads, seq_q, 2)
            # padding
            pad_dims = (
                0,
                B_P_SIZE - query.shape[3],
                0,
                0,
                0,
                B_P_SIZE - query.shape[1],
                0,
                0
            )
            query = F.pad(query, pad_dims, "constant", 0)
            key = F.pad(key, pad_dims, "constant", 0)
            value = F.pad(value, pad_dims, "constant", 0)
            key_cache = F.pad(key_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
            value_cache = F.pad(value_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
            from vllm.attention.ops.nki_flash_attn import context_attention_fwd
            output = context_attention_fwd(
                query.permute(0,2,3,1),
                key.permute(0,2,3,1),
                value.permute(0,2,1,3),
                key_cache=key_cache.permute(0,1,2,3),
                value_cache=value_cache.permute(0,1,2,3),
                block_table=attn_metadata.active_block_table,
                attn_mask=attn_metadata.attn_mask,
                # n_kv_head=self.num_kv_heads,
                # head_size=self.head_size,
                # query_lens=torch.tensor([seq_len]),
                # seq_lens=torch.tensor([seq_len]),
                # batch_size=batch_size,
                # block_size=block_size,
                # max_model_len=seq_len,
            )
            # output = torch.ops.xla.flash_attention(
            #     query.permute(0, 2, 1, 3),
            #     key.permute(0, 2, 1, 3),
            #     value.permute(0, 2, 1, 3),
            #     True,
            # )
            output = output[:,:,:seq_len,:self.head_size]
            output = output.permute(0, 2, 1, 3)
            # else:
            #     # Prefill with paged KV cache.
            #     # TODO(woosuk): Tune the below knobs.
            #     num_kv_pages_per_compute_block = 16
            #     num_queries_per_compute_block = 16
            #     assert seq_len % num_queries_per_compute_block == 0
            #     # output = torch.ops.xla.multi_queries_paged_attention(
            #     #     query,
            #     #     key_cache,
            #     #     value_cache,
            #     #     attn_metadata.context_lens,
            #     #     attn_metadata.block_tables,
            #     #     attn_metadata.effective_query_lens,
            #     #     num_kv_pages_per_compute_block,
            #     #     num_queries_per_compute_block,
            #     #     use_kernel=True,
            #     # )
            #     from vllm.attention.ops.nki_flash_attn import context_attention_fwd
            #     output = context_attention_fwd(
            #         query.permute(0,2,3,1),
            #         key.permute(0,2,3,1),
            #         value.permute(0,2,1,3),
            #         # query,
            #         # key,
            #         # value,
            #         key_cache=key_cache,
            #         value_cache=value_cache,
            #         block_table=None,
            #         attn_mask=None,
            #         n_kv_head=self.num_kv_heads,
            #         head_size=self.head_size,
            #         # query_lens=attn_metadata.effective_query_lens,
            #         # seq_lens=attn_metadata.context_lens+attn_metadata.effective_query_lens,
            #         # batch_size=batch_size,
            #         # block_size=block_size,
            #     )
        else:
            # Decoding run.
            assert kv_cache[0].numel() > 0
            query = query.squeeze(dim=1)
            pages_per_compute_block = 16  # TODO(woosuk): Tune this value.

            assert attn_metadata.block_tables is not None
            assert attn_metadata.context_lens is not None
            # NOTE(woosuk): The PagedAttention Pallas kernel stores the entire
            # block table in SMEM. Therefore, if the block table is too large,
            # the kernel compilation will fail. To avoid this, we split the
            # batch dimension into smaller chunks and run the kernel multiple
            # times.
            MAX_SMEM_USAGE = 512 * 1024
            size_per_seq = 4 * attn_metadata.block_tables.shape[1]
            max_num_seq = MAX_SMEM_USAGE // size_per_seq

            if batch_size <= max_num_seq:
                output = paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    pages_per_compute_block,
                    self.megacore_mode,
                )
            else:
                chunk_size = max_num_seq
                # Make sure the chunk size is a multiple of 2.
                chunk_size = chunk_size // 2 * 2
                num_chunks = (batch_size + chunk_size - 1) // chunk_size

                output = torch.empty_like(query)
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = chunk_start + chunk_size
                    # NOTE(woosuk): We skip this line because it causes Dynamo
                    # compilation error. Instead, we rely on the slice operation
                    # to handle the out-of-bound case.
                    # chunk_end = min(chunk_end, batch_size)
                    chunk_output = paged_attention(
                        query[chunk_start:chunk_end],
                        key_cache,
                        value_cache,
                        attn_metadata.context_lens[chunk_start:chunk_end],
                        attn_metadata.block_tables[chunk_start:chunk_end],
                        pages_per_compute_block,
                        self.megacore_mode,
                    )
                    output[chunk_start:chunk_end] = chunk_output

        # Reshape the output tensor.
        return output.reshape(batch_size, seq_len, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    # torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    # torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    pages_per_compute_block: int,
    megacore_mode: Optional[str],
) -> torch.Tensor:
    batch_size = query.shape[0]
    if megacore_mode == "batch" and batch_size % 2 != 0:
        megacore_mode = None
    else:
        megacore_mode = megacore_mode

    # NOTE(woosuk): A temporary workaround to avoid the error:
    # "xla::paged_attention() Expected a value of type 'str' for
    # argument 'megacore_mode' but instead found type 'NoneType'."
    if megacore_mode is not None:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
            megacore_mode=megacore_mode,
        )
    else:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
        )
    return output
