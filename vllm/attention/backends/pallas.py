from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

from neuronxcc.nki.kernels.attention import flash_fwd, FlashConfig
from neuronxcc.nki import baremetal
import neuronxcc.nki.language as nl
import numpy as np

numeric_func = baremetal(flash_fwd)

def softmax(x: np.ndarray, dim: int, zero_max_mode=False,
            mixed_precision=False, return_max_reduce=False):
    max_value = np.amax(x, axis=dim, keepdims=True)
    max_value = np.maximum(0, max_value) if zero_max_mode else max_value
    exp = np.exp(x - max_value)
    if mixed_precision:
        reduce = np.add.reduce(exp.astype(np.float32), axis=dim, keepdims=True).astype(x.dtype)
    else:
        reduce = np.add.reduce(exp, axis=dim, keepdims=True)
    if return_max_reduce:
        return exp / reduce, -max_value, np.reciprocal(reduce)
    return exp / reduce
 
 
def cpu_attention_forward(q, k, v, use_causal_mask=True, mixed_precision=True):
    def mixed_precision_matmul(a, b):
        input_dtype = a.dtype
        a, b = a.astype(np.float32), b.astype(np.float32)
        c = np.matmul(a, b)
        return c.astype(input_dtype)

    _, _, d, _ = q.shape

    # Compute golden output
    softmax_scale = 1.0 / (d ** 0.5)
    q_scaled = q * softmax_scale
    nheads = q.shape[1]
    kv_heads = k.shape[1]
    if nheads > kv_heads:
        k = np.repeat(k, nheads//kv_heads, axis=1)
        v = np.repeat(v, nheads//kv_heads, axis=1)
    raw_score = mixed_precision_matmul(q_scaled.transpose(0, 1, 3, 2), k)

    if use_causal_mask:
        # raw_score has K seq in the most inner dim
        # we want to mask all elements where Q idx is smaller than K idx with -inf
        # this maps to the upper triangle of the final two axes
        for i in range(raw_score.shape[0]):
            for j in range(raw_score.shape[1]):
                # -inf triggers invalid input error in softmax implementation, use a small negative instead
                # k=1 to exclude the diagonal, because each token can still attend to itself
                raw_score[i, j][np.triu_indices_from(raw_score[i, j], k=1)] = -9984.0

    norm_score, cached_negative_max, cached_sum_reciprocal = \
        softmax(raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True)

    # Transpose the result so it has the same layout as ours
    out_golden = mixed_precision_matmul(norm_score, v.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    return out_golden, cached_negative_max, cached_sum_reciprocal

class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PallasMetadata"]:
        return PallasMetadata

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
        return (num_kv_heads, num_blocks, block_size, head_size)

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

@dataclass
class PallasMetadata(AttentionMetadata):

    # Currently, input sequences can only contain all prefills
    # or all decoding.
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    effective_query_lens: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
        return self

    @property
    def decode_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.block_tables is not None
        assert self.context_lens is not None
        return self



class PallasAttentionBackendImpl(AttentionImpl):

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

        self.megacore_mode = None
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
        attn_metadata: PallasMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
        nki_impl: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache[0] = [num_kv_heads, num_blocks, block_size, head_size]
            kv_cache[1] = [num_kv_heads, num_blocks, block_size, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor 
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        if nki_impl:
            # q = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
            # k = (np.random.random_sample([bs, kv_heads or nheads, d, seqlen]) - 0.5) * 2
            batch_size, seq_len, hidden_size = query.shape
            bs = batch_size
            nheads = self.num_heads
            d = hidden_size / nheads
            seqlen = seq_len
            kv_heads = self.num_kv_heads
            dtype = np.float32
            use_causal_mask = True
            tile_size = 2048

            q = query.view(batch_size, nheads, d, seqlen)
            k = key.view(batch_size, kv_heads or nheads, d, seqlen)
            should_transpose_v = False
            if should_transpose_v:
                v = (np.random.random_sample([bs, nheads, d, seqlen]) - 0.5) * 2
                cpu_permute = (0, 1, 2, 3)
            else:
                v = (np.random.random_sample([bs, kv_heads or nheads, seqlen, d]) - 0.5) * 2
                cpu_permute = (0, 1, 3, 2)
            o_proj = np.zeros(shape=[bs, nheads, seqlen, d], dtype=dtype)
            q = nl.static_cast(q, dtype)
            k = nl.static_cast(k, dtype)
            v = nl.static_cast(v, dtype)
            training = False
            seed = None
            out_lse = np.zeros(shape=[bs, nheads, int(nl.tile_size.pmax), seqlen // nl.tile_size.pmax], 
                                    dtype=np.float32) if training else None

            # o_proj_golden, cached_negative_max, cached_sum_reciprocal  = \
            # cpu_attention_forward(q, k, v.transpose(cpu_permute), use_causal_mask=use_causal_mask,mixed_precision=True)
            # o_proj_golden = o_proj_golden.transpose(0,1,3,2) # (b,h, d, seq)
            cached_negative_max = cached_negative_max.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                            nl.tile_size.pmax).transpose(0, 1, 3, 2)
            cached_sum_reciprocal = cached_sum_reciprocal.reshape(bs, nheads, seqlen // nl.tile_size.pmax,
                                                                nl.tile_size.pmax).transpose(0, 1, 3, 2)
            # lse_golden = -1.0 * (cached_negative_max + np.log(cached_sum_reciprocal)) if training else None
            config = FlashConfig(**{'seq_tile_size':tile_size, 'training':training, 'should_transpose_v':should_transpose_v})

            heads = nheads if kv_heads is None else kv_heads
            numeric_func[bs, heads](q, k, v, seed, o_proj, out_lse, seed,
                                    use_causal_mask=use_causal_mask, mixed_precision=True, config=config)
            # o_proj (b,h,d,seq)
            output = o_proj
        else:
            assert k_scale == 1.0 and v_scale == 1.0
            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")
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
            if attn_metadata.num_prefills > 0:
                assert seq_len % 16 == 0, (
                    "Pallas FlashAttention kernel requires seq_len to be a "
                    f"multiple of 16 but got {seq_len}")

                # Handle GQA/MQA.
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=-2)
                    key = key.view(batch_size, seq_len, self.num_heads,
                                self.head_size)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=-2)
                    value = value.view(batch_size, seq_len, self.num_heads,
                                    self.head_size)
                # FlashAttention requires [batch_size, num_heads, seq_len, d_model]
                # while the input is [batch_size, seq_len, num_heads, d_model].
                # Permute the input to match the required format.
                output = torch.ops.xla.flash_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    True,
                )
                output = output.permute(0, 2, 1, 3)
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
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

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
