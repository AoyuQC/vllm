from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadataBuilder, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

B_P_SIZE = 128
import torch.nn.functional as F
from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_nkifunc

@torch.library.custom_op("mylib::neuron_paged_attn", mutates_args=())
def neuron_paged_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    n_kv_head: int = None,
    head_size: int = None,
    B_P_SIZE: int = 128,
    LARGE_TILE_SZ: int = 2048,
    return_debug_tensors: bool = False,
    mixed_precision: bool = True,
) -> torch.Tensor:
    output_nki = flash_attn_varlen_nkifunc(
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        attn_mask,
        n_kv_head,
        head_size,
        B_P_SIZE,
        LARGE_TILE_SZ,
        return_debug_tensors,
        mixed_precision,
    )
    return torch.tensor(output_nki)

@neuron_paged_attn.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    n_kv_head: int = None,
    head_size: int = None,
    B_P_SIZE: int = 128,
    LARGE_TILE_SZ: int = 2048,
    return_debug_tensors: bool = False,
    mixed_precision: bool = True,
) -> torch.Tensor:
    # return query.new_empty(query.shape)
    return torch.empty_like(query.transpose(-2, -1))


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

    # @staticmethod
    # def swap_blocks(
    #     src_kv_cache: torch.Tensor,
    #     dst_kv_cache: torch.Tensor,
    #     src_to_dst: torch.Tensor,
    # ) -> None:
    #     raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # @torch.compile(backend="openxla")
    # @staticmethod
    # def copy_blocks(
    #     kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    #     src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    # ) -> None:
    #     src_indices, dst_indices = src_to_dists
    #     for k_cache, v_cache in kv_caches:
    #         torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
    #         k_cache[:, dst_indices] = k_cache[:, src_indices]
    #         torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
    #         v_cache[:, dst_indices] = v_cache[:, src_indices]


@dataclass
class NeuronAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int # Number of tokens excluding padding
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_start_loc: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    num_active_blocks: int
    active_block_table: torch.Tensor
    attn_mask: torch.Tensor
    num_input_tokens: int = 0 # Number of tokens including padding
    # context_lens: Optional[torch.Tensor] = None

class NeuronAttentionMetadataBuilder(AttentionMetadataBuilder[NeuronAttentionMetadata]):
    ...

class NeuronAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    @torch.inference_mode()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
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
        torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]
        
        num_tokens = query.shape[1]
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache[0].numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            write_to_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        pad_dims = (
            0,
            B_P_SIZE - query.shape[2],
            0,
            0,
            0,
            B_P_SIZE - query.shape[0],
        )
        # output = torch.empty_like(query)
        # output = []
        # padding
        query = F.pad(query, pad_dims, "constant", 0)
        key = F.pad(key, pad_dims, "constant", 0)
        value = F.pad(value, pad_dims, "constant", 0)
            

        k_cache = F.pad(k_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        v_cache = F.pad(v_cache, (0, B_P_SIZE - self.head_size), "constant", 0)

        query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        input_args = (
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata.active_block_table,
            attn_metadata.attn_mask,
        )
        input_kwargs = dict(
            n_kv_head=self.num_kv_heads,
            head_size=self.head_size,
            mixed_precision=False,
        )
        output = neuron_paged_attn(*input_args, **input_kwargs)
        output = output.permute(
            0, 2, 1, 3)[:, :num_tokens, :, :self.head_size]
        output = output.reshape(1, num_tokens, self.num_heads * self.head_size)
        # output = output.transpose(1,2).reshape(1, num_tokens, self.num_heads * self.head_size)
        return output
        # assert k_scale == 1.0 and v_scale == 1.0
        # batch_size = 1
        # seq_len, hidden_size = query.shape
        # # print(f"hidden size is {hidden_size}")
        # query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        # key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        # value = value.view(batch_size, seq_len, self.num_kv_heads,
        #                    self.head_size)
        

        # # print(f"key is {key}")
        # # print(key.shape)
        # # print(f"self.heads {self.num_heads}")
        # if kv_cache[0].numel() > 0:
        #     slot_mapping = attn_metadata.slot_mapping
        #     key_cache, value_cache = kv_cache
        #     write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        # # query = query * self.scale
        # pad_dims = (
        #     0,
        #     B_P_SIZE - query.shape[3],
        #     0,
        #     0,
        #     0,
        #     B_P_SIZE - query.shape[1],
        #     0,
        #     0
        # )
        # # output = torch.empty_like(query)
        # # output = []
        # # padding
        # query = F.pad(query, pad_dims, "constant", 0)
        # key = F.pad(key, pad_dims, "constant", 0)
        # value = F.pad(value, pad_dims, "constant", 0)
        # key_cache = F.pad(key_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        # value_cache = F.pad(value_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        # # key_cache=key_cache.permute(1,2,0,3).contiguous()
        # # value_cache=value_cache.permute(1,2,0,3).contiguous()
        # # from vllm.attention.ops.nki_flash_attn import context_attention_fwd, context_flash_attention_fwd
        # from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_nkifunc
        # attn_mask = attn_metadata.attn_mask
        # q = query.permute(0,2,3,1).contiguous()
        # k = key.permute(0,2,3,1).contiguous()
        # v = value.permute(0,2,1,3).contiguous()
        # out = flash_attn_varlen_nkifunc(
        #     q,
        #     k,
        #     v,
        #     key_cache,
        #     value_cache,
        #     block_table=attn_metadata.active_block_table,
        #     attn_mask=attn_mask,
        #     n_kv_head=self.num_kv_heads,
        #     head_size=self.head_size,
        # )
        # # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
        # out = out.permute(
        #     0, 2, 1, 3)[:, :seq_len, :, :self.head_size]
        # return out.reshape(seq_len, hidden_size)

def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:

    key_cache = key_cache.flatten(0, 1)
    value_cache = value_cache.flatten(0, 1)

    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)

# def write_to_kv_cache(
#     key: torch.Tensor,
#     value: torch.Tensor,
#     key_cache: torch.Tensor,
#     value_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
# ) -> None:
#     torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
#     torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

#     key = key.flatten(0, 2)
#     value = value.flatten(0, 2)
#     key_cache = key_cache.flatten(0, 2)
#     value_cache = value_cache.flatten(0, 2)
#     key_cache.index_copy_(0, slot_mapping, key)
#     value_cache.index_copy_(0, slot_mapping, value)

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
