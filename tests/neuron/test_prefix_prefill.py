import random
from typing import Optional

import pytest
import torch
import torch.nn.functional as F


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


def ref_softmax(x: torch.Tensor,
                dim: int,
                mixed_precision=False,
                return_max_reduce=False):
    max_value = torch.amax(x, dim=dim, keepdims=True)
    exp = torch.exp(x - max_value)
    if mixed_precision:
        sum_value = torch.sum(exp.astype(torch.float32),
                              dim=dim,
                              keepdims=True).astype(x.dtype)
    else:
        sum_value = torch.sum(exp, dim=dim, keepdims=True)
    if return_max_reduce:
        return exp / sum_value, max_value, torch.reciprocal(sum_value)
    return exp / sum_value


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    return_max_reduce: Optional[bool] = False,
) -> torch.Tensor:
    scaled_qk = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        masked_score = scaled_qk + attn_mask.float()
    if return_max_reduce:
        norm_score, cached_max, cached_sum_reciprocal = ref_softmax(
            masked_score, dim=-1, return_max_reduce=True)
    else:
        norm_score = ref_softmax(masked_score, dim=-1)
    out = torch.einsum("hqk,khd->qhd", norm_score, value)
    if return_max_reduce:
        return (
            out,
            cached_max,
            cached_sum_reciprocal,
            norm_score,
            masked_score,
            scaled_qk,
        )
    else:
        return out


def ref_context_attention(
    query,
    key,
    value,
    query_lens,
    seq_lens,
    head_size,
    num_kv_heads,
    num_heads,
    num_queries_per_kv,
    return_max_reduce=False,
):
    scale = float(1.0 / (head_size**0.5))
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    attn_mask, _ = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens)

    # convert binary mask to -inf values
    attn_mask = torch.logical_not(attn_mask)
    attn_mask = attn_mask.float() * -30000

    output, cached_max, cached_sum_reciprocal, lse, masked_score, scaled_qk = (
        ref_masked_attention(
            query,
            key,
            value,
            scale,
            attn_mask,
            return_max_reduce=return_max_reduce,
        ))

    output = output.unsqueeze(1)
    if return_max_reduce:
        return (
            output,
            cached_max,
            cached_sum_reciprocal,
            lse,
            masked_score,
            scaled_qk,
        )
    else:
        return output


@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size",
    [
        (4, 2, 8),
    ],
)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
) -> None:
    import os

    os.environ["NEURON_CC_FLAGS"] = (
        " --model-type=transformer -O1 "
        " --internal-hlo2tensorizer-options='--verify-hlo' ")

    random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    min_ctx_len = 2
    max_ctx_len = 64
    min_query_len = 2
    max_query_len = 64
    prefill_batch_size = 2
    decode_batch_size = 6
    batch_size = prefill_batch_size + decode_batch_size
    block_size = 32
    max_model_len = (max_query_len + max_ctx_len) * 4

    max_block_per_request = max_model_len // block_size
    dtype = torch.float32
    cache_size = (batch_size * max_block_per_request) + 2
    ctx_lens = [
        random.randint(min_ctx_len, max_ctx_len)
        for _ in range(prefill_batch_size)
    ] + [
        random.randint(min_ctx_len, max_ctx_len)
        for _ in range(decode_batch_size)
    ]
    query_lens = [
        random.randint(min_query_len, max_query_len)
        for _ in range(prefill_batch_size)
    ] + [1 for _ in range(decode_batch_size)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1, 1)
    torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1, 1)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:batch_size * max_block_per_request].view(
        batch_size, max_block_per_request)
    torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(batch_size):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1

    (
        output_ref,
        cached_max,
        cached_sum_reciprocal,
        lse,
        masked_score,
        scaled_qk,
    ) = ref_context_attention(
        query,
        key,
        value,
        query_lens,
        seq_lens,
        head_size,
        num_kv_heads,
        num_heads,
        num_queries_per_kv,
        return_max_reduce=True,
    )

    from vllm.attention.ops.nki_flash_attn import nki_context_attention
    output_nki = nki_context_attention(
        query,
        key,
        value,
        key_cache,
        value_cache,
        query_lens,
        seq_lens,
    )
    output_ref = output_ref_padded.transpose(0, 1)[0, :n_queries, :, :]

    torch.testing.assert_close(output_nki, output_ref, atol=1e-2, rtol=0)
