import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, cdiv, is_pin_memory_available)
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
# from vllm.config import set_current_vllm_config

# vllm v1
from vllm.v1.attention.backends.neuron_attn import (NeuronAttentionBackend, NeuronAttentionMetadata)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.utils import bind_kv_cache

# debug configs for torch dynamo
import torch._dynamo.config
import logging
# torch._dynamo.config.verbose = True
# torch._logging.set_logs(dynamo=logging.DEBUG)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# TODO (aoyu), a better way to customize p/f parameters
# build neuron program
B_P_SIZE = 128
B_F_SIZE = 512
LARGE_TILE_SZ = 2048

# FIXME(aoyu): this is a hack to avoid out-of-bound index
_PAD_SLOT_ID = 1_000_000_000

@dataclass
class NeuronInputData:
    logits_indices: torch.Tensor = None
    attn_metadata: NeuronAttentionMetadata = None

class NeuronModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        input_registery: InputRegistry = INPUT_REGISTRY,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        self.input_registry = input_registery

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}
        # List[k_cache, v_cache]
        # self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            vocab_size=model_config.get_vocab_size(),
            # device=self.device,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device="cpu")
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device="cpu")
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device="cpu")

        # self.neuron_compilation_batch_sizes = list(reversed(self.vllm_config.compilation_config.capture_sizes))
        self.neuron_compilation_batch_sizes = [512]

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request in the batch.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
            start_index = len(req_state.block_ids) - len(
                req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    # def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    #     # Remove stopped requests from the cached states.
    #     # Keep the states of the pre-empted requests.
    #     for req_id in scheduler_output.finished_req_ids:
    #         self.requests.pop(req_id, None)
    #         self.encoder_cache.pop(req_id, None)
        
    #     # Free the cached encoder outputs.
    #     for req_id, input_id in scheduler_output.free_encoder_input_ids:
    #         encoder_outputs = self.encoder_cache.get(req_id)
    #         if encoder_outputs is not None:
    #             encoder_outputs.pop(input_id, None)
    #             if not encoder_outputs:
    #                 self.encoder_cache.pop(req_id, None)

    #     # Remove the requests from the persistent batch.
    #     stopped_req_ids = set().union(
    #         scheduler_output.preempted_req_ids,
    #         scheduler_output.finished_req_ids,
    #     )
    #     removed_req_indices: List[int] = []
    #     for req_id in stopped_req_ids:
    #         req_index = self.input_batch.remove_request(req_id)
    #         if req_index is not None:
    #             removed_req_indices.append(req_index)

    #     # Update the states of the running requests.
    #     for req_data in scheduler_output.scheduled_running_reqs:
    #         req_id = req_data.req_id
    #         req_state = self.requests[req_id]
    #         req_index = self.input_batch.req_id_to_index[req_id]

    #         # Update the num_computed_tokens.
    #         req_state.num_computed_tokens = req_data.num_computed_tokens
    #         self.input_batch.num_computed_tokens_cpu[req_index] = (
    #             req_data.num_computed_tokens)

    #         # Update the block table.
    #         num_new_blocks = len(req_data.new_block_ids)
    #         if num_new_blocks == 0:
    #             continue
    #         start_index = len(req_state.block_ids)
    #         end_index = start_index + num_new_blocks
    #         req_state.block_ids.extend(req_data.new_block_ids)
    #         self.input_batch.block_table_cpu[
    #             req_index, start_index:end_index] = req_data.new_block_ids

    #     req_ids_to_add: List[str] = []
    #     # Add new requests to the cached states.
    #     for req_data in scheduler_output.scheduled_new_reqs:
    #         req_id = req_data.req_id
    #         sampling_params = req_data.sampling_params
    #         if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
    #             generator = torch.Generator(device=self.device)
    #             generator.manual_seed(sampling_params.seed)
    #         else:
    #             generator = None

    #         self.requests[req_id] = CachedRequestState(
    #             req_id=req_id,
    #             prompt_token_ids=req_data.prompt_token_ids,
    #             prompt=req_data.prompt,
    #             mm_inputs=req_data.mm_inputs,
    #             mm_positions=req_data.mm_positions,
    #             sampling_params=sampling_params,
    #             generator=generator,
    #             block_ids=req_data.block_ids,
    #             num_computed_tokens=req_data.num_computed_tokens,
    #             output_token_ids=[],
    #         )
    #         req_ids_to_add.append(req_id)

    #     # Update the cached states of the resumed requests.
    #     for req_data in scheduler_output.scheduled_resumed_reqs:
    #         req_id = req_data.req_id
    #         req_state = self.requests[req_id]

    #         req_state.block_ids = req_data.block_ids
    #         req_state.num_computed_tokens = req_data.num_computed_tokens
    #         req_ids_to_add.append(req_id)

    #     # Add the new or resumed requests to the persistent batch.
    #     # The smaller empty indices are filled first.
    #     removed_req_indices = sorted(removed_req_indices, reverse=True)
    #     for req_id in req_ids_to_add:
    #         req_state = self.requests[req_id]
    #         if removed_req_indices:
    #             # Fill the empty index
    #             req_index = removed_req_indices.pop()
    #         else:
    #             # Append to the end
    #             req_index = None
    #         self.input_batch.add_request(req_state, req_index)
        
    #     # Condense the batched states is there are empty indicies
    #     if removed_req_indices:
    #         self.input_batch.condense(removed_req_indices)


    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> NeuronInputData:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs

        # OPTIMIZAION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.block_table[:num_reqs].copy_(
            self.input_batch.block_table.block_table_cpu[:num_reqs],
            non_blocking=True)
        
        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (num_reqs, 1))
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
        arange = arange_matrix[mask]

        # Get positions.
        positions = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        positions_np = positions.numpy()
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices = torch.from_numpy(token_indices)
        input_ids = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.index_select(torch.from_numpy(
            self.input_batch.token_ids_cpu).flatten(),
                           0,
                           token_indices,
                           out=input_ids)

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_numbers = self.input_batch.block_table.get_device_tensor().flatten()[
            req_indices * self.max_num_blocks_per_req +
            positions_np // self.block_size]
        block_offsets = torch.from_numpy(positions_np % self.block_size)
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)
        
        # _PAD_SLOT_ID = self.num_blocks * self.block_size
        padded_num_tokens = self._get_padded_batch_size(total_num_scheduled_tokens)
        slot_mapping_pad_length = padded_num_tokens - slot_mapping.shape[0]
        slot_mapping = torch.nn.functional.pad(
            slot_mapping,
            (0, slot_mapping_pad_length),
            'constant',
            _PAD_SLOT_ID
        )

        # Prepare the attention metadata.
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max()
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc_np = seq_start_loc.numpy()
        seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=seq_start_loc_np[1:])

        self.input_ids[:total_num_scheduled_tokens].copy_(input_ids,
                                                          non_blocking=True)
        self.positions[:total_num_scheduled_tokens].copy_(positions,
                                                          non_blocking=True)

        seq_lens = torch.diff(seq_start_loc)
        query_lens = torch.diff(query_start_loc)
        context_lens = seq_lens - query_lens
        def shift_bit_length(x):
            return 1 << (x - 1).bit_length()
        num_active_blocks_shifted = shift_bit_length(
            ((context_lens+ self.block_size - 1) // self.block_size).sum().item()
        )
        num_active_blocks_factor = (LARGE_TILE_SZ // self.block_size // num_active_blocks_shifted)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
        # FIXME AOYU: for block_size = 2048, num_active_blocks_factos = 0, it's not correct

        assert (num_active_blocks * self.block_size) == LARGE_TILE_SZ, f"invalid {num_active_blocks=}, {self.block_size=}, {LARGE_TILE_SZ=}"

        context_kv_len = num_active_blocks * self.block_size
        assert context_kv_len == LARGE_TILE_SZ, f"invalid {context_kv_len=}, {LARGE_TILE_SZ=}"


        block_table = self.input_batch.block_table.block_table[:num_reqs]
        def get_active_block_tables(block_tables, query_lens, seq_lens, block_size,
                                            num_blocks):
            context_lens = seq_lens - query_lens
            blocks_per_seq = (context_lens + block_size - 1) // block_size
            num_seqs = len(seq_lens)
            active_blocks: list[int] = []
            for seq_id in range(num_seqs):
                active_blocks = (
                    active_blocks +
                    block_tables[seq_id, :blocks_per_seq[seq_id]].tolist())
            return nn.functional.pad(
                torch.tensor(active_blocks),
                (0, num_blocks - len(active_blocks)),
                "constant",
                0,
            )
        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor(query_lens),
            torch.tensor(seq_lens),
            self.block_size,
            num_active_blocks,
        )

        prior_mask, active_mask = (
            BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                query_lens=query_lens.tolist(), seq_lens=seq_lens.tolist(), block_size=self.block_size
            )
        )
        
        attn_mask = torch.concat(
            [
                nn.functional.pad(
                    prior_mask,
                    (
                        0,
                        LARGE_TILE_SZ - prior_mask.shape[1],
                        0,
                        B_P_SIZE - prior_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
                nn.functional.pad(
                    active_mask,
                    (
                        0,
                        B_F_SIZE - active_mask.shape[1],
                        0,
                        B_P_SIZE - active_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
            ],
            dim=1,
        )
        
        logits_indices = query_start_loc[1:] - 1
        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        slot_mapping = slot_mapping.long().to(self.device, non_blocking=True)
        active_block_table = active_block_table.to(torch.int32).to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device)
        attn_metadata = NeuronAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_tables=self.input_batch.block_table.block_table[:num_reqs],
            slot_mapping=slot_mapping,
            num_active_blocks=num_active_blocks,
            active_block_table=active_block_table,
            attn_mask=attn_mask,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        return attn_metadata, logits_indices
        return NeuronInputData(
            logits_indices=logits_indices,
            attn_metadata=attn_metadata,
        )


        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)

        # Get active block tables
        block_table = self.input_batch.block_table
        block_size = self.block_size

        # Get context lens list
        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1, 1))
        index = positions.to(torch.int64)
        positions = positions[:num_reqs]

        context_lens = positions.reshape(-1) - 1

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:num_reqs]

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.gather(
            input=self.input_batch.block_table_cpu_tensor,
            dim=1,
            index=(index // self.block_size))
        block_offsets = index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        # FIXME AOYU: solve slot_mapping index for 
        # slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:num_reqs]

        num_scheduled_tokens = []
        positions_list = list(positions)
        token_ids_list = list(token_ids)
        slot_mapping_list = list(slot_mapping)
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens > 1:
                positions_list[idx] = self.prefill_positions[:, :num_tokens]
                updated_block_number = self.prefill_positions[:, :num_tokens] // self.block_size
                updated_block_offset = self.prefill_positions[:, :num_tokens] % self.block_size
                slot_mapping_list[idx] = updated_block_number * self.block_size + updated_block_offset
                token_ids_list[idx] = torch.from_numpy(self.input_batch.token_ids_cpu[
                idx, :num_tokens].reshape(1, -1))
            else:
                positions_list[idx] = positions[idx].unsqueeze(0)
                slot_mapping_list[idx] = slot_mapping[idx].unsqueeze(0)
                token_ids_list[idx] = token_ids[idx].unsqueeze(0)
            num_scheduled_tokens.append(num_tokens)
        positions = torch.concat(positions_list, dim=1).squeeze(0)
        slot_mapping = torch.concat(slot_mapping_list, dim=1).squeeze(0).long()
        token_ids = torch.concat(token_ids_list, dim=1).unsqueeze(0)
        
        ctx_lens = list(context_lens)
        query_lens = num_scheduled_tokens
        seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]

        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        block_table = self.input_batch.block_table_cpu_tensor[:
                                                              num_reqs]

        def get_active_block_tables(block_tables, query_lens, seq_lens, block_size,
                                    num_blocks):
            context_lens = seq_lens - query_lens
            blocks_per_seq = (context_lens + block_size - 1) // block_size
            num_seqs = len(seq_lens)
            active_blocks: list[int] = []
            for seq_id in range(num_seqs):
                active_blocks = (
                    active_blocks +
                    block_tables[seq_id, :blocks_per_seq[seq_id]].tolist())
                if seq_lens[seq_id] == 0:
                    break
            return F.pad(
                torch.tensor(active_blocks),
                (0, num_blocks - len(active_blocks)),
                "constant",
                0,
            )
        
                # Build and pad input tensors
        def shift_bit_length(x):
            return 1 << (x - 1).bit_length()

        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        num_active_blocks_shifted = shift_bit_length(
            ((context_lens + block_size - 1) // block_size).sum().item())
        num_active_blocks_factor = (LARGE_TILE_SZ // block_size //
                                    num_active_blocks_shifted)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor

        assert (num_active_blocks *
                block_size) == LARGE_TILE_SZ, "invalid {num_active_blocks=}"

        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor(query_lens),
            torch.tensor(seq_lens),
            self.block_size,
            num_active_blocks,
        )

        context_kv_len = num_active_blocks * block_size
        assert context_kv_len == LARGE_TILE_SZ, f"invalid {context_kv_len=}"
        prior_mask, active_mask = (
            BlockDiagonalCausalFromBottomRightMask.from_seqlens(query_lens, seq_lens, block_size=block_size))

        attn_mask = torch.concat(
            [
                F.pad(
                    prior_mask,
                    (
                        0,
                        context_kv_len - prior_mask.shape[1],
                        0,
                        B_P_SIZE - prior_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
                F.pad(
                    active_mask,
                    (
                        0,
                        B_P_SIZE - active_mask.shape[1],
                        0,
                        B_P_SIZE - active_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
            ],
            dim=1,
        )

        token_ids = torch.flatten(token_ids).to(self.device)
        positions = positions.unsqueeze(0).to(self.device)

        # slot_mapping = torch.tensor(range(num_tokens)).repeat_interleave(self.num_kv_heads).reshape(-1, self.num_kv_heads)
        # slot_mapping = slot_mapping*self.num_kv_heads + torch.arange(0, self.num_kv_heads)
        # slot_mapping = slot_mapping.flatten()
        # slot_mapping = slot_mapping.to(device=self.device).long()

        logits_indicies =  torch.tensor(query_lens)-1

        return NeuronInputData(
            token_ids=token_ids,
            position_ids=positions,
            query_lens=query_lens,
            seq_lens=seq_lens,
            logits_indicies=logits_indicies,
            attn_metadata=NeuronAttentionMetadata(
                slot_mapping=slot_mapping.to(device=self.device),
                block_tables=block_table.to(self.device),
                context_lens=context_lens.to(self.device),
                active_block_table=active_block_table.to(torch.int32).to(self.device),
                attn_mask=attn_mask.to(self.device)
        ))

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        # skip_copy = True
        # if (scheduler_output.finished_req_ids
        #         or scheduler_output.preempted_req_ids):
        #     skip_copy = False
        # if (scheduler_output.scheduled_new_reqs
        #         or scheduler_output.scheduled_resumed_reqs):
        #     skip_copy = False
        # # Create the sampling metadata.
        # sampling_metadata = self.input_batch.make_sampling_metadata(skip_copy)
        # return sampling_metadata
        skip_copy = True
        if (scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        req_id_output_token_ids: Dict[str, List[int]] = \
            {req_id: req.output_token_ids \
                for req_id, req in self.requests.items()}

        sampling_metadata = self.input_batch.make_sampling_metadata(
            req_id_output_token_ids, skip_copy)
        return sampling_metadata

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: List[MultiModalKwargs] = []
        req_input_ids: List[Tuple[int, int]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[input_id])
                req_input_ids.append((req_id, input_id))
        batched_mm_inputs = MultiModalKwargs.batch(mm_inputs)
        batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                       device=self.device)

        # Run the encoder.
        # `encoder_outputs` is either of the following:
        # 1. A tensor of shape [num_images, feature_size, hidden_size]
        # in case when feature_size is fixed across all images.
        # 2. A list (length: num_images) of tensors, each of shape
        # [feature_size, hidden_size] in case when the feature size is
        # dynamic depending on input images.
        encoder_outputs = self.model.get_multimodal_embeddings(
            **batched_mm_inputs)

        # Cache the encoder outputs.
        for (req_id, input_id), output in zip(req_input_ids, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = output

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self._get_padded_batch_size(num_scheduled_tokens)
        
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if encoder_outputs:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, encoder_outputs)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        # hidden_states = self.model(
        #     input_ids=input_ids.unsqueeze(0).to(self.device),
        #     positions=self.positions[:num_input_tokens].unsqueeze(0).to(self.device),
        #     kv_caches=self.kv_caches,
        #     attn_metadata=attn_metadata,
        #     inputs_embeds=inputs_embeds.to(self.device) if inputs_embeds is not None else None,
        # ).cpu()
        with set_forward_context(attn_metadata, self.vllm_config, 0):
            hidden_states = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                positions=self.positions[:num_input_tokens].unsqueeze(0).to(self.device),
            ).cpu()

        hidden_states = hidden_states[0, :num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices.cpu()]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_token_ids = sampler_output.sampled_token_ids
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs.cpu()
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    # @torch.no_grad()
    # def execute_model(
    #     self,
    #     scheduler_output: "SchedulerOutput",
    # ) -> ModelRunnerOutput:
    #     self._update_states(scheduler_output)

    #     neuron_input_data = self._prepare_inputs(scheduler_output)
    #     num_reqs = self.input_batch.num_reqs
    #     num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    #     sampled_token_ids = torch.empty(num_reqs, dtype=torch.int32)
    #     # FORWARD.
    #     # selected_token_ids = self.model(neuron_input_data.token_ids,
    #     #                                 neuron_input_data.position_ids,
    #     #                                 neuron_input_data.attn_metadata,
    #     #                                 self.kv_caches)
    #     hidden_states = self.model(neuron_input_data.token_ids,
    #                                     neuron_input_data.position_ids,
    #                                     neuron_input_data.attn_metadata,
    #                                     self.kv_caches).cpu()
    #     logits_indicies = neuron_input_data.logits_indicies.cpu()
    #     hidden_states = hidden_states[:num_scheduled_tokens]
    #     hidden_states = hidden_states[logits_indicies]
    #     logits = self.model.compute_logits(hidden_states, None)
    #     sampling_metadata = self._prepare_sampling(scheduler_output)
    #     sampler_output = self.model.sample(
    #         logits=logits,
    #         sampling_metadata=sampling_metadata,
    #     )

    #     selected_token_ids = sampler_output.sampled_token_ids
    #     # # print(f"first {self.kv_caches[0][0][0][:5]}")
    #     # # NOTE: TPU<>CPU sync happens here.
    #     # # We need to call .cpu() first to avoid recompilation.
    #     # token_ids = selected_token_ids.cpu()
    #     token_ids = selected_token_ids
    #     # # token_ids = argmax_token_ids 
    #     # # sampled_token_ids_list = token_ids.tolist()
    #     # sampled_token_ids_list = token_ids
    #     # # HACK AOYU add squeeze to last dimension
    #     # sampled_token_ids = token_ids
    #     # # sampled_token_ids = token_ids.squeeze(-1)

    #     # # UPDATE REQUEST STATE.
    #     b_seq_start_loc = torch.cumsum(torch.tensor([0] + neuron_input_data.seq_lens[:-1],
    #                                                 dtype=torch.long),dim=0)
    #     # # num_tokens = scheduler_output.num_scheduled_tokens['0']
    #     # # head_size = self.head_size
    #     # # num_kv_heads = self.num_kv_heads

    #     # # kv = torch.empty(1, num_tokens, 2, num_kv_heads, head_size, dtype=torch.float, device=self.device)
    #     # # kv.uniform_(-1,1)
    #     # # key, value = kv.unbind(dim=2)
    #     # # def update_cache(
    #     # #     key: torch.Tensor,
    #     # #     key_cache: torch.Tensor,
    #     # #     slot_mapping: torch.Tensor,
    #     # # ) -> None:
    #     # #     torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)

    #     # #     key = key.flatten(0, 2)
    #     # #     key_cache = key_cache.flatten(0, 2)
    #     # #     key_cache.index_copy_(0, slot_mapping, key)
    #     # # update_cache_callable = torch.compile(update_cache,
    #     # #                                     backend="openxla",
    #     # #                                     fullgraph=False,
    #     # #                                     dynamic=False)
    #     # # key_cache, value_cache = self.kv_caches[0]
    #     # # print(f"first {key_cache[0][:5]}")
    #     # # print('****************************************************************************************************')
    #     # # update_cache_callable(key, key_cache, neuron_input_data.attn_metadata.slot_mapping)
    #     # # print(f"second {key_cache[0][:5]}")
    #     # # print('****************************************************************************************************')

    #     # # update_cache(key, self.kv_caches[0][0], neuron_input_data.attn_metadata.slot_mapping)
    #     # # print(f"third {self.kv_caches[0][0][0][:5]}")
    #     # # print('****************************************************************************************************')

    #     current_output_scheduled_token_offset = 0
    #     for i, req_id in enumerate(
    #             self.input_batch.req_ids[:num_reqs]):
    #         req_state = self.requests[req_id]

    #         seq_len = (req_state.num_computed_tokens +
    #                     scheduler_output.num_scheduled_tokens[req_id])
    #         token_ids_cpu_offset = seq_len + b_seq_start_loc[i]

    #         # # token_id = sampled_token_ids_list[i][0]
    #         # current_output_token_index = current_output_scheduled_token_offset + scheduler_output.num_scheduled_tokens[req_id] - 1
    #         # # token_id = sampled_token_ids_list[current_output_token_index][0]
    #         # token_id = sampled_token_ids_list[current_output_token_index]
    #         # HACK AOYU, assume one token per decoding
    #         token_id = token_ids[i]
    #         self.input_batch.token_ids_cpu[i, token_ids_cpu_offset] = token_id
    #         req_state.output_token_ids.append(token_id)

    #         current_output_scheduled_token_offset = current_output_scheduled_token_offset + scheduler_output.num_scheduled_tokens[req_id]


    #     # output_list = sampled_token_ids.tolist()
    #     output_list = token_ids
    #     if not isinstance(output_list, list):
    #         output_list = [output_list]

    #     return ModelRunnerOutput(
    #         req_ids=self.input_batch.req_ids[:num_reqs],
    #         req_id_to_index=self.input_batch.req_id_to_index,
    #         sampled_token_ids=output_list,
    #         logprob_token_ids_cpu=None,
    #         logprobs_cpu=None,
    #   

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                )
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec
 
    def load_model(self) -> None:
        # TODO(gnovack) - Add memory profiler during model load
        with torch.inference_mode():
            logger.info("Starting to load model %s...", self.model_config.model)
            model = get_model(vllm_config=self.vllm_config).eval().to(self.device)
            self.model = torch.compile(model, backend="openxla", fullgraph=True, dynamic=False)

        # # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # # process, the ranks can be different from the ranks internally assigned
        # # by the xm runtime. Therefore, there is a mismatch in the rank
        # # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # # This is not a problem in linear layers because all-reduce is
        # # rank-agnostic. However, it matters for all-gather as the ranks
        # # determine the order of concatenating the output tensors.
        # # As a workaround, we use the xm's rank assignment only when loading
        # # the embedding weights.

        # # xm_tp_rank = xr.global_ordinal()
        # # with patch(
        # #         "vllm.model_executor.layers.vocab_parallel_embedding."
        # #         "get_tensor_model_parallel_rank",
        # #         return_value=xm_tp_rank):
        # #     model = get_model(vllm_config=self.vllm_config)
        # model = get_model(vllm_config=self.vllm_config)
        # model = model.eval()
        # xm.wait_device_ops()
        # with set_current_vllm_config(self.vllm_config):
        #     self.model = ModelWrapper(model)
        #     # def update_cache(
        #     #     key: torch.Tensor,
        #     #     key_cache: torch.Tensor,
        #     #     slot_mapping: torch.Tensor,
        #     # ) -> None:
        #     #     torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)

        #     #     key = key.flatten(0, 2)
        #     #     key_cache = key_cache.flatten(0, 2)
        #     #     key_cache.index_copy_(0, slot_mapping, key)
        #     # self.update_cache_callable = torch.compile(update_cache,
        #     #                                     backend="openxla",
        #     #                                     fullgraph=False,
        #     #                                     dynamic=False)

    # @torch.inference_mode()
    @torch.no_grad()
    def _dummy_run(self, batch_size: int, seq_len: int,
                   kv_caches: List[torch.Tensor], is_prompt: bool) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        input_ids = torch.zeros((batch_size, seq_len),
                                dtype=torch.int32,
                                device=self.device)
        position_ids = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = None if is_prompt else torch.zeros(
            (batch_size, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        context_lens = None if is_prompt else torch.ones(
            (batch_size, ),
            dtype=torch.int32,
            device=self.device,
        )
        attn_metadata = NeuronAttentionMetadata(
            is_prompt=is_prompt,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # NOTE: There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if is_prompt:
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        # Dummy run.
        self.model(input_ids,
                   position_ids,
                   attn_metadata,
                   kv_caches,
                   is_prompt=is_prompt)

    def profile_run(self) -> None:
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        dummy_kv_caches = [(
            torch.tensor([], dtype=torch.float32, device=self.device),
            torch.tensor([], dtype=torch.float32, device=self.device),
        ) for _ in range(self.num_attn_layers)]

        # Round to multiple of 16.
        seq_len = (self.max_num_tokens + 15) // 16 * 16

        # Run empty forward.
        self._dummy_run(batch_size=1,
                        seq_len=seq_len,
                        kv_caches=dummy_kv_caches,
                        is_prompt=True)

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        # Prefill shapes.
        start = time.perf_counter()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self._dummy_run(batch_size,
                                seq_len,
                                self.kv_caches,
                                is_prompt=True)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)
                if seq_len >= self.model_config.max_model_len:
                    break
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.perf_counter()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Decode shapes.
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self._dummy_run(batch_size,
                            seq_len,
                            self.kv_caches,
                            is_prompt=False)
            xm.wait_device_ops()
            logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("Compilation for decode done in %.2f s.", end - start)

    # def initialize_kv_cache(self, num_blocks: int) -> None:
    #     assert len(self.kv_caches) == 0
    #     kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
    #         num_blocks, self.block_size, self.num_kv_heads, self.head_size)
    #     for _ in range(self.num_attn_layers):
    #         self.kv_caches.append((
    #             torch.zeros(kv_cache_shape,
    #                         dtype=self.kv_cache_dtype,
    #                         device=self.device), 
    #             torch.zeros(kv_cache_shape,
    #                         dtype=self.kv_cache_dtype,
    #                         device=self.device)))

    # def initialize_kv_cache(self, num_blocks: int) -> None:
    #     assert len(self.kv_caches) == 0
    #     self.num_blocks = num_blocks

    #     with torch.inference_mode():
    #         kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
    #             num_blocks + 1, self.block_size, self.num_kv_heads, self.head_size)
    #         for _ in range(self.num_attn_layers):
    #             cache = torch.zeros(kv_cache_shape,
    #                             dtype=self.kv_cache_dtype,
    #                             device='cpu')
    #             self.kv_caches.append(cache.to(self.device))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: Dict[str, torch.Tensor] = {}

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                cache = torch.zeros(kv_cache_shape,
                                dtype=self.kv_cache_dtype,
                                device='cpu')
                kv_caches[layer_name] = cache
            else:
                raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def _get_padded_batch_size(self, batch_size: int) -> Optional[int]:
        return batch_size
        # # TODO: Optimize this?
        # for size in self.neuron_compilation_batch_sizes:
        #     if batch_size <= size:
        #         return size
        # return None



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
