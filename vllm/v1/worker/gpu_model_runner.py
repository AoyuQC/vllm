import gc
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, cdiv, is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend,
                                                   FlashAttentionMetadata)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.engine.mm_input_mapper import MMInputMapperClient
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from .model_runner_base import ModelRunnerBase

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class GPUModelRunner(ModelRunnerBase):
    """GPU-specific model runner implementation."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        
        # GPU-specific settings
        self.pin_memory = is_pin_memory_available()
        
        # Compute encoder budgets
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Initialize multimodal support
        self.mm_input_mapper_profiling = MMInputMapperClient(self.model_config)
        self.mm_input_mapper_profiling.use_cache = False

        # Request states
        self.requests: Dict[str, CachedRequestState] = {}

        # CUDA graph settings
        self.use_cuda_graph = (
            self.vllm_config.compilation_config.level == CompilationLevel.PIECEWISE
            and not self.model_config.enforce_eager
        )
        self.cudagraph_batch_sizes = sorted(
            self.vllm_config.compilation_config.cudagraph_capture_sizes,
            reverse=True
        )

    def load_model(self) -> None:
        """Load model into GPU memory."""
        self.model = get_model(self.vllm_config)
        self.model.to(device=self.device, dtype=self.dtype)

    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        self.verify_loaded()
        return self.model

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        """Execute model on GPU."""
        self.verify_loaded()

        # Process input batch
        input_batch = InputBatch.from_scheduler_output(
            scheduler_output,
            self.requests,
            self.model_config,
            self.device
        )

        # Execute model
        with torch.inference_mode():
            output = self._execute_model_with_batch(input_batch)

        return output

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get KV cache specifications for GPU."""
        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        kv_cache_spec: KVCacheSpec = {}
        
        for layer_name, attn_module in forward_ctx.items():
            if not isinstance(attn_module, Attention):
                continue
                
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                )
                
        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize GPU KV cache."""
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )

        kv_caches: Dict[str, torch.Tensor] = {}
        
        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
                    num_blocks, 
                    layer_spec.block_size,
                    layer_spec.num_kv_heads,
                    layer_spec.head_size
                )
                kv_caches[layer_name] = torch.zeros(
                    kv_cache_shape,
                    dtype=layer_spec.dtype,
                    device=self.device
                )
            else:
                raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches
        )

    def _dummy_run(
        self, 
        num_tokens: int,
        dummy_kv_caches: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Perform a dummy forward pass for warmup/profiling."""
        # Create dummy inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=self.device)
        
        # Set forward context
        with set_forward_context(self.vllm_config.compilation_config.static_forward_context):
            # Run forward pass
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=dummy_kv_caches or self.kv_caches,
            )
            
        return hidden_states

    def capture_model(self) -> None:
        """Capture model for CUDA graphs."""
        if not self.use_cuda_graph:
            return

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with graph_capture(device=self.device):
            for num_tokens in self.cudagraph_batch_sizes:
                for _ in range(self.vllm_config.compilation_config.cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens)
                self._dummy_run(num_tokens)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        
        logger.info(
            f"Graph capturing finished in {end_time - start_time:.0f} secs, "
            f"took {cuda_graph_size / (1 << 30):.2f} GiB"
        )

    def _execute_model_with_batch(self, input_batch: InputBatch) -> ModelRunnerOutput:
        """Execute model with prepared input batch."""
        # Implementation details for model execution
        pass
