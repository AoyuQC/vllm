from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

# HACK AOYU, set to neuron v1 platform for debuging purpose
class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON_V1
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    supported_quantization: list[str] = ["tpu_int8"]

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.NEURON_ATTN_V1:
            logger.info("Cannot use %s backend on Neuron Device.", selected_backend)
        return _Backend.NEURON_ATTN_V1

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        compilation_config = vllm_config.compilation_config
        if compilation_config.level == CompilationLevel.NO_COMPILATION:
            # TPU does not support NO_COMPILATION
            compilation_config.level = CompilationLevel.DYNAMO_ONCE
        assert compilation_config.level < CompilationLevel.PIECEWISE,\
            "TPU does not support Inductor."

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                parallel_config.worker_cls = \
                    "vllm.worker.multi_step_tpu_worker.MultiStepTPUWorker"
            else:
                parallel_config.worker_cls = "vllm.worker.tpu_worker.TPUWorker"

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False