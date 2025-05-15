# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD
import enum
import os
from functools import lru_cache
=======
import os
>>>>>>> f39116508 (adding support for neuron tp in v1)
from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class NeuronFramework(enum.Enum):
    TRANSFORMERS_NEURONX = "transformers-neuronx"
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    dispatch_key: str = "XLA"
    supported_quantization: list[str] = ["neuron_quant"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"
    simple_compile_backend: str = "openxla"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        if not use_v1:
            logger.info("Neuron backend is only supported in V1")
        logger.info("Using NKI flash-attention backend.")
        return "vllm.v1.attention.backends.neuron_attn.NeuronAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if envs.VLLM_USE_V1:
            cls.check_and_update_config_v1(vllm_config)
        else:
            cls.check_and_update_config_v0(vllm_config)

        assert (vllm_config.lora_config
                is None), "LoRA is not supported for Neuron backend."

    @classmethod
    def check_and_update_config_v1(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm.v1.worker.neuron_worker.NeuronWorker")

        cache_config = vllm_config.cache_config
        if cache_config:
<<<<<<< HEAD
            # neuron needs block_size = max_model_len
            vllm_config.cache_config.block_size = \
                vllm_config.model_config.max_model_len  # type: ignore
=======
            # neuron V1 only: the preferred block_size is 32
            vllm_config.cache_config.block_size = 32

        model_config = vllm_config.model_config
        if model_config:
            model_config.enforce_eager = False

        compilation_config = vllm_config.compilation_config
        if compilation_config:
            compilation_config.custom_ops = ["silu_and_mul"]
            compilation_config.level = CompilationLevel.DYNAMO_AS_IS
            compilation_config.cudagraph_capture_sizes = [
                vllm_config.scheduler_config.max_num_batched_tokens
                if vllm_config.scheduler_config else 128
            ]

        # Disable functionalization on xla, due to https://github.com/pytorch/xla/issues/8640
        os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
        os.environ["NEURON_CC_FLAGS"] = " -O1 --model-type=transformer --retry_failed_compilation "

    @classmethod
    def check_and_update_config_v0(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm.worker.neuron_worker.NeuronWorker")

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        cache_config = vllm_config.cache_config
        if cache_config:
            # neuron V0 only: needs block_size = max_model_len
            vllm_config.cache_config.block_size = (
                vllm_config.model_config.max_model_len)
>>>>>>> f39116508 (adding support for neuron tp in v1)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if envs.VLLM_USE_V1:
            return "vllm.distributed.device_communicators.neuron_communicator.NeuronCommunicator"  # noqa
        else:
            return Platform.get_device_communicator_cls()

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    @lru_cache
    def is_neuronx_distributed_inference(cls) -> bool:
        try:
            import neuronx_distributed_inference
        except ImportError:
            neuronx_distributed_inference = None
        return neuronx_distributed_inference is not None

    @classmethod
    @lru_cache
    def is_transformers_neuronx(cls) -> bool:
        try:
            import transformers_neuronx
        except ImportError:
            transformers_neuronx = None
        return transformers_neuronx is not None

    def get_neuron_framework_to_use(self):
        """Return the specified framework if corresponding installations are
        available.

        If no framework is specified, use neuronx-distributed-inference by
        default.
        If that's unavailable, check and switch to transformers-neuronx.
        """
        if not self.is_neuron():
            raise AssertionError(
                f"Neuron Framework unavailable for platform: {self}")

        tnx_installed = self.is_transformers_neuronx()
        nxd_installed = self.is_neuronx_distributed_inference()

        specified_framework = os.environ.get("VLLM_NEURON_FRAMEWORK")
        tnx_framework = NeuronFramework.TRANSFORMERS_NEURONX.value
        nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE.value
        if specified_framework == tnx_framework and tnx_installed:
            return self.TRANSFORMERS_NEURONX

        if ((specified_framework == nxd_framework and nxd_installed)
                or (specified_framework is None and nxd_installed)):
            return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

        if specified_framework is None and tnx_installed:
            return NeuronFramework.TRANSFORMERS_NEURONX

        return None

    def use_neuronx_distributed(self):
        """
        Return True if the framework determined in get_neuron_framework_to_use()
        is NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, False otherwise. This
        is used to select the Neuron model framework and framework-specific
        configuration to apply during model compilation.
        """
        nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
        return self.get_neuron_framework_to_use() == nxd_framework

    def use_transformers_neuronx(self):
        """
        Return True if the framework determined in get_neuron_framework_to_use()
        is NeuronFramework.TRANSFORMERS_NEURONX, False otherwise. This is used
        to select the Neuron model framework and framework-specific
        configuration to apply during model compilation.
        """
        return self.get_neuron_framework_to_use(
        ) == NeuronFramework.TRANSFORMERS_NEURONX
