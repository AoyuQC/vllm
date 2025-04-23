# SPDX-License-Identifier: Apache-2.0
import torch
import os

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase)
from vllm.platforms import current_platform

if current_platform.is_neuron():
    import torch_xla.core.xla_model as xm


# class NeuronCommunicator(DeviceCommunicatorBase):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.process_group_name = self.unique_name or "default_process_group"
#         # Ensure process groups are registered for XLA operations
#         if current_platform.is_neuron():
#             import torch.distributed as dist
#             if not dist.is_initialized():
#                 # Initialize process group if not already initialized
#                 world_size = self.global_world_size
#                 rank = self.global_rank
#                 dist.init_process_group(
#                     backend="xla",
#                     init_method="env://",
#                     world_size=world_size,
#                     rank=rank,
#                 )

#     def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
#         return xm.all_reduce(xm.REDUCE_SUM, x)
#         # return xm.all_reduce(xm.REDUCE_SUM, x, groups=self.process_group_name)

#     def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
#         assert dim == -1, "Neuron only supports dim=-1 for all-gather."
#         print(f"all_gather device: {x.device}")
#         return xm.all_gather(x, dim=dim)
from torch.distributed import ProcessGroup
from typing import Optional

class NeuronCommunicator(DeviceCommunicatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if xm.is_xla_tensor(x):
            return xm.all_reduce(xm.REDUCE_SUM, x)
        else:
            return torch.ops.vllm.all_reduce(x, group_name=self.unique_name)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        if xm.is_xla_tensor(x):
            return xm.all_gather(x, dim=dim)
        else:
            group = self.cpu_group
            world_size = self.world_size
            if dim < 0:
                # Convert negative dim to positive.
                dim += x.dim()
            input_size = x.size()
            # NOTE: we have to use concat-style all-gather here,
            # stack-style all-gather has compatibility issues with
            # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
            output_size = (input_size[0] * world_size, ) + input_size[1:]
            # Allocate output tensor.
            with torch.inference_mode(False):
                output_tensor = torch.empty(output_size,
                                            dtype=x.dtype,
                                            device=x.device,
                                            requires_grad=False)
                # All-gather.
                torch.distributed.all_gather_into_tensor(output_tensor,
                                                        x,
                                                        group=group)
            # Reshape
            output_tensor = output_tensor.reshape((world_size, ) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                (world_size *
                                                input_size[dim], ) +
                                                input_size[dim + 1:])
            return output_tensor
