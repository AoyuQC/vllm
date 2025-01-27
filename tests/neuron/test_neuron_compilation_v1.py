import glob
import os
# Neuron compiler flags
os.environ["NEURON_CC_FLAGS"]= " --model-type=transformer -O1 --internal-hlo2tensorizer-options='--verify-hlo' --retry_failed_compilation "
# Use V1
os.environ["VLLM_USE_V1"]="1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import tempfile

import depyf

from vllm.config import CompilationLevel

# temp_dir = tempfile.mkdtemp()
# print(f"temp dir for compile debug: {temp_dir}")
# with depyf.prepare_debug(temp_dir):
from vllm import LLM, SamplingParams

# prompts = [
#     "Girl",
# ]
# answers = [
#     " or, through inaction, allow a human being to come to harm.",
# ]
# # prompts = [
# #     "A robot may not injure a human being",
# #     "It is only with the heart that one can see rightly;",
# #     "The greatest glory in living lies not in never falling,",
# #     "This is a test",
# # ]
# # answers = [
# #     " or, through inaction, allow a human being to come to harm.",
# #     " what is essential is invisible to the eye.",
# # ]
# N = 1
# # Currently, top-p sampling is disabled. `top_p` should be 1.0.
# sampling_params = SamplingParams(temperature=0.7,
#                                     top_p=1.0,
#                                     n=N,
#                                     max_tokens=16)

# # Set `enforce_eager=True` to avoid ahead-of-time compilation.
# # In real workloads, `enforace_eager` should be `False`.

# # disable custom dispatcher, let Dynamo takes over
# # all the control
# llm = LLM(model="TinyLlama/TinyLlama_v1.1",
#             enforce_eager=True,
#             block_size=128)
#             # compilation_config={"level": CompilationLevel.DYNAMO_AS_IS})
# outputs = llm.generate(prompts, sampling_params)
# for output, answer in zip(outputs, answers):
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#     assert generated_text.startswith(answer)

# compiled_code = sorted(
#     glob.glob(os.path.join(temp_dir, "__transformed_code*.py")))

# # we should only trigger Dynamo compilation three times:
# # one for the profiling phase without kv cache
# # one for the prefill phase with symbolic shapes
# # one for the decode phase with symbolic shapes
# # and later calls should not trigger Dynamo compilation again.
# # NOTE: it might still trigger XLA compilation.

# # check we have three compiled code
# # this is the assumption when we use the custom dispatcher
# assert len(compiled_code) == 3

# # check all the compilations are as expected
# compiled_fn = sorted(
#     glob.glob(os.path.join(temp_dir, "__compiled_fn*Captured*.py")))

# # the first compilation is the profiling phase,
# # it should not have any kv cache
# with open(compiled_fn[0]) as f:
#     content = f.read()
#     assert "kv_caches" not in content

# # the second compilation is the prefill phase,
# # it should have kv cache and the flash_attention op
# with open(compiled_fn[1]) as f:
#     content = f.read()
#     assert "kv_caches" in content and "torch.ops.xla.flash_attention" in content

# # the third compilation is the decode phase,
# # it should have kv cache and the paged_attention op
# with open(compiled_fn[2]) as f:
#     content = f.read()
#     assert "kv_caches" in content and "torch.ops.xla.paged_attention" in content

import os
os.environ["VLLM_USE_V1"] = "1"
 
from vllm import LLM, SamplingParams
 
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=1)

# Create an LLM.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    max_model_len=128,
    block_size=128,
    device="neuron",
    enforce_eager=True,
    tensor_parallel_size=1,
    disable_async_output_proc=True,
    enable_chunked_prefill=True,
    worker_cls="vllm.v1.worker.neuron_worker.NeuronWorker"
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
