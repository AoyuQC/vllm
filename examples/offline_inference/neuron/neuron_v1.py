# SPDX-License-Identifier: Apache-2.0
import os
import PIL

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"

# # Multi modal data
# # Refer to the HuggingFace repo for the correct format to use
# prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# # Load the image using PIL.Image
# image = PIL.Image.open("demo.jpg")

# # Single prompt inference
# llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

# outputs = llm.generate(
#     [
#         {
#             "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image},
#         },
#         {
#             "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image},
#         }
#     ]
# )

# for o in outputs:
#     generated_text = o.outputs[0].text
#     print(generated_text)

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)

# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    # model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model="TinyLlama/TinyLlama_v1.1_chinese",
    # The neuron backend for V1 is currently experimental.
    # Here, we limit concurrency to 8, while enabling both chunked-prefill
    # and prefix-caching.
    max_num_seqs=8,
    max_num_batched_tokens=128,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
