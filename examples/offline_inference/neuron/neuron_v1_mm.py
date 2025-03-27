# SPDX-License-Identifier: Apache-2.0
import os

from vllm import LLM, SamplingParams
# One image multiple questions
from vllm.assets.image import ImageAsset

os.environ["VLLM_USE_V1"] = "1"

# Multi modal data

model_name = "Qwen/Qwen2-VL-2B-Instruct"

llm = LLM(
    model=model_name,
    max_model_len=1024,
    max_num_seqs=5,
    max_num_batched_tokens=128,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 512 * 28 * 28,
    },
    disable_mm_preprocessor_cache=False,
)

placeholder = "<|image_pad|>"

# Input image and question
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

questions = [
    "What is the content of this image?",
]
# questions = [
#     "What is the content of this image?",
#     "Describe the content of this image in detail.",
#     "What's in the image?",
#     "Where is this image taken?",
# ]

prompts = [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n") for question in questions]
stop_token_ids = None

# We set temperature to 0.2 so that outputs can be different
# even when all prompts are identical when running batch inference.
sampling_params = SamplingParams(temperature=0.2,
                                 max_tokens=64,
                                 stop_token_ids=stop_token_ids)

# Use the same image for all prompts
num_prompts = 1
inputs = [{
    "prompt": prompts[i % len(prompts)],
    "multi_modal_data": {
        "image": image
    },
} for i in range(num_prompts)]

outputs = llm.generate(inputs, sampling_params=sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

# Multiple images multiple questions
