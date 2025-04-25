"""
This module performs inference using vLLM.

I used this to generate the results on the test set for the original model (Qwen2.5-Coder-7B-Instruct) and
Qwen2.5-Coder-14B-Instruct.

The results on the test set of the model trained with Unsloth are not generated with this script, due to Unsloth
bugs. See unsloth_inference.py for more details.

The results saved by this script can be evaluated with the eval.py script.
"""

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or "Qwen/Qwen2.5-Coder-14B-Instruct"

# ! pip install vllm datasets

import os
import datasets
from vllm import LLM, SamplingParams


# Load the model
llm = LLM(model=MODEL_NAME, max_model_len=2048, gpu_memory_utilization=0.9)


# Prepare the dataset
SYSTEM_PROMPT = """You are a precise event scheduler.
1. First, reason through the problem inside <think> and </think> tags. Here you can create drafts, compare alternatives, and check for mistakes.
2. When confident, output the final schedule inside <schedule> and </schedule> tags. Your schedule must strictly follow the rules provided by the user."""

USER_PROMPT = """Task: create an optimized schedule based on the given events.

Rules:
- The schedule MUST be in strict chronological order. Do NOT place priority events earlier unless their actual start time is earlier.
- Event start and end times are ABSOLUTE. NEVER change, shorten, adjust, or split them.
- Priority events (weight = 2) carry more weight than normal events (weight = 1), but they MUST still respect chronological order.
- Maximize the sum of weighted event durations.
- No overlaps allowed. In conflicts, include the event with the higher weighted time.
- Some events may be excluded if needed to meet these rules.


You must use this format:  

<think>...</think>
<schedule>
<event>
<name>...</name>
<start>...</start>
<end>...</end>
</event>
...
</schedule>

---

"""

ds = datasets.load_dataset("anakin87/events-scheduling", split="test")

ds = ds.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + x["prompt"]},
        ]
    }
)

conversations = [ex["prompt"] for ex in ds]


# Perform inference and save the results
outputs = llm.chat(conversations, sampling_params=SamplingParams(max_tokens=2000))

path = f"results/{MODEL_NAME.split('/')[-1]}"
os.makedirs(path, exist_ok=True)

for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text

    with open(f"{path}/{i}.txt", "w") as f:
        f.write(generated_text)
