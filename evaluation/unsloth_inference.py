"""
This module performs inference with the trained LoRA adapter using Unsloth.

Unfortunately, all techniques to save the trained adapter to use it with other libraries (Transformers, vLLM) are
currently not working. (See https://github.com/unslothai/unsloth/issues/2009)

The results saved by this script can be evaluated with the eval.py script.
"""

MODEL_NAME = "anakin87/qwen-scheduler-7b-grpo"

# ! pip install "unsloth==2025.3.19" datasets

import os
import datasets
from unsloth import FastLanguageModel


# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1500,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=False,
    gpu_memory_utilization=0.8,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

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


# Perform inference and save the results
path = f"results/{MODEL_NAME.split('/')[-1]}"
os.makedirs(path, exist_ok=True)

for i, ex in enumerate(ds):
    print(i)
    prompt = ex["prompt"]

    text = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,  # Must add for generation
        tokenize=False,
    )
    res = model.generate(**tokenizer([text], return_tensors="pt").to("cuda"))
    generated = tokenizer.decode(res[0], skip_special_tokens=True).rpartition(
        "assistant\n"
    )[-1]

    with open(f"{path}/{i}.txt", "w") as f:
        f.write(generated)
