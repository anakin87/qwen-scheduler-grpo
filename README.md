# Qwen Scheduler GRPO 👑 🗓️

I tried to teach a Language Model how to create a schedule from a list of events and priorities.

I used GRPO, meaning the model is trained over time with Reinforcement Learning to reason and maximize
some reward functions. This differs from Supervised Fine-Tuning methods: we are not showing the model any target
completions, only the prompts.

I hope this experiment proves useful to someone!

<!-- TODO: put images and add link to the article -->

# The problem

Given a list of events and priorities, we ask the model to create a schedule that maximizes the total weighted duration.
In this contenxt, priorities just means events with a weight of 2, while normal events have weight of 1.

## Example input

Events:
- Event A (01:27 - 01:42)
- Event B (01:15 - 02:30)
- Event C (15:43 - 17:43)

Priorities:
- Event B

## Example output

```xml
<think>A detailed reasoning</think>
<schedule>
<event>
<name>Event B</name>
<start>01:15</start>
<end>02:30</end>
</event>
<event>
<name>Event C</name>
<start>15:43</start>
<end>17:43</end>
</event>
</schedule>
```

## Why?

Teaching a Language Model something that can be easily and efficiently solved with deterministic programming might 
sound not very useful... That's because it isn't.

But it's 2025, and everyone is trying GRPO with GSM8K or the Countdown Game...

I find it fascinating to make a Language Model learn from just prompts and rewards, not by examples. And I wanted to try
something different.

Choosing a different problem (events scheduling) forced me to think about the problem setting, generate data, choose the base model, design reward functions and and run multiple rounds of training, hoping that my model would learn something.

A fun and rewarding 😄 experience.

## Results

Evaluated on the test set, the model developed some capabilities to reason on this task and create valid schedules, even outperforming a model twice its size.

Still, it struggles with overlapping events, suggesting the reward functions could be improved in that area.

## Repository organization
- [Dataset generation](./dataset_generation/): Script to generate the [Events Scheduling dataset](https://huggingface.co/datasets/anakin87/events-scheduling).
- [📓 GRPO Training Notebook](train_grpo.ipynb)
- [Evaluation](./evaluation/): Inference + evaluation scripts for both the original and trained models.

## Materials
- [Events Scheduling dataset](https://huggingface.co/datasets/anakin87/events-scheduling)
- [Model: anakin87/qwen-scheduler-7b-grpo](https://huggingface.co/anakin87/qwen-scheduler-7b-grpo)
- [Weight and Biases report](https://api.wandb.ai/links/stefanofiorucci/22oryc3v)