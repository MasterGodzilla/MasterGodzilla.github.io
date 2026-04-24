---
title: 'In 2025, Has Post-Training Become More Important Than Pretraining?'
date: 2025-07-03
permalink: /posts/2025/07/posttrain-vs-pretrain/
tags: [AI, LLM, training]
---

Hard to say. The more I look at post-training, the more unnatural it feels.

Take Kimi K2's success. According to their own account, a large part of it came from agent pretraining, rather than entirely, or even mainly, from post-training.

They built a large-scale synthetic pipeline for agent trajectory data. Tool calls, MCP, search, and all kinds of other things were synthesized offline at very large scale, then used for pretraining.

So, as a non-reasoning model, its SWE-Bench score reached 65.8%, higher than most reasoning models.

SWE-Bench is a software-development benchmark: it takes a bunch of GitHub repo issues, gives the LLM a set of tools, such as a terminal or file manager, and asks it to fix the repo.

Or maybe SWE-Bench was never really testing pure coding ability in the first place, but tool-use ability?

Something similar happened in the first half of the year, when everyone publishing RL papers was tuning Qwen.

One method would make Qwen 2.5 improve, but would not work on Llama 3. The reason, as people generally saw it, was that Qwen 2.5's pretraining text had already mixed in several trillion tokens of reasoning data, all synthesized by Qwen 2 Math and Qwen 2 Coder.

So it already knew how to reason. The only thing it had not learned was that line: "wait, perhaps I should try another approach."

Of course post-training can definitely push scores up. But these abilities seem to become truly learned by the next-generation model only when they are turned, through large-scale synthetic data, into pretraining samples for that next generation.

You could put it this way: post-training raises the ceiling, while pretraining is what makes the model thoroughly master something.

If we look at this from the perspective of statistical samples, RL post-training is extremely unnatural.

Generally speaking, for a sample to be meaningful, it usually needs to be randomly drawn, independent and identically distributed.

Pretraining data is shuffled, so it roughly fits that standard.

But RL training data is not like that. If a model goes from weak to strong during training, then the data it sees is not uniformly distributed.

What exactly this affects is also hard to say. It just feels weird.
