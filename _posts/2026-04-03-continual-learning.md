---
title: 'Why AI May Not Need Continual Learning'
date: 2026-04-03
permalink: /posts/2026/04/continual-learning/
tags: [AI, LLM, continual learning]
---

The problem with biomimetics is that the way carbon-based organisms implement something is not necessarily the optimal design for silicon-based organisms.

Take flight, for example. Birds get lift and forward thrust by flapping their wings. The control requirements are so high that even today humans still cannot really build aircraft like that.

But humans use fixed wings plus propellers, and get speeds far beyond what birds can do.

Something similar may be happening again.

For AI, perhaps the biggest unsolved mystery right now is how to achieve continual learning.

Human memory and large language models are completely different:

Human memory is sequential, while the data baked into a large language model during pretraining has no real before-and-after order.

Humans have to learn knowledge in order: first addition, subtraction, multiplication, and division; then elementary algebra and plane geometry; and finally calculus. But the training samples for a large language model must be IID, independent and identically distributed. In other words, the sample order has to be completely shuffled.

Human skills may change dynamically during training. A large language model can only wait until the next round of post-training to adjust.

And so on.

According to Dario in a recent interview, if we sort learning and memory by timescale, human learning and evolution contain multiple modes: very long-term genetic evolution, medium- and long-term continual learning, and of course short-term working memory.

But the outer-loop pretraining of large language models and the inner-loop in-context learning do not strictly correspond to those learning modes. If you drew a diagram, it would probably look something like that.

So why is this the case? Why can't we implement continual learning?

This post tries to propose the reverse view: it may be precisely because we did not take the continual-learning path that we managed to reach something close to human-level intelligence ten years early.

First, how much compute does the human brain have?

This question is hard to answer. Maybe it is equivalent to 10,000 H100s, maybe 100,000. I do not really know.

But humans have one feature that is completely different from AI: our training compute and inference compute are equally large.

This may mean that every person's brain makes the largest possible adjustments according to that person's own situation. Neurons are adjusted at any time; ways of thinking are reviewed at any time. One brain per person, training and inference fused together.

In that setting, continual learning is the best solution.

AI, however, does not have the same compute budget for training and inference.

During training, it is possible to mobilize thousands, tens of thousands, or even hundreds of thousands of H100s.

During inference, though, GPUs are usually serving customers in parallel. For example, with DS V3, more than a hundred cards doing inference together can reach 2,000 tokens per second per card. But once divided across users, each user gets at most about 40 tokens per second, which is roughly equivalent to each person using only 1/50 of a card during inference.

The ratio between inference compute and training compute differs by something like hundreds of thousands of times.

In this situation, using the KV cache as "fast parameters" for personalization is already one of the bigger compensations we can make for the absence of continual learning.

Why does it work this way?

One of the biggest advantages artificial intelligence has over carbon-based intelligence is that model parameters can be copied.

As a human, I cannot directly copy my brain and stuff it into someone else's brain. But a large language model can.

Being able to do something means a constraint has been lifted, and that must be a good thing. But how exactly does this good thing show up?

I think the benefit is precisely this: when per-person inference compute is only one hundred-thousandth or one millionth of a human brain, we can still economically amortize the cost of training the model across tens of millions or hundreds of millions of users. Through an alternative method called pretraining, it can achieve superhuman intelligence at low cost.

This difference may also be why it succeeded. It may not be as adaptable as humans, but the breadth of its knowledge is genuinely beyond any individual human.

Now imagine giving every user 10,000 GPUs, with both training and inference running on those same 10,000 GPUs. What would happen?

First, the parameter count would definitely go up.

Chinchilla optimal says that if total training compute is fixed, the parameter count and data volume should be about 1:20 for the best model performance. After using Muon, this number seems to become roughly 1:8.

But normal people today do not train models this way, because you cannot consider only training cost and ignore inference cost.

The more common approach is to use a smaller model and train it on far more data than Chinchilla optimal. This is not the strongest model you could train under a fixed compute budget, but it is competitive in the market. Users can only use it if it is cheap enough.

So with 10,000 GPUs, scaling the parameter count to 100T, plus a sparsification scheme even more aggressive than MoE, should not be much of a problem for one person.

Second, training would not have to use next-token prediction. It could all use something like RL pretrain instead.

Yann LeCun once described the famous cake analogy: training should be mainly unsupervised pretraining, with reinforcement learning only as the final decorative layer, because reinforcement learning does so many rollouts and in the end gets only one bit of information.

But if we have far more compute than data, does that still matter? It seems like it might not.

For every text we read, we could do a huge amount of analysis to decide whether to absorb it, how to absorb it, whether to filter it out, or whether to use it for data augmentation. All of that would be possible.

Third, in order to compress information, deep learning models often use superposition to squeeze lots of information into one vector space, which makes things like memory editing difficult. If there are that many parameters, could we just turn all of it into sparse memory retrieval? It seems possible too.

Of course, the economics absolutely do not allow us to do this.

In the future, if every real person averages 10,000 GPUs, will we use human-like continual learning?

Hard to say. Maybe it depends on how much an application needs personalization. But it is also possible that people will discover that the extra hundreds of thousands of times more compute is still better pooled together for shared learning.

For example, if we want to build autonomous driving, we could do it like humans: use a huge amount of redundant compute to learn on site, watching the road while driving, which would be L5.

But we could also memorize all the maps of all city roads during pretraining, leaving the compute in the car to make only small adjustments based on road conditions, which would be L4.

The latter might require one hundredth or one thousandth as much compute per car, though it would definitely be less adaptable.

Fine, road traffic is fairly common. Everyone shares the same road network, so maybe pretraining can handle it.

But what about humanoid robots?

For example, if I want one to clean my home, it has to memorize what is in the house, what is safe and what is dangerous, where things are placed, and what my habits are.

This seems like it should still depend on continual learning.

But if I am a robot manufacturer, I could simply use simulation or real-world scenes, build ten million common home layouts, and let the robot learn most of what it needs in advance. When it arrives, because it has seen similar cases before, it can directly generalize in context. That also seems workable?

Both of these dumb methods may be able to brute-force abilities that, in principle, should have been achieved through continual learning. Does that mean inference-side compute can be smaller by thousands of times?

In short, the fact that continual learning has not been implemented is not necessarily a bad thing.

Maybe we should look at it from another angle: because we exploited the copyability of model parameters and performed shared learning, spreading the cost of learning across many users, we were able to achieve near-human performance while using per-person inference compute that is tens of thousands, hundreds of thousands, or even millions of times smaller.
