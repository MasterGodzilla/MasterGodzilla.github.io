---
title: 'How to Train 100B+ Large Models on a Single GPU'
date: 2026-04-05
permalink: /posts/2026/04/megatrain/
tags: [AI, LLM, training]
---

Small GPU, huge model. The GPU-poor are about to rise up!

Today, we are officially releasing the MegaTrain training framework. It can train 100B+ large models on a single GPU, in full precision, with full parameters, without slowing down.

The config follows the Llama Factory format, so it works out of the box. The only thing to watch out for is that the larger your batch size, the faster it runs. Some of you may need to change old habits.

I hope this helps everyone escape the pain of being GPU-poor.

GitHub: MegaTrain  
github.com/DLYuanGod/MegaTrain

MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU  
arxiv.org/abs/2604.05091

On a single H200, training gpt-oss 120b. The 30% MFU number is with basically unoptimized speed; if needed, we can tune the kernels later.

To be honest, those of us doing AI must have skipped quite a few classes when we were young.

When we studied math, besides writing "assume the model satisfies such-and-such conditions," we did not really learn rigorous proofs or the abstraction of formal logic. So whenever we start writing, it often turns into: "A page full of nonsense, a chain of inequalities. Ask the reviewer what they think, and they still say reject."

When we studied CS, maybe we skipped fewer classes, but classics like Computer Architecture were still mostly muddled through. Why do we need different floating-point formats? Why do we need a memory hierarchy? How do we design parallel algorithms? We did not care. We are used to Python: create a tensor, multiply a matrix, and AGI is here. Who cares how the lower layers actually work?

Take LLM training, for example.

Most people, having suffered through CUDA Out of Memory, at least know how much memory their GPU has.

If you do a quick calculation, you may realize: I am training a 7B model. Parameters take 2x in bf16, gradients take 2x in bf16, and Adam's momentum and variance add another fp32 * 2. At minimum, that is 12x the parameter count in memory, or 96GB. Add activations and all the other odds and ends, and it just fills an H200. Any larger and you cannot train it. The batch size can only be a few thousand tokens.

But above HBM (High Bandwidth Memory), there is SRAM, meaning L2/L3 cache. Tri Dao already gave us a make-up lesson on this in FlashAttention. Below HBM, there is system memory on the CPU side, DDR/LPDDR, plus NVMe SSD solid-state storage. What can they do? We are not too sure, so we treat them like useless accessories and leave them off to the side.

I was one of those people too. I often used my math background as an excuse for not properly learning computer systems. To be honest, I was not that good at math either.

But every day I suffered because I could not borrow enough GPUs and could not do post-training for large models. Then Qwen came along with supernatural distillation. The 4B small model works amazingly if you do not train it, and collapses the moment you train it. It cannot even do tool calls anymore. So all I could do was complain every day about the rich people at OpenAI: with all those Blackwells, they still will not let me use any. How is that "Open"?

So I ran small-scale experiments, either fine-tuning or GRPO, and scraped together papers to submit.

Then the reviews came back. One by one, they all said: yours is too small.

At that moment, the frustration really makes you want to shout: "Are you here just to mess with me?"

But then I think about it. When I am a reviewer, I also like saying other people's models are too small.

After all, post-training research studies emergent capabilities. If the model is too small, the experiment really is meaningless. Some emergent capabilities simply will not show up.

So when the rejection arrives and months of work go down the drain, what else can you do besides feel wronged and angry?

And yet all this suffering comes from not understanding the memory hierarchy. It is one of the most important ideas in computer systems, and they taught it in class when we were young.

Humans have designed all kinds of storage. From fast to slow, expensive to cheap, small to large, each type is implemented differently and serves a different purpose.

To make full use of all these types of storage, people usually organize them into a hierarchy.

For example, on a GPU, each compute unit has registers, only a few KB.

Nearby there is SRAM, static random-access memory, usually only tens to around a hundred MB, but the bandwidth can reach tens of TB/s. It is also expensive, thousands of dollars per GB.

Further out is GPU memory, which everyone is more familiar with.

Common cards use GDDR: tens of GB, roughly 1TB/s of bandwidth, and before the price spikes you could buy it for about $2/GB. This is what gaming cards carry.

Or you can use HBM, High Bandwidth Memory, which can exceed 100GB. Roughly speaking, it is high-bandwidth DRAM stacked in 12 to 16 layers depending on the version. This is much more expensive: about $25/GB.

On the host side, there is system memory, which some people call CPU memory. Capacity can reach several TB, but bandwidth drops by an order of magnitude.

Further out, there are NVMe SSDs, the so-called solid-state drives. Capacity is measured in tens of TB, and one GB costs only a few cents.

How should we use all this memory? Usually, we arrange information by how often it is used.

It is like arranging an office.

The thing you are using right now is in your hand.

The things you use often are on the desk, reachable at any moment.

The things you use less often go into the bookcase or onto a shelf. You need to walk a few steps, but it is still convenient.

As for the things you may not use even once in several years, those get packed up and sent to the warehouse.

Now that the memory hierarchy lesson is over, we can finally introduce MegaTrain.

Once you understand the great idea of the memory hierarchy, every design choice in MegaTrain becomes obvious.

To be clear, all the code was written by my collaborator zhengqing. I only helped out and did the promotion and such. He is a legend in our group: not from a so-called top school, but absurdly good at code. If there is a chance, someone really should interview him about his legendary story.

First, as mentioned above, LLM training uses three kinds of memory. Persistent storage, including parameters, gradients, and optimizer states; activations, meaning the intermediate states stored for backpropagation; and some other memory use.

Persistent storage grows with model size. Together it is 12x the parameter count. This is the main memory problem we handle.

For the model parameters, gradients, and optimizer states, we put all of them in host memory. The GPU is used only as a temporary compute engine, or you can think of it as a higher-level cache, rather than the place where all information is stored.

Then the parameters do not need to stay on the card all the time. We transfer whichever layer is needed.

During this process, gradients are handled similarly. Going backward, once the gradient for a layer is computed, it is transferred back down.

As for optimizer states, we follow DeepSpeed and do not upload them at all. They are computed entirely on the CPU.

Why? First, optimizer states take 8x memory. There is absolutely no need to trigger an I/O round for them.

If you use Adam, the gradient update is just a string of additions, subtractions, multiplications, and divisions. There is no matrix multiplication involved.

Of course, the clever kids will ask: what about Muon?

To be honest, we have not figured that out yet. If it is Muon without momentum, that is still manageable: just do the Newton-Schulz step before transferring down. With momentum, you need to add another 2x GPU upload for the momentum.

The figure shows the forward, backward, and recomputation design inside one block. Subscripts indicate the layer index inside the block. F is forward compute, W is weight, and G is gradient. At the beginning, the GPU has the weights for layer 0. While it computes, the weights for layer 1 start uploading, and so on. After the forward pass finishes and backpropagation begins, we upload the weights for layer 1 again to complete recomputation inside the block. Once recomputation is done, gradients for layers 3, 2, 1, and 0 are transferred down in sequence.

Another problem is that CPU-GPU communication is much slower. For example, an H200 over PCIe seems to be only around 128GB/s, far slower than HBM at 4.8TB/s. Wouldn't the I/O cost be huge?

As mentioned earlier, persistent storage only changes with parameter count. It has nothing to do with batch size.

So if I make the batch size extremely large, say hundreds of thousands or even millions of tokens, then the I/O cost amortized over each token becomes very small.

Because of this design, we also implemented single-GPU training for 512k-long contexts with a 7B model. With normal FSDP plus CP and all that, you would need at least 64 GPUs, right?

With that done, we move on to activations.

We decided not to transfer activations down, because this number grows with batch size. In practice, transferring or not transferring can both work. We just have not optimized it well enough, and transferring them hurts speed.

So how do we limit their growth? Very aggressive recomputation.

We basically recompute every few layers, which greatly reduces memory use. Of course, recomputation this aggressive adds extra compute cost and makes things a little slower, but I assume everyone here is GPU-poor and will not mind.

The remaining optimizations are mainly about overlapping communication and computation.

For example, zhengqing wrote a lot of new memory-management tricks, reserving a region on both the CPU and the GPU.

On the GPU, it is called a buffer. It stores at most two layers, so while the current layer is computing, the next layer is already being transferred. No waiting is needed.

On the CPU, it is called a gradient slab, also used to reserve space ahead of time for gradient transfer. The GPU can use Direct Memory Access (DMA).

Another change is that normal PyTorch computation needs an autograd graph, which records the entire backpropagation path. It is complicated and creates all kinds of scheduling inconvenience.

We changed this to keep only a one-layer template, then bind weights on the fly after they are transferred up, since most models have very similar layers. To be honest, I did not understand this part either.

This post introduced the design ideas behind MegaTrain.

In the future, we plan to keep maintaining MegaTrain so that all local training can break free from memory constraints. Please send us your requests and feedback.

At the same time, we sincerely welcome everyone to help build this open-source library. If you are interested, feel free to DM.

Here are the projects we plan to support soon:

- RL/DPO
- Single-machine multi-GPU training. We may not support multi-node training.
- Dedicated MoE optimizations
- Diffusion/video/image generation
- Muon/FlashOptimizer

Thank you, everyone!
