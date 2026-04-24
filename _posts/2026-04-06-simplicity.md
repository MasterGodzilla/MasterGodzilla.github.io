---
title: 'Simplicity: On Paper Writing Styles'
date: 2026-04-06
permalink: /posts/2026/04/simplicity/
tags: [research, writing, AI]
---

I think the best writing style for a paper must be extremely simple. So simple that reviewers reject it for not being innovative enough. Only then can the paper be considered fully revised.

The first time I encountered this idea was when Sora had just come out. I saw Saining Xie write the following about Diffusion Transformer (DiT):

When Bill Peebles and I worked on DiT, we did not chase novelty. We prioritized two things: *simplicity* and *scalability*.

At the time, that sentence hit me hard, but I still did not understand what it really meant.

Two years have passed. In that time I have read a lot of papers, watched all kinds of techniques develop and get applied, and done a few projects myself. Only now do I feel there is something almost Daoist about it: the great way is simple. I probably understand at most one third of it, but here is the part I can explain.

This post discusses why papers should pursue extreme simplicity from three angles: scientific exploration, marketing, and engineering practice, and then tries to explore how to practice it in actual work.

## 1. Scientific Exploration

A paper should pursue simplicity first because science requires rigor and a spirit of investigation.

Suppose one day a researcher proposes an ML method. They add a bunch of engineering tricks, run experiments, see the numbers go up a bit, and very often just send the paper out.

But many questions remain:

Why did the numbers go up?

Which specific trick caused the improvement?

What is the mechanism behind it?

Can the method be reproduced and generalized?

Most people do not investigate carefully. It is like a traditional Chinese medicine prescription: throw in a dozen herbs, call it a compound formula, and in fact have no idea which ingredient is doing the work. Maybe only one molecule in one herb is useful, and the rest are actively harmful. Vague, muddled, half-asleep, and then they call it experience or touch.

For this kind of problem, not figuring it out is extremely harmful; figuring it out is extremely valuable.

First, if you can identify why the numbers went up, you will often find that the original method contains mechanisms A, B, C, and D, but only mechanism A actually matters. At that point, you can usually cut B, C, and D entirely and propose a new method that contains only the core mechanism.

Second, once the mechanism is clear, you can better understand whether it is a targeted method or something with generality. For example, if an architecture improves a language model, is it exploiting a property of language, or can vision use it too?

Finally, from the perspective of black-box optimization, the more methods and variables you have, the more the number and complexity of possible recipes grow exponentially. Considering interactions among methods, the experiments are basically impossible to finish. So the recipe people guess is usually based on intuition, not because it is actually optimal.

To avoid exponential grid search, industry usually uses two kinds of methods:

1. Variable control similar to Shapley value. First guess an optimal recipe, then run comparison experiments by removing or changing each variable, proving in the paper that the whole recipe really is optimal.
2. Start from a base version, add several tricks one by one, and observe whether performance increases consistently.

But both methods have problems, because neither truly accounts for interactions among tricks. So when writing a paper, it is best to cut the number of variables below three. Better still, propose only one improvement.

Back to DiT. DiT deliberately did not chase novelty. It simply combined the mature Vision Transformer (ViT) with the mature Latent Diffusion Model (LDM), and added only AdaLN-Zero. So when they say it both improves performance and can scale up, the conclusion is very credible.

## 2. Marketing

@CPAPCF (Founder of ACLism) once told me that doing research means pursuing impact, and the most important part of impact is the spread of ideas.

An idea that spreads must be simple. People should understand it the moment they hear it. At the same time, it must be extreme enough. Only an idea pushed to the extreme has the force to cut through.

I vaguely remember taking Andrew Ng's online course in high school. He told us to read the ResNet paper and gave us half an hour. I finished it in five minutes. At the time I thought: residual connection is so simple. Shouldn't any random person have been able to think of it?

Years later, looking back, I realized that was completely wrong. I understood it not because I was smart, but because Kaiming He's writing is world-class. He had studied the problem extremely thoroughly and then wrote it in the simplest possible way. The intuition is direct, the logic is tight, and there is not a crack between one sentence and the next. After reading it, you feel it is obvious, as if deep learning was naturally supposed to work this way.

But is reality like that? Is ResNet really that simple and obvious? Of course not. Otherwise, so many earlier deep models would not have collapsed during training, and VGGNet would not have needed to be trained layer by layer.

I once heard, or maybe I imagined, that Kaiming and his coauthors tried many things to stop the model from collapsing during training. For example, why not use Highway Net? Why must the coefficient of the residual branch be 1? But after they found a stable training path, they kept running ablation after ablation, kept deleting things, and left only the most essential mechanism. Then they looked at it and saw that it was quite similar to the concept of residuals. They did not claim novelty for the sake of novelty. They still called it residual. Everyone understood it immediately, and the idea became easy to spread. Only then were they satisfied.

There is also a funny phenomenon around whether writing is simple.

The less innovative a paper is, say I build some new agent system, the more likely it is to adopt a complex writing style and cram in as many innovation points as possible.

Colorful flowcharts, exhaustive experiment reports, a pile of borrowed concepts, all kinds of NP-Hard graph-theoretic structures, and a truckload of benchmarks. Then reviewers can at least give a high score out of respect for the labor.

Calling this carving flowers on shit may be slightly excessive. Calling it melodrama over nothing is absolutely fair.

But if I am doing truly basic research, trying to innovate on some underlying mechanism, then merely getting readers to accept the idea is already hard. I cannot make a complex problem even more complex. I have to pursue simplicity.

As the saying goes, first you read a book from thin to thick, then from thick back to thin.

There is one writing choice worth paying attention to: do you claim the method is new, or do you reuse an existing name to describe it and honestly report where it came from?

If you reuse the name, readers understand it more easily and can connect it to previous research on that technique, giving them more ways to think about the paper. But reviewers may feel there is not enough novelty.

If you do not reuse the name, reviewers may feel it is innovative, but readers have to pay an extra comprehension tax.

Take ResNet. Kaiming He could certainly have said he invented some new thing, called it Highway-something or KaimingNet or whatever, and people would still have accepted it. But he chose to connect it back to the concept of residuals, reducing complexity.

Here is a pretty funny example from nearby.

Interns in our group previously wrote a paper about benchmarking benchmarks. That is not the point; I also think the starting point was a little strange. In it, they used an iterative method similar to PageRank, but not on an arbitrary graph. It was a period-two random walk on a Bipartite Graph.

The paper had a rough journey. It was submitted three times: ICML, NeurIPS, and finally, only recently, ICLR, where it got accepted.

I asked: why did the scores suddenly go up?

They said that in the earlier submissions, they called the algorithm bipartite PageRank. Although reviewers did not write this explicitly, perhaps they saw "PageRank" and thought, ah, not innovative enough. This time they did not mention PageRank anywhere in the paper, so the reviewers did not quite understand it but found it impressive, called it very innovative, and gave high scores.

After hearing this I nearly lost it. I told them: for the camera-ready, change it back. After all, reviewers do not count as human for this purpose, so you can do that to them. But when facing readers, be a little more sincere.

Another question in idea spread is: who is your audience?

If you think your paper will be remembered, you will not want to write only for a tiny subfield. You will want the whole field to read it.

Among paper readers there will be new students, frontline engineers, venture capitalists from outside the field, company founders, even high school teachers. If you make the writing too hard, they will not understand it. So you should write more simply.

In short, the simpler a paper is, the farther its ideas can spread. Like the iPhone in the Steve Jobs era: the design looks minimalist on the surface, subtracting again and again until even a fool can pick it up and use it, while behind it lie countless technical breakthroughs and refinements.

After reading such a paper, readers should not exclaim how exquisite it is. They should think:

Ah, it is this simple? No way, right? Surely nobody actually failed to think of this?

If a paper can make readers sigh like that, I think it is done.

## 3. Engineering Practice

When I was young, I loved watching documentaries about the development of major national engineering projects: airplanes, rockets, high-speed rail, colliders.

They often mentioned a concept called systems engineering.

For example, they would say that every generation of a major project needs preliminary R&D and breakthroughs in key technologies. Once key technical specifications are set, the technology must be frozen. New technologies cannot be added.

Each generation can use at most 30% new technology. The Long March 5, for example, used 70% new technology, so it broke, and they had to debug it for more than two years before fixing it.

I remembered these words from childhood, but I never understood what they meant.

Only recently, after actually doing engineering, listening to interviews about large-model development, and talking with technical people, did I understand the importance of managing complexity.

The logic is actually the same as above. The search space for an optimal recipe grows exponentially with the number of variables. If each generation introduces two or three new technologies, the search space is roughly 2^3, still controllable. If, like Llama 4, you introduce a whole pile at once, and each technique has not been carefully validated through small-scale experiments and ablation, then of course it will collapse.

Of course, some technical risks can be separated.

For example, infra-side system acceleration is often fine. As long as numerical precision is aligned, kernel acceleration and parallelism strategies do not interfere with each other too much. Though honestly, even that is hard to say.

Or take post-training. Since you can try many times, each attempt burns at most tens of thousands of dollars, which is not a big deal. This generation of the model is released, and next update you can change it again. So go ahead and try recipes. Try as many as you want. If it trains successfully, use that checkpoint to synthesize data and merge it into the main model. If it collapses, no problem.

For risks that can be isolated, complexity grows linearly with the number of new technologies, which is acceptable.

Now imagine you are the chief engineer of a large model and have to decide the architecture of the next generation.

Researchers under you bring out their proposals one after another.

A says: I propose simple technique A. The principle is simple, and in practice it is simple and useful.

You: works.

B says: I propose mature solution B. DeepSeek and NVIDIA have both made it work, and so many companies have validated it. It should be fine.

You: also fine.

C says: I saw a paper at a top conference. It says that if you use CDEFGHI, several tricks together, it beats the baseline by a full ten points! So exquisite, so innovative!

You: uh...

B is actually the lowest-risk option. A is also controllable if the ablations are clear. But if you listen to C, the entire generation's tolerance for complexity has been used up by this one technique, or this one lump of techniques, and it will inevitably bring nasty surprises.

Why did Sora succeed with DiT? Similar reason. A very mature ViT plus a very mature LDM. How does that collapse? The remaining complexity budget can then be used to explore other technologies for video generation, such as flexible aspect ratio and the like.

So how should we practice this in paper writing?

First, when you get good experimental results, dig to the root and investigate the mechanism.

Do not be lazy with ablation. Demand extreme simplicity. Cut everything that can be cut.

Whenever you feel pleased with yourself, ask: have I really understood this clearly? Would I dare show this draft to Kaiming He? If not, keep revising.

Of course, this research style takes more energy than the usual approach. Bad work is not worth this treatment.

But the question is: if you already know the current work is bad, then unless it is for checking a box, a grant, or a company performance review, why do it at all?

And if it is good work, are you really content with only a few people seeing it?

I think doing scholarship this way may take only about three times as much effort, but the return is at least a thousandfold.

As for submission, writing this way may indeed increase the chance of rejection. After all, DiT really was rejected by CVPR back then.

But if an extremely simple paper is scientifically rigorous, spreads widely as an idea, is truly used in today's models, and benefits humanity, then what difference does rejection make?
