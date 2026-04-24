---
title: 'Pseudoscience'
date: 2025-07-05
permalink: /posts/2025/07/ai-pseudoscience/
tags: [AI, math, optimization]
---

I generally call artificial intelligence pseudo-science, pseudo-math, and pseudo-engineering: a trinity.

Pseudo-science: Chinese-medicine-style empirical science.

Pseudo-math: at most around the mathematical difficulty of stochastic differential equations.

Pseudo-engineering: we write code and do not even write unit tests.

Here are a few of the mathematically hardest directions in AI, at least the ones I have heard of, and they are actually all very simple.

First, large models themselves. Nobody can really explain them. Most of the techniques people use are just the most basic dynamical-systems and PDE ideas, such as neural tangent kernels and tensor programs, and they do not go beyond nineteenth-century tools.

Second, diffusion models. They barely use stochastic differential equations, but with things like score matching, you can basically route around them. Some people also work on diffusion over non-Euclidean spaces, which uses basic Riemannian geometry.

Third, reinforcement learning. It uses some simple statistical methods for variance reduction.

Fourth, optimization. But optimization mainly studies convex optimization, while neural networks are nonconvex and cannot really be analyzed. Another direction is stochastic optimization. I have heard that quite a bit of it uses martingales, but I do not understand that.

Then information theory and optimal transport theory show up from time to time.

In short, the math used in artificial intelligence is not hard, and the harder something is, the less useful it tends to be. At its deepest, it is about 1930s mathematics, still very far from the frontier.

More than thirty comments appeared in no time, with experts from all over showing up, which honestly scared me.

Let me casually say a few things about why nonconvex problems cannot be analyzed. Please just take them casually. If anything is wrong, please correct me or teach me; thanks in advance.

I definitely do not understand this, but I also have not seen anyone who understands all of it at a macro level. Everyone is like the blind men touching the elephant. What follows is just hearsay from people who have touched a few different parts, repeated secondhand. I am sure some people have studied one part extremely thoroughly, but it is still hard to imagine what the whole elephant looks like.

The problems studied by traditional nonlinear continuous optimization methods are usually traditional statistical models.

For normal pre-2014 models, basically 99% of them were convex. If something was not convex, people had to make it convex, for example by changing L0 regularization into lasso-style L1.

At the same time, the number of model parameters was not large, so you could use second-order methods such as LBFGS. I do not understand this either; I am just talking nonsense.

The biggest advantage of this kind of method is scale invariance. That is, if two dimensions differ by a huge multiplicative factor, the parameters will not immediately blow up.

The downside is that you have to store second-order information like a Hessian matrix, and the quadratic storage cost is simply impossible to fit.

Neural networks, however, have three features that make analysis extremely complicated:

First, the number of parameters is too large, so you can only use first-order methods. According to optimization theory, scale issues should make it very easy for them not to converge. Add gradient explosion and vanishing on top, and you should not be able to get an optimal solution at all. VGGNet did indeed run into this problem back then.

Solving this problem was Kaiming He's main contribution at the time. The specific method was to ensure, through model design, that gradient magnitudes stayed consistent, preferably with each dimension differing only by a small factor. This included:

normalization

Kaiming initialization

residual connections

With those three tools, the scales were basically aligned.

Second, mini-batch training. The model is not trained on the whole dataset each time, but batch by batch. On one hand, this introduces more variance. On the other hand, more iterations bring higher efficiency. This means it has to be analyzed with stochastic optimization methods, and I do not understand the details.

Third, nonconvexity.

When I took Andrew Ng's course in high school, this was how he explained why nonconvex optimization does not get stuck in local minima:

Suppose the model has N parameters. Then the loss landscape has dimension N.

A local minimum means the Hessian at that point is positive semidefinite. In other words, every dimension is "curving upward," so together they form a pit. If each dimension is independent, then the probability that all of them curve upward is p to the Nth power, which is extremely low.

So in reality, you only need to worry about saddle points and plateaus, meaning regions where the local gradient is very flat.

And because SGD is stochastic, the model parameters wander around and can jump out of saddle points.

I repeated this explanation to the older Hungarian professor in my optimization class. He said the assumption was wrong: why should the dimensions of the loss landscape be independent of one another?

Of course, I am just saying all this casually. What I want to express is that from the optimization point of view, neural networks should not converge. They should fly off. So it is very hard to give theoretical guarantees.

To give theoretical guarantees, you need to understand the properties of neural networks themselves very deeply, including over-parameterization. Only then can you make stronger assumptions before writing the proof and obtain a bound tight enough to mean anything.

This is probably why experts who purely understand optimization cannot produce an analysis of neural network convergence.
