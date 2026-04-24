---
title: 'How Do You Come Up With a Research Idea That Actually Works?'
date: 2024-07-02
permalink: /posts/2024/07/workable-idea/
tags: [research, AI, ideas]
---

## 1.

The simple method is A + B.

My math teacher once said that research is basically either putting old wine in a new bottle, or new wine in an old bottle.

Ideas that have already been validated on LLMs can usually be copy-pasted directly into CV, audio, medicine, psychology, and so on. That is basically stable. If you can add a little domain expertise on top, you can usually get past the reviewers.

At the same time, this is also the kind of work most likely to have a positive impact on society. Do not feel guilty about it.

## 2.

A better method is to play with models without a specific goal. You need a sense of play.

Take an LLM, first really figure out prompting, then build a pile of API Agents: write a dating copilot, make robots fight traditional Chinese characters, test some bias, mess around with social media accounts, and so on.

You build taste while playing. Then read more related papers, write down what you played with, and there is your top-conference paper.

Purely playing like this was enough to publish papers last year. This year, prompting probably cannot get published anymore. Agents still can, and the opportunities are only increasing.

Half a step further, start playing with fine-tuning. Fine-tune models on all kinds of weird data and observe their behavior. For example, synthesize a bunch of low-resource-language data, or invent a new logical language, play with hallucination, do some RLAIF. Papers like this should be easy to publish this year. Mainly because post-training is in a state where sometimes one scenario can support one paper. There is no end to the things you can do.

After that, you can even consider playing with pretraining and model architecture. But if you are GPU-poor, forget it. This path burns a lot of cards.

## 3.

Turn your "I read a few papers" into "I read a few hundred papers," and that is enough. Though I feel like even when I had just started doing research, I could sometimes come up with ideas that were not bad.

## 4.

For things with theoretical guarantees, sometimes you can predict the result in advance. My recent paper was like this too. At the time I was only unsure whether it could speed things up by 3-10%, but in my heart I knew there would definitely be a speedup. In the end it came out to 5%.

## 5.

After doing research for a while, a person gains a kind of understanding ordinary people do not have.

Take scaling law, for example. Different people at OpenAI arrived at it for different reasons. Dario Amodei, who later founded Anthropic, noticed it while training speech models at Baidu. He found that as the internal state dimension of the LSTM increased, as the number of layers increased, and as the amount of training increased, model performance improved steadily. He could not help wondering: if we pushed scale to the limit, what would happen?

Richard Sutton, author of The Bitter Lesson and one of the fathers of reinforcement learning, realized the meaning of scale around the same time. The difference is that Ilya brought up scaling law the second time he met Hinton as an undergraduate. It was almost like an obsession he was born with, and nobody knows where it came from. So when people call him a "prophet," it comes from many such pieces of seemingly groundless understanding.

Other fields are the same. Shing-Tung Yau wrote in his autobiography that at the beginning of his career, he consciously tried to use partial differential equations to connect topology and geometry. Over the following decades, that idea brought him abundant results.

Once you have this kind of understanding, it is as if you have a useful hammer in your hand and can go looking everywhere for nails. For the OpenAI people, once scaling law was established, the only remaining question was what to scale. Luckily, they eventually found the perfect carrier: large language models.

I have not figured out what my own main line of thought is. But from where I stand now, the thing that looks promising, and that I am trying hard to explore, is this claim:

RL = genetic algorithms = black-box optimization = zeroth-order optimization = dumb-as-hell stuff, so we should look for gradient information everywhere and use it instead.

There may be papers along these lines later. We will see.
