---
title: 'The Business Model of LLMs (Part 1): The Learning Curve'
date: 2026-04-30
permalink: /posts/2026/04/business-model-llms/
tags: [AI, LLM, business]
---

Ever since ChatGPT was released, I’ve been thinking about the business logic and competitive dynamics of large language models.
I’ve struggled to find a good framework for it.
A few days ago, DeepSeek V4 came out.
I wrote a long post critiquing some of V4’s design decisions.
And suddenly, three years of unresolved business questions clicked into place.

This series will explore the business models, competitive dynamics, and pricing strategies of LLMs.
I’ll also make some predictions that readers can test against in the future.

So, how should we understand the competitive logic of LLMs?
I believe we can apply the logic of the semiconductor industry.
Training a model is like R&D in semiconductors.
Inference is like manufacturing.

How does the semiconductor industry determine winners and losers?
The core principle is the learning curve.
I briefly touched on this in my V4 post.
Now I’ll explain it fully.

The logic of the learning curve is simple.
The more you do something, the better you get at it.
But the real competitive implication is something fewer people think about.
Because the more you do it, the better you get.
And the better you get, the better your models and products become.

Meanwhile, costs also come down.
So users prefer your product.
More users means more revenue.
More revenue means more R&D funding.
Which lets you develop the next, even better product.
A virtuous cycle.

Maintain that cycle for decades, and competitors drop off one by one.
They lose the ability to compete in the next generation.

Over time, specialization deepens.
The number of competitors naturally shrinks.
Because each generation costs more to develop.
Capital and talent concentrate into fewer hands.

Once most companies realize that no matter how much they spend on R&D, they can’t win meaningful market share or amortize those costs, they naturally exit.

Take chip manufacturing.
When TSMC was founded in 1987, the dedicated foundry model barely existed.
Most semiconductor companies designed and manufactured their own chips.
So there were many capable manufacturers — possibly hundreds.

But as process nodes shrank and technical difficulty increased, by around 2000, only about a dozen could produce advanced chips.
By 2010, that number fell to four or five.

TSMC mass-produced 7nm in 2018, then introduced EUV at the N7+ node.
At that point, only Samsung and Intel still had a shot at keeping up.

A few more years passed.
At the 2nm node, the fixed investment for each generation exceeds $30 billion.
Even Intel and Samsung couldn’t sustain it.
And so we arrive at today’s TSMC-dominant landscape.

This pattern applies to virtually all mass-market products.
GPUs, for example, are dominated by NVIDIA.
TPUs and AMD take second and third place.
And anything behind AMD is essentially negligible.

Gross margins also vary by rank.
NVIDIA can charge margins of up to 70%.
The second-place player might get 40% to 50%.
From third place onward, making GPUs becomes a losing proposition.
The volume is too small to cover R&D costs.

Memory follows the same pattern.
So does industrial software.
Many industries do.

I believe LLMs will follow semiconductor dynamics exactly.
That means capturing the largest market share is a matter of life and death for any company.

If your revenue is large enough, you can reinvest a portion of it.
When your investment exceeds your rivals’, your models train better and your inference costs go lower.

So more people want to use your model.
Revenue grows.
And the cycle repeats.
If your market is too small, you can’t raise enough money to fund the next generation.
You gradually fall behind.

This should be obvious.
Yet, myself included, most people are distracted by other factors.
For example, in the near term, LLM R&D is still mostly funded by external financing.
Equity, debt, or whatever form it takes.
But if LLMs are to become a sustainable business, they must eventually transition to a stage where revenue — specifically, positive cash flow from inference — funds R&D.

If you believe in AGI, and in its potential to generate enormous wealth for humanity, then both the returns and the required investment will dwarf most existing industries.
This industry cannot survive on perpetual external subsidies.

Another reason this pattern isn’t obvious is that, for the past few years, investment size didn’t directly determine model quality.

Before 2022, Google far exceeded OpenAI in compute, talent, and resources.
But OpenAI had better organization and tighter focus.
The decision-maker coordinated directly and was deeply technical.
He mobilized the entire company around a single mission.
And achieved far more than Google.

Google’s LLM research remained largely academic.
Resources and talent were scattered.
Multiple teams built similar things.
But no single team had the decisive compute to pull off something like GPT-4.
That was Google’s failure.

After 2022 — in 2023, 2024, and 2025 — we also saw many large companies fail to build good models despite heavy investment.
Meanwhile, smaller companies built better models through better organizational culture.

But a good organizational culture won’t remain a scarce resource forever.
Talent moves.
Good culture spreads.

By now, many companies have found the organizational structure needed to develop LLMs.
Can anyone really claim OpenAI’s culture is ten times better than Kimi’s?
I think that gap has largely closed.

In the short term, Chinese startups can indeed train models only three to six months behind the frontier using roughly one-thirtieth the compute of their American counterparts.
To close that gap, Chinese companies can also use distillation and other methods to gradually approach frontier models.

I don’t know whether that trend can persist.
Someday, training a frontier model may require hundreds of millions, even billions, of dollars for a single pre-training run.
Unlike today, when Kimi K2 was reportedly trained for just $8 million.

Of course, maybe Chinese researchers really are that much smarter.
But holding that variable constant — say, looking only at competition between two Chinese companies — the one that spends more will likely train the better model.

In the long run, when two companies have similar culture and organization, the deciding factor is still R&D investment.
And the most important input in R&D is compute and talent.
Specifically, the salaries you’re willing to pay.

So DeepSeek’s decision not to raise funding last year was, in hindsight, a massive strategic mistake.
Without fundraising, there are no GPUs.
And researchers can’t be paid competitively.
This hurts both R&D and retention.

I’ve seen many comments attributing DeepSeek’s recent struggles to key talent being poached.
And Liang Wenfeng cannot match the salaries offered by rival firms.
Why can’t he match them?
Is he unable to get the money?
Not at all.

In 2025, countless funds — official, private, national, domestic, and overseas — wanted to invest in him.
He simply turned them down.

If you briefly reach the top spot in China, and everyone wants to give you their best resources, but you refuse, then what gives you the right to stay on top?
Why would the best people keep working for you?

Culture matters, of course.
People want to work at DeepSeek.
But how far can culture really go?
If ByteDance offers 10 million yuan, and DeepSeek offers 6 million — a 40% discount — maybe many stay.
But if DeepSeek can only offer one or two million, I think most people would leave.

Do you actually believe in AGI?
Do you believe talent is essential to achieving it?
If you do, and you believe it will generate enormous societal value, then you must believe in the demand for it.
You must scale up, invest more, raise salaries, and raise as much money as you possibly can.
If you just want to run a closed-door research lab, then why should society bet on you at a time when industrial competition is a matter of life and death?

In the following sections, I will discuss several directions.
First, a look at the supply side.
Second, a discussion of the demand side.
Third, whether patterns from the semiconductor industry can be applied to LLM development.

## The Business of LLMs (Part 2): Inference

Let’s start with the supply side.
LLM supply is straightforward.
Your production inputs are GPUs, or data centers.
Your output is tokens.

Jensen Huang talks about the "token factory" all the time because he’s spent 40 years in semiconductors.
This is obvious to him.
Most of us come from the previous era of software logic and don’t think in terms of hardware manufacturing.

To simplify, I will use H100s for all estimates below.

First, what does an H100 cost?
For easy calculation, I’ll use rental prices, since they already include depreciation, facility costs, maintenance, and so on.

H100 rental prices have risen.
They used to go as low as $2 per hour.
Now, due to GPU shortages and surging demand, they’re up to about $2.50 per hour.
That’s roughly $21,900 per year.

How many tokens can an H100 produce?
Using DeepSeek V3’s figure of 2,000 tokens per second, that’s 7.2 million tokens per hour.
So the cost per million tokens is roughly $2.50 divided by 7.2, or about $0.35.

Of course, data centers aren’t running at full utilization around the clock.
So if the final price stays somewhere near $1 per million tokens, that’s roughly break-even.

Token pricing is hard to pin down, though.
OpenAI and Anthropic API prices are inflated.
In practice, their rates for coding plans and consumer-facing products are lower.
B2B and B2C users also have different price sensitivities.

Let’s ignore those differences for now and assume a fixed gross margin.

Based on last year’s reports, Anthropic and OpenAI probably have inference gross margins around 40% to 45%.
But I think the real number is probably a bit higher.
Let’s use 50%.

So one H100, at $21,900 per year, should generate roughly $44,000 in revenue and $22,000 in gross profit.
Scaling up, 10,000 GPUs would mean about $400 million in revenue and $200 million in gross profit.
1 GW roughly corresponds to 1 million H100s.
That’s about $40 billion in revenue and $20 billion in gross profit.

With this model, we can even do a rough valuation of OpenAI.
How much compute does OpenAI have?
By the end of 2025, it should just have crossed 1 GW.
By end of 2026, it might reach 3 to 5 GW.
Using the earlier estimates, that corresponds to $120 billion to $200 billion in revenue and $60 billion to $100 billion in gross profit.

What about its long-term plan?
By 2030, OpenAI claims it will have 30 GW of compute.
That would mean about $1.2 trillion in revenue and $600 billion in gross profit.
If we discount that to net profit — say, $400 billion — and apply a 20x P/E ratio, that would imply an $8 trillion valuation.

Of course, current private-market and pre-IPO trades value OpenAI at around $1 trillion.
Would I buy at $1 trillion a company that might one day be worth $8 trillion?
I can’t say for sure.
There are still too many uncertainties.

How much of that revenue can support reinvestment?

Based on last year’s financials, OpenAI’s compute investment for R&D was about $9 billion.
That translates to roughly 450,000 H100s.
This is roughly consistent with its total fleet of over 1 million H100-equivalent GPUs.

OpenAI is currently over-investing in R&D.
Far beyond what its current revenue can support.
Because so far, R&D is still being funded by external capital.

When the promised 30 GW is fully deployed, how much of it will be allocated to R&D?
For a tech company, a common benchmark is reinvesting about 10% of revenue into R&D.
With $1.2 trillion in revenue, that supports at most $120 billion in R&D.
Of that, salaries and operating costs take a chunk.
In compute terms, that translates to roughly 3 GW.
That’s a reasonable share of its 30 GW total.

Now back to production — the inference side.
Continuing from the 30 GW figure.

Recently, OpenAI researcher Noam Brown gave a talk.
I didn’t attend, but I saw a screenshot.
He argued that control over inference-side compute has become a strategic advantage.
While model weights themselves are less important.
That is exactly the point I’m making here.

To contrast, consider a Chinese startup.
Because China cannot buy NVIDIA’s top-tier GPUs, no matter how strong its technology, its inference scale is capped by domestic GPU supply.

How many domestic GPUs are there?
Last year, Ascend 910C production was probably around 1 million units.
Of course, that depended on two things.
First, compute chip cores that TSMC manufactured for Huawei in 2020.
Second, over 10 million HBM memory chips that Samsung supplied before the relevant sanctions kicked in around 2022 or 2023.

Those inventories are long gone.
This year’s 950PR and 950DT series rely entirely on domestic supply chains.

What is the bottleneck ceiling of those domestic supply chains?
I read a SemiAnalysis article last year.
It argued that because Huawei’s phone business is not under sanctions, SMIC’s 7nm capacity could be fully allocated to Ascend chips.
That translates to roughly 7 million units.

By comparison, CXMT’s memory production is the bigger bottleneck.
In the most optimistic scenario, Dylan Patel estimated CXMT could expand to 30,000 wafers per month.
Annualized, that’s less than the memory needed for 1 million GPUs.
But I’ve recently looked into the memory used in the 950PR.
It’s not standard HBM, but a custom Huawei specification called HIBL.

HIBL is lower-performance memory.
Still large in capacity, but bandwidth is roughly half that of HBM3.
I couldn’t find much authoritative information on the process details.
I posted a question on Zhihu hoping someone knowledgeable could answer.
But perhaps it uses simpler stacking and packaging techniques, lowering production difficulty.
So it might bypass the earlier 1 million GPU memory constraint.

CXMT’s total capacity is about 300,000 wafers per month.
If all of that could be converted to HBM, it would correspond to roughly 60 million to 90 million HBM dies.
At 6 HBM stacks per GPU, that’s enough for about 10 million GPUs.
Of course, this is impossible — just a theoretical upper bound.

So Huawei’s output this year is probably in the low single-digit millions.
And those millions must be split across many companies.
Suppose a company — whether Zhipu, MiniMax, or someone else — secures 200,000 Ascend units.
Each unit offers roughly half the compute of an H100, with throughput slightly above half.
That translates to roughly $2 billion in annual revenue.
If it still allocates one-tenth to R&D, that’s only $200 million.
No matter how brilliant Chinese researchers are, you can’t rapidly train a better model than the US with that little compute.

This shows that inference scale and R&D investment are linearly related.
And because the benefits of R&D radiate across all production, R&D costs are amortized by scale.

## The Business of LLMs (Part 3): Demand

Everything above rests on one premise: you can find enough demand.
That demand won’t materialize automatically.
It may come from waves of product-market fit discovery, or from enormous effort spent integrating applications into the last mile.

But looking over a 3-to-5-year deployment horizon, if you believe in AGI and in its immense societal value, then that demand will appear.
That is a strategic-level judgment.

So at the strategic level, you should set capacity as high as you can.
At the tactical level, however, how to fill that capacity and find that demand remains a problem business teams must solve.

First, an LLM company must go global.
Compared with the relatively smaller domestic market, overseas markets are far larger.
Even if US access is blocked by sanctions, companies should do everything they can to enter Europe, the Middle East, Southeast Asia, Latin America, and other markets.
This kind of internationalization must be planned in advance.

Back when Huawei couldn’t sell its switches domestically, it sent large teams abroad to open markets in Europe, Russia, Africa, and Latin America.
It built an elaborate overseas organization.
Eventually, Huawei generated enough orders and profits to subsidize domestic research, becoming the R&D powerhouse it is today.

By the same logic, if a Chinese LLM company wants a place on the global stage, it must go global, internationalize, and bring overseas revenue back to fund domestic model development.
Of course, AI model exports should be much easier than hardware exports.
Most people in LLM companies speak English, so language is not the main obstacle.
You are exporting tokens.
Tokens just travel through fiber-optic cables.
You don’t need complex offline distribution channels, warehouses, service centers, or layers of local distributors.

If something this easy isn’t being done, how could you possibly handle harder orders?

Some might say we’re still in the externally-funded R&D phase, and current overseas revenue is negligible.
But commercial capability is part of a company’s organizational culture, and it also takes time to build.
If you don’t start cultivating a global sales culture today, by the time you truly need overseas revenue, it will be too late.

Moreover, OpenAI and Anthropic may each generate over $100 billion in revenue this year.
If you capture just 5% of that, it’s $5 billion.
That’s real money.

Besides, domestic token production costs may be lower than overseas.
Alibaba’s Qwen 3.5 Plus is priced at $2.40 overseas and $0.60 domestically.
If overseas tokens are more expensive and the market is larger, why not go earn that margin?

I’m less certain about whether companies need to expand B2B sales teams.
I’ve always had a nagging concern that too many B2B salespeople can damage a company’s culture.
But some things may have to be done.
And this B2B effort should also be global.
You should go directly to foreign companies to win foreign orders.

If Americans won’t buy, won’t Middle Eastern sheikhs?
If a Chinese LLM company simply guards the domestic market, waits for organic user growth, waits for more funding, and waits for orders to come in, then it is not really a company competing globally.
It is just a research lab.
And in this industry, research labs can win for a while, but they rarely win in the end.

Beyond language models, video generation will also be a huge source of demand.
YouTube’s annual revenue in 2025 already exceeds $60 billion, including ads and subscriptions.
If we assume 30% of that goes to content production, that’s $18 billion.
If half of that production cost is eventually served by LLM calls, that’s $9 billion in demand.
This is a very rough estimate, but it shows that video generation alone is enough to form a very large token market.

If we convert that to tokens, today’s video model algorithms are still quite inefficient.
Diffusion models still require multi-step generation.
Long-context architectures for video are still nascent.
But in the future, video generation costs should gradually approach the cost of one or two forward passes.
At that point, a single GPU could easily exceed language model decoding throughput.
For example, 10,000 tokens per second.

And because video generation is naturally more parallelizable, it may not even require HBM.
Slower, cheaper GDDR, or even LPDDR, could suffice.
If you design dedicated inference cards for video, costs could drop even further.

When video production costs fall dramatically, won’t that unlock even more demand?
I think that’s highly likely.

Today YouTube is already a platform with over $60 billion in revenue.
If part of content production is genuinely replaced by model generation, video generation won’t just be a cool demo.
It will become a massive, production-grade demand stream for the LLM industry.

Further out, what demand will AGI bring?
If LLMs solve interaction tasks on computers, the next step for AGI is solving interaction with the physical world.
If it can replace all physical-world interactions that currently depend on humans, that market is dozens of times larger than computer-based work.
It could potentially exceed $100 trillion annually.

Imagine a model deployed on a local robot controller.
Input is video, plus possibly speech, text, tactile sensors — multimodal signals.
Output is action.
It sees its surroundings, then cleans your house or works on a factory floor.
Each such robot may require the equivalent of today’s eight H100s, or even more.
If there are a billion such machines in the world, that might mean ten billion GPUs.
Of course, none of this may materialize until around 2035.
Honestly, that’s not so far away.

## The Business of LLMs (Part 4): Industry Consolidation

As discussed, we start from two assumptions.
First, LLMs are a mass-market commodity.
Second, revenue can be roughly estimated from token production.
Any industry satisfying these conditions will eventually settle into a pattern of one dominant winner, a second-place player barely scraping by, and everyone else losing money.

 domestically, the market has already narrowed from the "hundred-model war" of 2023 to roughly a dozen players.
The startup cohort includes DeepSeek, Moonshot AI, StepFun, MiniMax, and Zhipu.
The large internet companies include ByteDance, Alibaba, Tencent, Xiaomi, Meituan, Xiaohongshu, Baidu, and a few others.

The reason consolidation hasn’t accelerated is, as noted earlier, that LLM development has not yet reached the point where massive capital is strictly required for success.
Also, these companies are still relying on external funding or parent-company subsidies to fund LLM R&D, rather than relying purely on inference-generated revenue.

When those two conditions change — when the cost of developing each generation rises from today’s hundreds of millions of dollars to tens or even hundreds of billions, and when companies can no longer find external funding at that scale — the market will shift from fragmentation to concentration.

If only two or three of these dozen can survive, why wait until costs are so high that most go bankrupt or quit?
Why not consolidate now?

Under the current paradigm, having so many companies doing essentially the same thing is a significant waste of resources.
First, talent and compute are spread too thin.
Each startup only has tens of thousands of GPUs.
Each large company only has hundreds of thousands — except perhaps ByteDance.

Everyone’s work is highly homogenized.
They’re all scraping similar data, building similar infrastructure, training similar models, and optimizing on similar tasks.
For example, why do we need a Kimi K2, a GLM-5, and a MiniMax M2 or M3?
Do we really need three separate models?
Wouldn’t it be enough to pool all the GPUs and train one?

Conversely, if these companies merged, the benefits would be enormous.
Each of them has its own strengths.
DeepSeek on infrastructure.
Kimi on alchemy.
MiniMax on agentic post-training.
Qwen on data.
Seed on research.
If they merged, plus good cultural integration, could they combine these strengths to build a better model, serve more customers, and compete globally?
(I didn’t include Zhipu because it has no major weaknesses, but no exceptional strengths either).

This is the logic of market competition.
Sooner or later, the only difference is the cost of waiting.

Possible merger patterns fall into two categories.
First, startup-to-startup mergers.
Second, large-company acquisitions of startups.
Here we assume large-to-large mergers won’t happen due to irreconcilable core conflicts.
ByteDance won’t acquire Alibaba, and Alibaba won’t acquire Xiaomi, for example.

What would startup-to-startup mergers look like?
One possibility: a startup raises enough capital to simply acquire the others.
For example, I think DeepSeek should acquire StepFun.
If Xiangyu Zhang were leading DeepSeek V4, many decisions would not have been so hasty and absurd, and the exploration of AGI could have been deeper.

Another possibility is a merger among equals among startups.
The main obstacle here is organizational culture.
If merging causes cultural friction, it may not be worth it.
So whether this happens depends heavily on personal relationships among founders.
If Zhilin Yang and Yanjun Yin are on good terms, they could initiate contact and collaboration — data sharing, infrastructure sharing, technology sharing.
After a period of working together, if things feel compatible, they could explore a merger.
That would happen more naturally.

A third path is simply poaching from rivals.
This returns to the core thesis that inference scale determines competitiveness.
You can poach because you have money.
And you have money because you sell more tokens.

Large-company acquisitions of startups follow similar patterns.
The obvious one is a large company buying a startup.
But if capital markets are bullish on the startup’s prospects, its valuation will be high.
The large company may not have enough cash on hand.
The second route is poaching.
I think ByteDance has already poached quite a few people from other companies.
ByteDance’s talent density may not be higher, but its headcount is at least 20 times that of the others.
That gap may keep widening.
Where does ByteDance get all that money?
From the commercialization of its products — Doubao, Douyin, and the rest.
So again, the point stands: commercial success is a necessary condition for research success, even if not sufficient.

We’ve discussed how startups exit competition.
Another path is for a startup to simply dissolve.
For example, Kai-Fu Lee’s 01.AI.
At the time, 01.AI launched Yi-Lightning with just a 20-person model team.
It reached sixth place on the LM Arena leaderboard.
But within days, Kai-Fu Lee announced he was giving up.
Most of the team moved to Qwen.

As for large companies exiting, I don’t work at one, so I don’t really know.
But the general pattern is: you see massive spending with no visible return.
And you kill the project, like Microsoft has done with some efforts.

## The Business of LLMs (Part 5): Specialization

Another phenomenon that may follow semiconductor dynamics is the emergence of vertical分工 and specialization.
Semiconductors are an extraordinarily complex industry.
As noted, due to the learning curve, each segment produces a winner that dominates the vast majority of market share.
Yet along the vertical chain, each company usually manages one segment, rather than one company trying to own the entire stack.

TSMC succeeded in foundry services precisely because it only did foundry work and did not engage in chip design or other areas.
This structure emerged for several reasons.

The first is that focus enables technical leadership and extreme customer service.
Intel does everything.
TSMC only does foundry.
So it is harder for Intel to provide more focused technology than TSMC.

Another point Morris Chang emphasized in his autobiography is that TSMC should never compete with its customers.
It should serve them.
By doing only foundry and not chip design, it avoids competing with its customers.
So customers trust it more.

By contrast, if a foundry also designs chips, when both the internal design team and a customer bring orders to the fab, whose wafer gets priority?
Will customers feel uneasy sharing their designs with your company?

I recall that Google’s TPU v1 or v2 was fabricated by Samsung.
After receiving the design, Samsung copied its own NPU version.
This severely damaged Samsung’s credibility.
From then on, Google moved its orders to TSMC.

A third benefit is that if a company monopolizes one segment of the vertical chain, its orders can diversify, reducing risk.
TSMC can accept all kinds of contracts: CPUs, GPUs, communications chips, and more.
On one hand, these orders mostly use shared technology, so TSMC doesn’t need to repeatedly develop entirely different processes.
At the same time, diverse orders stabilize revenue.
When GPUs are hot, TSMC earns GPU money.
When CPUs are hot, it earns CPU money.
Risk and volatility are greatly spread out.

Furthermore, the ability to rapidly serve diverse customer needs becomes a core competitive advantage.
That is something Intel could not replicate.

A fourth point is that different segments, due to different R&D rhythms, production difficulties, and customer service characteristics, often require different organizational cultures.
Consumer-facing 2C businesses, for example, require attention to product, marketing, and promotion.
2B-heavy enterprises need large sales teams and strong relationships with major accounts.

These are the main reasons vertical分工 emerged in semiconductors.
We can see companies that specialize in lithography machines, like ASML.
Companies that specialize in optics, like Zeiss.
Or companies like TSMC that only do foundry.
Each has achieved near-absolute dominance in its own domain, yet each occupies only one link in the chain.

As Jensen Huang puts it, NVIDIA pursues "Do as much as needed, and as little as possible".

Will vertical分工 emerge in LLMs?
Since ChatGPT, the dominant theme in model development has been co-design: cross-layer optimization.
For example, product agents need to be tightly integrated with post-training.
Product people communicate requirements directly to post-training teams, prepare corresponding data, and improve model performance on real tasks.

Inference infrastructure teams need to talk frequently with architecture teams to ensure models are as efficient as possible after training.
Pre-training teams, in turn, need to work closely with infrastructure teams to maximize training throughput on large clusters.

The emergence of Claude Code and the evolution of RL algorithms are examples of this trend.
From this perspective, the trend in LLMs seems to be tighter integration across stages, not分工.

However, the V4 release made me see another possibility.
I remember when DeepSeek V3 came out at the end of 2024.
The paper described a series of very principled optimization techniques.
Parallel scheduling strategies, overlapping communication and computation, inference system design, recompute design, and so on.

Industry practitioners were blown away.
They studied it extensively and tried to integrate DeepSeek’s key kernels into their own training systems.
A year later, many teams have mastered those system designs.

But the optimizations in the V4 paper can’t even be described as principled, let alone极致.
They are borderline insane.
Black magic.

For example, MegaMOE in MoE computation and inference.
If you use Expert Parallelism, you need two rounds of inter-GPU communication: dispatch and combine.
To overlap communication and computation and avoid waste, normal teams split data into two micro-batches.
While one batch computes, the other communicates, reducing idle time.

But DeepSeek simply wrote a single massive MegaKernel.
It fused everything — communication, computation, the whole thing — into one kernel.
It doesn’t rely on micro-batches for overlapping.
Or rather, it has infinitely many micro-batches.
After every tiny block GEMM, it immediately performs communication.
So that no matter what the workload, everything overlaps perfectly.
The original text used the word "wave," suggesting a surging ocean.

A friend of mine who works on ML systems said MegaKernel is something the industry wouldn’t even dare to imagine.
Because when you fuse all communication and computation together, there is no way to debug it.
If something goes wrong, is it a communication issue, a compute kernel issue, or a broken card?
There is no way to diagnose it.
The difficulty is staggering.

All I can say is: the DeepSeek team is incredibly bold and skilled.

Beyond MegaKernel, V4 uses a series of similar techniques.
For SCA and HCA attention computations, with so many tiny kernel launches, they still saturate the compute.
After reading it, I just thought: is all this really necessary?

With such an incredible infrastructure team, the alchemy team seems to have gotten carried away.
And it’s not just the attention design.
When hit by loss spikes — a problem that shouldn’t exist in 2026 — the algorithm team’s first instinct was to solve it with system tricks.
They came up with a patchwork fix called Anticipatory Routing.

To exaggerate slightly: if the DeepSeek V4 team wanted to train a large model at a time when there was no attention, only LSTM, they would not pursue a parallelizable algorithm.
They would simply write an infrastructure system that saturates GPU compute on LSTM.

A second-tier alchemist I know put it this way: a weak infrastructure team actually acts as a regularization on algorithm design.
If an architectural change doesn’t bring a huge improvement, you can’t persuade the infrastructure team to modify the training code.
So you avoid falling into local minima.

System optimization and algorithm design require fundamentally different team cultures.
The biggest difference is that system optimization gains are predictable.
You know a card’s compute, its bandwidth.
If you reorder execution, overlap communication and computation more deeply, run the arithmetic.
You can conclude exactly how much faster things will get.

And with good code management, changes to kernels are isolated.
Modifying the front compute doesn’t affect the rear compute.
So overall system complexity grows only linearly with the number of changes.

Alchemy, by contrast, is pure black magic.
When you make a change, there is no way to predict whether it improves or degrades model performance.
And different components interact in mysterious ways.
For example, touching attention might affect MoE computation.
When errors occur, it’s hard to tell whether the problem is algorithmic, optimizer-related, architectural, or numerical.

System complexity grows exponentially with the number of changes.
So good alchemists are extremely cautious.
They pursue minimal changes.
They seek deeper understanding of phenomena.
And they propose the simplest possible algorithms.
Because even then, the underlying understanding remains a black box.
Just a slightly shallower one.

In terms of team culture, infra-bros and algo-bros are very different.
Infra-bros take pride in mastering complex systems.
They glory in solving hard problems.
They look down on storytelling and hype.

Algorithm people pursue minimalism.
They want algorithms and system designs to be as simple and universal as possible.
And they care deeply about storytelling.
Because although a yin-yang five-element theory of understanding is far from the true nature of things, telling a story is still better than not telling one.

I suspect that inside DeepSeek, the infra-bros are so numerous and so dominant that the algorithm people have been squeezed out.
For example, if there were an Ilya-like prophet in the company, speaking in riddles every day.
Or worse, a Jason Wei, who got a paper out of the sentence "Models should think before answering" and became a celebrated figure.
Their attitude would probably not be admiration or respect.
It would be contempt and mockery.

Here, I want to make a bold claim.
Perhaps DeepSeek’s best path forward is to transform into a pure infrastructure company.
Its business could be providing inference frameworks for other manufacturers.
For example, a model trained by Kimi or ByteDance may be costly to deploy internally.
Hand it to DeepSeek, and DeepSeek would look at it and say, this is basically a V3.
Then they’d write a system with极致 optimization, fully overlapping MegaKernels, maybe even overclock the GPU.
Cut inference costs dramatically.
The savings get split between the model owner and DeepSeek.
Isn’t that ideal?

For their customers, managing and continuously optimizing inference systems is a huge hassle.
It involves too much low-level kernel optimization that has nothing to do with the pursuit of AGI.
If all of this could be outsourced to DeepSeek, the customer could focus purely on the model.

For example, if Moonshot keeps its headcount at three to five hundred people, for cultural and anti-bureaucracy reasons.
Then those three to five hundred people could all be like Jianlin Su.
Or half Jianlin Su and half post-training data cleaners.
The company’s direction becomes more focused.
Alchemy quality goes up.

And for DeepSeek, it can finally stop resenting the black magic of alchemy.
And focus on what it does best: infrastructure optimization.
You love patching kernels?
Go patch them.
There are endless patches to apply.
You’ll never run out.
It’s so satisfying.

In model quality, DeepSeek V4 isn’t even the best in China.
Globally, it probably ranks fifth or lower.
That’s very different from V3’s brief moment of leadership.

But in infrastructure?
I don’t believe there are many people in the world stronger than this team.
It could be an absolute number one.
An absolute monopoly.

As emphasized earlier, market leadership is a matter of life and death.
If it could capture even half of China’s total inference market — if ByteDance and others outsourced to it — that would be an enormously profitable business.
In other words, it would be taking money from upstream cloud providers, not competing with downstream model makers.

Liang Wenfeng has said his goal is AGI.
But who says providing infrastructure isn’t AGI?
If their existence lets every company in China focus on alchemy without worrying about tedious kernel optimization, doesn’t that accelerate China’s march toward AGI?

Of course, such companies already exist.
Notable examples abroad include Together AI and Fireworks.
But they seem to face intense competition and are doing just okay.
Why wouldn’t DeepSeek end up the same way?

First, before V4, no company was two or three generations ahead in inference optimization.
But V4’s technology is genuinely insane.
Second, overseas companies mostly use NVIDIA cards.
NVIDIA cards are well-supported and relatively easy to optimize.
Most model companies still have a chance to build reasonably efficient inference systems in-house.

But in China, the foreseeable future depends on Ascend.
Ascend’s ecosystem is very weak.
And there are even issues with compute precision.
Getting Ascend to work well is beyond the reach of ordinary companies.
If DeepSeek offered this service, and a model ran as fast on a 950DT as it would on an H100 at over 60% throughput, for less than half the cost, that would be an enormous, unassailable advantage.

By analogy, the foreign company best positioned to do this is Thinking Machines.
Like DeepSeek, they gathered a group of elite engineers who spend all day patching infrastructure and kernels, not doing anything "proper".
I’d suggest Amazon acquire Thinking Machines.
Their Trainium is probably about as hard to use as Ascend.
Let the Thinking Machines folks fix Trainium.
I think that could work.

## The Business of LLMs (Part 6): Customization

TSMC founder Morris Chang repeatedly emphasized in his autobiography that he preferred "customized" products over "mass-market" products.
Because the former typically command higher gross margins.
Mass-market products, with standardized specs, face fierce competition.
As discussed, winner takes all, small players lose money.
But customized goods are not in fully commoditized competition.
There is more room for pricing negotiation.

We can ask the same question: will the LLM market see customization for different companies?
There have been many attempts, broadly falling into two categories.

The first is startups attempting to fine-tune specialized models.
Medical LLMs, financial LLMs, serving professional clients.
They may find early success in niches that large model companies ignore.
But quickly, when companies like OpenAI or Anthropic notice the opportunity, their specialized models lose to new general-purpose models.
The reason is simple.
A domain-specific LLM is really just different data.
The underlying technology is largely the same.
If you research in only one domain, you can’t advance as fast as someone researching across all domains simultaneously.
And different domains often produce useful cross-pollination.
For example, reasoning model techniques were originally tuned for math tasks.
But they quickly affected every domain.

The second category is companies offering fine-tuning services for clients.
Applying the same techniques to different domains based on customer needs.
This includes startups like Prime Intellect and Thinking Machines, as well as model providers like OpenAI and Anthropic.
Does this work?
To some extent.
We often hear news like Anthropic providing a reduced-safety-review model version for the US government, for military and defense applications.
But overall, these businesses have not performed very well.

I suspect this is related to the lack of continual learning.
Today, LLM learning processes can be broadly divided into two categories by duration and training cost.

One is pre-training and post-training.
I’ll group these together because they have to be done all at once.
The total training volume is in the tens to hundreds of trillions of tokens.
And the cycle lasts months.
Once the model is released, its weights are fixed.

The second category is in-context learning.
The model adjusts its behavior based on context.
But context length is at most around 1 million tokens.
What can be learned is limited.
And it does not rely on customized weights or dedicated deployments.

I discussed this in my article "Perhaps Continual Learning Should Not Be Achieved".
I encourage you to read it.

Recently, however, some new post-training techniques may point toward a form of continual learning.
Pre-trained and SFT models often suffer from catastrophic forgetting.
If training data is not independent and identically distributed, not shuffled, but sequential — learning one thing first, then another — the model forgets earlier skills while learning later ones.
So model training must be done in one big batch.

But people have discovered that reinforcement learning and on-policy distillation suffer far less forgetting.
For example, both Zhipu and Nemotron used cascade RL when training models.
Instead of learning all RL domains simultaneously, they learned them one at a time.
First RLHF, then math, then coding, then agentic tasks.
The result: when learning later domains, performance on earlier domains barely dropped, at most by two points.

And on-policy distillation shows even less forgetting.
A stream of recent papers has explored this, spawning techniques like self-distillation.
Dario also mentioned in an interview that continual learning might be achieved in some form this year.

If this does happen soon — if we can insert a billion-token or tens-of-billions-token level of continual learning between the hundreds-of-trillions pre-training stage and the 1-million-context in-context learning stage — then customization might actually become viable.

For example, a company has internal documents, operating norms, customer information, and so on.
Under current agentic paradigms, these are often packed into files or skills, then retrieved, read, and reacted to on the fly.
This often fails to produce genuine behavioral change, misses important information, or prevents cross-domain association.
If continual learning is achieved, the model could organically absorb and synthesize all this context, like a veteran employee.
It might deliver significantly more value than using a pre-trained general-purpose model directly.

I mentioned in "Perhaps Continual Learning Should Not Be Achieved" that implementing continual learning for individual users may not be worth the cost.
Because it means deploying an entirely new model weight set for each user.
That is expensive and underutilizes compute.

But if a company with thousands or tens of thousands of employees gets its own dedicated inference deployment, and the improvement is truly substantial, the cost could be amortized.
It is worth keeping an eye on these technologies.
If a breakthrough does occur, a new business model could emerge.

This leads to a second question: if it does happen, will model providers capture this market?
Or will specialized companies take it?
I don’t know, but we can look at it from two angles.

First, does a 2B business require a different organizational form and company culture?
For example, serving a company in this way often requires sending engineers on-site, collecting offline data, and debugging in real time.
Would a company like Anthropic be willing to do that?
Maybe.
But maybe not.

Second, might the profit margins in early stages be too small to attract large companies?
Here I am drawing on the arguments of Clayton M. Christensen, author of The Innovator’s Dilemma.
If a new technology can serve existing markets and existing users, large companies will fight for it.
Small companies have no chance.
But if an emerging market initially generates too little revenue — say, only millions or tens of millions — large companies disdain it.
Small companies, however, are happy to enter.
A few years later, as the emerging technology matures and becomes mainstream, large companies that want to enter find it is too late.
The small company has accumulated years of know-how and surpassed the large players.
