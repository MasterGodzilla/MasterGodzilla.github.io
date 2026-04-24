---
title: 'When Did Math Start Feeling Counterintuitive?'
date: 2024-07-01
permalink: /posts/2024/07/nonintuitive-math/
tags: [math, education, reflection]
---

Math seems to have been counterintuitive all along.

The human brain is not naturally good at complicated math. Years of training are what reshape it.

Here are the moments that left the deepest impression on me.

The first was in kindergarten. I was trying to calculate 7 + 8, and no matter how long I thought about it, I could not understand why the answer was 15.

My mom told me to count on my fingers. I said I did not have enough fingers. She said I could borrow both of her hands. After several minutes of counting, I finally reached 15, but I was still completely confused about why.

The second time was just after I had learned the distributive law. My mom asked me what $(a + b)^2$ was, and I said $a^2 + b^2$.

At the time I was lying in the back seat while she was driving me to school. I had no paper or pen to help me, so I just kept thinking about where the $2ab$ term came from.

The third time was in high school physics. The teacher gave us a problem: suppose there is a gravitational dipole, meaning a planet with positive mass and a planet with negative mass. You are at a point far away on the midline between them. What is your gravitational potential energy?

My formulas told me that if I integrated along the midline, since the gravitational force along that line was 0, I could push myself out to infinity without spending any energy. So the potential energy should be 0. But no matter what, I could not convince myself that this answer was right.

The teacher said that although it was an exam, we were allowed to write code to test it. I quickly made an animation in Visual Python. I found that gravity would make me move in a semicircle around the dipole until I reached the midline on the other side, and then I would come back.

I thought: if my gravitational potential energy is 0, then with even a tiny bit of initial kinetic energy, I should be able to escape to infinity.

So I added a tiny velocity to the positive-mass ball that represented me in the code, and ran the simulation. The ball still moved around the same orbit, and nothing seemed to change.

I thought my formula must be wrong, so I crossed out my answer and turned in a blank response.

The teacher said the answer was 0. I had originally written the correct answer.

Later I opened the code again and realized that the ball actually had changed. The radius of each semicircular pass increased by a tiny bit. If the program ran long enough, it really could reach infinity. My code was not wrong. I just was not patient enough, and I had not tried a larger initial velocity or a faster time scale.

That episode has stayed with me for a long time. I had actually produced the right answer, and still insisted on the wrong view.

The fourth time was in high school math class. The teacher was explaining various numerical methods for integration: estimate with little rectangles, estimate with trapezoids, and then naturally, estimate with parabolas.

The magical part was that the textbook said the parabolic method only has error when estimating polynomials of degree 4 or higher. In other words, Simpson's rule can perfectly estimate the area under a cubic curve.

I calculated it for a long time, and sure enough, it worked. But because I never studied numerical analysis, I still do not understand it to this day.

The fifth time was that rational numbers are countable.

The sixth time was that irrational numbers are uncountable, along with its consequences: Gödel's incompleteness theorem and Turing's halting problem.

But after I got to college, counterintuitive things actually became rarer and rarer. After all, I had already received so much training.

I also have to thank 3Blue1Brown. A lot of the time, I first had the right intuition and only later saw the concrete formula details.

We were in an AP Physics class. The teacher said that you could get an A from the normal exam, but to get an A+ you had to solve the bonus problems. There was one bonus problem per chapter. They were either very hard or very tedious. You could spend up to three hours on one, but usually if you could not solve it in the first hour, you were never going to solve it. You were not allowed to look things up in the book, but you could write code on the spot.

The dipole was one of those problems. It was not actually hard, but it was very clever.

I remember a few other fairly evil problems too.

For example, suppose there is an upside-down parabolic track. From the lower, inner edge of the track, you launch a small cart with a certain initial velocity. At which point is the centripetal force on the cart largest?

You can imagine three cases.

In the first case, the cart is not fast enough, so it falls off the track before reaching the top. If I remember correctly, the largest centripetal force is at the starting point, though I may be misremembering.

In the second case, the cart is very fast and shoots through the entire parabolic track. In that case, the point of maximum centrifugal force is the highest point of the track.

The third case is the evil one: the speed is just enough for the cart to make it through the whole track by relying on centrifugal force, but it is not that fast. In that case, the cart's centrifugal force reaches its maximum somewhere in the middle. I remember the written answer being three lines long, but the teacher said it was wrong.

Anyway, I think I was the only person in our class who ever solved one of the bonus problems.
