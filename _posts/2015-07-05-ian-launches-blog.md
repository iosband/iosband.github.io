---
layout: post
title: Another machine learning blog
date: 2015-07-05
---

## First steps

~~~
print('Hello world')
~~~

Welcome to my (mostly) academic blog on data, decisions and learning.

I know what you're thinking, do we really need another machine learning blog on the internet?
I think the answer is yes and it's time for me to join in, here are some reasons why:

1. Openly sharing ideas, even when only half-formed, helps the entire field progress.
2. Academic journals are great for *precise* statements, but may be more difficult for sparking interest and *intuition*.
3. Ensembles make good learners, let's get some different voices in this ensemble.
4. Writing is hard, so I hope I get better through practice!


## Roadmap

I'm interested in how to make good decisions under uncertainty and I want to write some articles that help make progress in understanding this problem.
This topic is so insanely broad that nobody really believes me when I say that's what I study... but it's true, or at least that is the goal.
What we usually study in machine learning makes up only one very small part of that bigger question.
In this blog, I hope to outline some of the small part of the research in this area that I have been involved in.

In particular, I intend to discuss some elements of the [reinforcement learning](http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html) problem, which has received far less attention than other more traditional machine learning topics such as classification or regression.
In fact, in its most general form reinforcement learning (RL) contains classification, regression and all supervised learning as a special case... so if we can solve RL then we don't need to worry about anything else!

<img src="http://truedemocracyparty.net/wp-content/uploads/IROBOT2.jpg" alt="iRobot" style="width:800px">

Of course I'm joking here, many problems really don't call for the reinforcement learning... but a lot of the important ones do.
Most statistical or machine learning methods were developed in a time where data and decisions were done separately.
Nowadays, many systems are being deployed that are completely automated so that the data they gather determines the decisions they will make.
However, it is also true that **the decisions they make will influence the data they get**... once this happens everything changes.
If you want to make a learner that influences its system, you're going to need reinforcement learning.


## Exploration vs Exploitation

One of the fundamental problems for decision making under uncertainty is the so-called [exploration vs exploitation](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem.
This describes one of the main dilemmas that comes when you are forced to make decisions under uncertainty.
You can either try to take the best action you can given what you know (exploitation), or you can try to do something you don't know a lot about in the hope that you'll increase your understanding (exploration).
Traditional statistical analyis is great for telling you how to make sense of a dataset, which can give you an estimate of how best to exploit.
However, if you're not careful, you could end up doing really badly in the long run if you don't make an effort to explore.

To make this more concrete I will use a very simple example.
Imagine you're a doctor, a data-driven doctor of course, and your goal is to save as many cancer patients as possible.
For simplicity sake we'll start with the setting where all cancer is the same, all people are the same and there are only two drugs available.
We will also assume that once someone is diagnosed they will certainly die that same day without any drugs, if you give them drug 1 they will be cured with probability $$p_1$$ and if you use drug 2 they will be cured with probability $$p_2$$.
What *should* you do?

<img src="http://lexfridman.com/blogs/thoughts/files/2012/12/Confused-Doctor.jpg" alt="Doctor" style="width:800px">


Of course, if you know the success rates of both drugs $$p_1, p_2$$ then this problem is trivial, pick the better drug and use that one.
However, in the real world you never *know* the exact probabilities... so what *can* you do?
Well usually people gather lots of experimental data to estimate the probabilities $$\hat{p}_1, \hat{p}_2$$ and then choose the the one with the higher estimate, all good?
Not really... here are some problems:

1. **Your estimates might be wrong.** We could have got unlucky and measured drug 1 was better than drug 2 in the trial. If we never try drug 2 again we will never learn and for the rest of time people will die needlessly.
2. **What about all the people we killed gathering the experimental data?** Even if we made the right conclusion in the end, if lots of people had to die to test out another suboptimal drug this seems like a bad outcome.
3. **What do we do when new drugs come on the market?**

You can see that even in this incredibly simple example, the traditional [A/B testing](https://en.wikipedia.org/wiki/A/B_testing) don't really give satisfactory answers.
This simple problem is known in the academic literature as a [two-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) and no it's not a very descriptive model for how an actual doctor works.
However, being such a simple problem it is one of the few topics I'll cover that actually has a *correct* answer.
In my next blog post I'll talk about this problem more formally and dive into some details of how to go about solving it.

First post down, phew. Be excellent to each other.

<img src="https://rachaeljames.files.wordpress.com/2013/07/3rn5dx.jpg" alt="Bill and Ted" style="width:800px">


