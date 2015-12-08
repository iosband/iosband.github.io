---
layout: post
title: What is reinforcement learning?
date: 2015-12-04
---

It's been a while since my [last post](http://iosband.github.io/2015/07/28/Beat-the-bandit.html) and my attempt at a "fun" bandit demo.
For what it's worth, it's turned out that not even I am immune to the [Bieber fever](https://www.youtube.com/watch?v=fRh_vgS2dFE)... that little dude's got some funk.
Anyyways - it's about time I delivered on some of my promises to blog about [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) - here we go!

## Learning with delayed consequences

We've seen that in the world of "big data" not all data is created equal.
Examples like medical testing show that we can potentially gather a huge amount of data, but if it is not the *right* data we might never learn the correct decision.
The point here is **if we never try a certain course of action, we might never obtain the data that we need to learn about that action**.
This problem is at the heart of the [exploration vs exploitation](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem and the so-called "multi-armed bandit".
If this is new to you, then I suggest you go back and have a look through some of my [earlier posts](http://iosband.github.io/2015/07/19/Efficient-experimentation-and-multi-armed-bandits.html).


The good news is, the techniques from multi-armed bandit (aka Bayesian optimization) do allow us to balance exploration and exploitation in these sorts of learning problems.
We have expanded our understanding from standard [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), which treats the data as fixed so that now we can account for feedback based upon our decisions.
**However, we have still made several simplifying assumptions about the problem**:

1. **Independent, identically distributed trials** - The unknown function $f^*$ (the one we are trying to optimize) remains constant at each timestep.
2. **No long term consequences** - If we make a decision at timestep $t$, then the only effect on the system will be at this exact timestep. At time $t+1$ the world will be reset.

But maybe this isn't always a very good model for the problem at hand...

Let's stick with the medical example but let's change it in a few ways.
**Imagine you're a doctor in charge of a hospital that manages hundreds of patients with diabetes**.
Every week, for every patient you can choose to:

- Do nothing
- Prescribe exercise + diet
- Drug A
- Drug B
- Amputate limb

and your goal is to maximize the cumulative sum of quality years of life for your patients.

<center>
<div>
 <img src="http://fewfice.com/dcim/llp/14/4101004-doctor.jpg" alt="doctor" style="height:300px">
</div>
</center>


Unfortunately you're a bit of a maverick and you slept/partied through that section of your med school training without picking much up.
Luckily you're still an expert on experimentation and data science, so you you can simply *learn* the optimal policy through a multi-armed bandit algorithm.
Every time a patient comes in, you'll run an experiment and try out one of the different options... yeah you might make a mistake once or twice, but you'll be gathering data and you'll learn the right answer right??


Well, in addition to the numerous moral+safety objections this plan is really bad science!
Why?
Because it's clear that this problem really violates the two modeling assumptions we had for the multi-armed bandit in major ways...


**(1) Not every patient is the same**.
Different patients will come in with "diabetes", but the specific characteristics of their disease might be significantly different from patient to patient.
Some of the patients might respond well to lifestyle changes, while others might require emergency surgery... but you don't want to mix them up!

**(2) This is a process with long-term consequences**.
Diabetes is not the sort of ailment we can hope to cure in a single treatment.
The choices we make a week $t$ make affect the livelihood of the patient for all $\tilde{t} \ge t$.
For example, if we amputate a limb one week - it's not going to be "reset" in the next week!

If you want to be able to learn make decisions in this kind of environment... you're going to need **reinforcement learning**:

<center>
<div>
 <img src="http://i.imgur.com/9lxWEIl.png" alt="rl_concentric" style="height:300px">
</div>
</center>

but how can you model an environment like this?

## Markov Decision Process

A [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) is a simple model for decision making in environments with long term consequences.
We will formally define the finite horizon MDP $M$ by a tuple:

$$ M = (S, A, R, P, H, \rho) $$

- **$S=$ the state space**. This is a set which describes all the possible states the environment can be in. This state should be a [sufficient statistic](https://en.wikipedia.org/wiki/Sufficient_statistic) for the system.
- **$A=$ the action space**. This is a set which describes all the possible actions the agent can take.
- **$R=$ the reward function**. At every timestep $t$, the agent receives a (potentially random) reward $r_t \sim R(s_t, a_t)$. This reward might depend on both the current state and action chosen.
- **$P=$ the transition function**. At every timestep $t$, the state of the environment evolves $s_{t+1} \sim P(s_t, a_t)$. Once again, this transition might depend on both the current state and action chosen.
- **$H=$ the problem horizon**. Every episode of the interaction with the MDP will reset after $H$ timesteps.
- **$\rho=$ the initial state distribution**. Every episode will begin with $s_1 \sim \rho(\cdot)$, the agent has no control over this.


This is a lot of notation that might seem kind of abstract, but it should all start to make sense once I link it back to our diabetes example.
At each timestep $t$:

- **The state** $s_t \in S$ describes the characteristics of our patient.
Here the state space $S$ is nothing more than a collection of possible patients.
- We (the doctor) then choose **the action** $a_t \in A$ which we want to use.
- **The reward** $r_t \sim R(s_t,a_t)$ might be as simple as $1$ for living and $0$ for dead.
- The patient's condition will now **transition** based upon our treatment $s_{t+1} \sim P(s_t, a_t)$, perhaps they improve or deteriorate.
- **The horizon** $H$ is more of a technical device so that the overall problem is finite. However, we know that nobody will live forever, so $H$ could just be an upper bound on human life expectancy...
- $\rho$ is **the distribution of initial patient characteristics** in the population, which we do not change.

<center>
<div>
 <img src="http://i.imgur.com/nNZ6XWH.png?1" alt="rl_diagram" style="height:300px">
</div>
</center>

The goal of the agent is exactly the same as before.
We want to maximize our sum of rewards through time:

$$ \max_\pi \mathbb{E}_{M, \pi} \left[ \sum_{k=0}^K \sum_{h=1}^H r_{t=kH+h} \right]$$

The important addition is that **our actions can now have long term consequences in the environment** through the influence of the transition function upon the state.
Actually, we can now view everything we've done before as a special case of this more general formulation.

**Multi-armed bandit** is a special type of decision problem where there is only one possible state that never changes.

**Contextual bandits** are an extension to the multi-armed bandit that allows for different states, but your actions don't do anything to change the state.
For example, we might need to treat men and women with different doses of a given drug for the flu, but our treatment won't change whether our patients are men or women.


## So... what is reinforcement learning?

Reinforcement learning is the most general machine learning framework for learning.

1. You only learn about the data you actually gather.
2. Your choices have impact beyond a single timestep.

In this setting above, it could refer to **learning to optimize an unknown MDP $M$ where the reward function $R$ and the transition function $P$ are not fully known**.
Just like the multi-armed bandit, the agent faces an exploration/exploitation dilemma, but the effects of delayed consequences can make it significantly more challenging!

In my next post I'm going to go into some more detail on what makes this problem so hard and, more importantly, how we can make good learning algorithms!
For now, if you are interested in learning more I'd recommend you check out the accepted intro to the field [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html).

That's enough for now, I've just touched down in Montreal for [NIPS](https://nips.cc/) - looking forward to seeing what is happening out there!
