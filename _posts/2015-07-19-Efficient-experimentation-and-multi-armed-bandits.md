---
layout: post
title: Efficient experimentation and the multi-armed bandit
date: 2015-07-19
---


In my [last post](http://iosband.github.io/2015/07/05/ian-launches-blog.html) I introduced the problem of *exploration vs exploitation*.
In particular, we used the example of a doctor trying to save lives using some medical procedures.
For each patient the doctor can either try out a new drug in order to gather more data (exploration) or she can use the drug which seems to be the best given the data she already has (exploitation).
This sort of tradeoff is not just is often referred to in the academic literature as a [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem.
This week I'm going to to define the problem more concretely and try to give some insight into how we to tackle this effectively.

## Multi-armed bandit

What is a multi-armed bandit? Well... here are a few types:

<div>
 <img src="http://www.rmtracking.com/blog/wp-content/uploads/2011/11/iStock_000014773673XSmall.jpg" alt="thief" style="height:170px">
 <img src="http://epguides.com/Zorro_1957/cast.jpg" alt="zorro" style="height:170px">
 <img src="http://research.microsoft.com/en-us/projects/bandits/MAB-2.jpg" alt="mab" style="height:170px">
</div>

Actually, it's the third photo and the goofy looking octopus that I'm talking about.
You see a "one-armed bandit" is an old name for a slot machine that you'd find in a casino, because it has one arm and it steals your money... laugh track please!
The *multi*-armed bandit problem now comes from a sylized problem of a casino which has many slot machines or one-armed bandits:

- You enter a casino which has $1,..,K$ distinct slot machines (arms).
- Each arm $i$ pays out 1 dollar with probability $p_i$ if it is played, otherwise it pays out nothing.
- Every timestep $t$ you have to pick a single arm $a_t \in \{1, .., K\}$ to play.
- Based on your choice, you receive a return of $r_t \sim Ber(p_{a_t})$.
- **How should you choose arms so as to maximize your total expected return?**

Now clearly this is a bit of a silly example and in the real world slot machines are a terrible way to make money.
The point of this example is to highlight the very simple tradeoff between *exploration and exploitation*.
If we can understand how to solve this toy problem then hopefully we will gain some insights we can pass on to settings which we actually care about... like medical testing, advertising, robotics.
The key point to this problem is that when you first enter the casino *you do not know the payour rates of each machine*.

### Bandits with generalization

The classic casino "slot-machine" example is particularly simple because we know that each slot machine either returns one or zero with *independent* outcomes.
If we want to learn about the casino then at the very least we will need to try every arm once since
*sampling one arm doesn't tell us anything about any other arm*.

However, this also severely limits the practical use of such algorithms.
Imagine if your doctor had to re-learn basic anatomy for every single patient they encountered... nobody would survive!
Even though every individual is distinct (and may benefit from personalized care) it is crucial that we can share information across similar patients.

>For efficient learning on large problems we need to generalize between similar experiences.

 <img src="http://homepages.inf.ed.ac.uk/amos/figures/gpprior.png" alt="family" style="height:250px">

Fortunately, we can enrich our multi-armed bandit model to allow for this type of shared information.
Once we do this we consider a general method for "learning to optimize" an unknown function $f^*$.
This is when the multi-armed bandit starts to get interesting:

- The environment is described by some unknown function $f^* : \mathcal{X} \rightarrow \mathbb{R}$.
- At each timestep $t$ you choose some action $x_t \in \mathcal{X}$.
- You receive a return $r_t = f^*(x_t) + w_t$ where $w_t$ is some zero-mean noise.
- **How should you choose arms so as to maximize your expected cumulative return?**

Sometimes this formulation is known as [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization), or the multi-armed bandit with dependent arms.
One example that's had a lot of attention in the literature is where actions parameterized by $d$ real numbers $\mathcal{X} = \mathbb{R}^d$ and the unknown funciton is linear $ f^* (x) = x^T \theta^* $.
Another common model is that the unknown function $ f^* $ is drawn from some [Gaussian process](http://www.gaussianprocess.org/), which provides a flexible model for nonparametric estimation.




### Some notation

Before we dive in we'll need just a little bit more notation:

- We'll write $H_t$ for the information seen up to time $t$, technically this should be a [filtration](https://en.wikipedia.org/wiki/Filtration_(mathematics)).
- A learning algorithm $\pi$ is a function which maps data $H_t$ to distributions over actions $\mathcal{X}$.
- We'll write $ x^* $ for a member of $\arg \max_x f^*(x)$ which is optimal for the underlying function.

I'll try to keep the intuition mathematics-free, so don't be too put off it this notation looks a bit overwhelming!


## What makes a good learning algorithm?

In most types of machine learning it's pretty easy to tell if you have a good algorithm.
An image recognition system is good if it classifies a lot of images correctly, a political pundit is good if they [predict the US presidential election](https://en.wikipedia.org/wiki/Nate_Silver).
Therefore, you might think that a bandit learning algorithm is good if it gets good rewards... well... sort of... but not quite.

The problem with the multi-armed bandit is that you can't really tell just from the rewards if you're doing well or not.
You might be getting good rewards because it's just an easy problem, when the *true* optimal actions are far better than you ever got.
Unlike predicting the election, you never get to find out what the *true* answer is unless you try it.
If we want to guarantee "good" performance for a learning algorithm, we'll need to introduce a clear notion of "good".

### Regret, or how much better things could have been.

It's not possible to guarantee that a learning algorithm will generally attain good rewards on any bandit problem.
Some functions just have maxima which aren't very high.
However, we might reasonably hope that our learning algorithm can get *close* to the best possible performance *for that problem*.

We formalize this notion in terms of the **regret** or, how much better *could* we have done if we had known the true function $f^*$ from the start.
Effectively this shows how much worse we did through following $\pi$ instead of a policy with full information:

$$  {\rm Regret}(T, \pi, f^*) = \mathbb{E} \left[ \sum_{t=1}^T f^*(x^*) - f^*(x_t)  \right] $$

Actually, this is pretty similar to our day to day definition of regret.
You have high regret whenever you look back on your actions and realize you could have done much better with your life... just like looking back on a bad photo:

<div>
 <img src="http://img.humorsharing.com/media/images/1212/i_weird_families_007_50e050e7c7542.jpg" alt="family" style="height:200px">
 <img src="https://s-media-cache-ak0.pinimg.com/236x/7c/55/ab/7c55ab425cf2cd1c264277a8792235d8.jpg" alt="mullet" style="height:200px">
  <img src="https://scontent-lhr3-1.xx.fbcdn.net/hphotos-xpf1/v/t1.0-9/11021099_10203352168477639_168498448054596353_n.jpg?oh=e2e9c5cc3cae9b1cbf6c1b1d88011aa5&oe=5648ABAB" alt="ian" style="height:200px">
 <img src="https://s-media-cache-ak0.pinimg.com/736x/e0/6c/88/e06c880bcbe59e9d1faeca1019858f85.jpg" alt="cats" style="height:200px">
</div>



We want algorithms that you can guarantee will end up with "low regret".
So although we are bound to make a few mistakes while learning, we want to be able to keep these mistakes to an absolute minimum!
Not only that, we'd like to make sure that this happens *as quickly as possible*...

## Algorithms and approaches to learning

We're now ready to try and consider how to make good decisions under uncertainty.
In my next post I'm going to add some code and interactive demos to help build some intuition... for now we'll have to make do with the high level descriptions.
There are a few blog posts [elsewhere](http://camdp.com/blogs/multi-armed-bandits) that have some simple examples ready to go.

### Algorithm 1: Be greedy
The most obvious thing you might try is to pick the arm that has the highest success rate given the data you've seen so far.
This is called the "greedy" solution since you greedily choose the best option for this single timestep with no thought for the future.
Here is the formal algorithm:

**For each timestep $t$**:

$$\text{Estimate } \hat{f}_t = \mathbb{E} [f^* | H_t]$$

$$\text{Choose }  x_t \in \arg \max_x \hat{f}_t $$

Intuitively it might seem like this is the best thing you can do... but over the long run it's not.
The problem is that may be choosing actions which do not increase your future understanding of the system.
By replacing the unknown $ f^* $ with your point estimate $\hat{f}_t$ you overstate your knowledge.
In fact, you cannot even guarantee that this algorithm will *ever* learn the correct policy.

### Algorithm 2: Be *mostly* greedy

Being greedy didn't work because sometimes we could become prematurely fixated on a particular arm and then never realize our mistake because we didn't try any other options.
We can get around this problem by a very simple alteration.
We'll keep the exact same algorithm as before except at every timestep with some samll probability $\epsilon$ (think 5%) we'll pick an arm completely at random.
This will ensure that we get to see some other spread of data

**For each timestep $t$**:

$$\text{Estimate } \hat{f}_t = \mathbb{E} [f^* | H_t]$$

$$\text{With probability } \epsilon \text{ choose } x_t \sim Unif(\mathcal{X}) \text{ , otherwise choose } x_t \in \arg \max_x \hat{f}_t $$


This algorithm is possibly the single most popular strategy in many settings of interest.
If we choose this $\epsilon$ small enough and decaying over time we can guarantee that the algorithm will converge on the optimal solution ${\rm Regret}(T, \pi) = o(T)$...
success... well not quite.

Although this algorithm (called $\epsilon$-greedy) will *eventually* converge to the optimal solution the **time it takes to learn the optimal policy will typically be exponential in the complexity of the problem**.
For example, in the case of casino it will grow like $e^K$... which gets ridiculously large even for $K=100$.
This means that for reasonably size problems although we have a guarantee the algorithm will converge assymptotically... we won't always have a guarantee it will converge this century.
That's not very good.

The reason this algorithm is so slow is because the way it explores is not directed efficiently.
The algorithm is either choosing greedily or taking an action at random.
If you want to learn efficiently you have to *plan to learn*.
The next algorithm we'll discuss does exactly that.

### Algorithm 3: Bayes optimal

In some sense, the *optimal* solution to the multi-armed bandit can be stated very simply using the techniques of [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics).
Simply stated, the agent should formulate their posterior beliefs and then solve for the action with the highest expected long term rewards:

**For each timestep $t$**:

$$\text{Evaluate posterior } \phi(\cdot | H_t) \text{ for } f^* $$

$$\text{Solve for } \max_x \mathbb{E}_\phi \mathbb{E} \left[ \sum_{j=t}^T f^*(x_j) \right]$$

Formally this characterises the optimal solution very concisely and in the case of independent arms this can be computed effectively by the results of [Gittins indices](https://www.google.co.uk/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=gittins%20index).
However, the problem of computing Bayes optimal solutions in settings *with generalization* is typically NP-hard.


There are some well known computational approximations to the Bayes-optimal solution, most notably those involving [Monte-carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).
However, these approximations typically do not maintain strong regret bounds for finite computation approximations and may fail spectacularly badly when they are implemented with finite resources.

Even if we were able to use infinite compute power, there is a fundamental limit even on the independent $K$-armed bandit that we can never get a regret bound better than:

$$ {\rm Regret}(T, \pi) = O(\sqrt{KT}) $$

for all possible configurations of the arms $1,..,K$.
The Bayes optimal solution acheives this through prioritizing exploration to arms which are either informative or promising or both, rather than $\epsilon$-greedy which is extremely wasteful in its exploration.
We will now look at a family of algorithms that seek to direct their exploration more efficiently than $\epsilon$-greedy but with far lower computational cost than the Bayes-optimal solution.


### Algorithm 4: Be optimistic

### Algorithm 5: Probability matching (aka Thompson sampling)


