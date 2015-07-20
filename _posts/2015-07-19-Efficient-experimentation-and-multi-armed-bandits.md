---
layout: post
title: Efficient experimentation and the multi-armed bandit
date: 2015-07-19
---


In my [last post](http://iosband.github.io/2015/07/05/ian-launches-blog.html) I introduced the problem of *exploration vs exploitation*.
In particular, we used the example of a doctor trying to save lives using some medical procedures.
For each patient the doctor can either try out a new drug in order to gather more data (exploration) or she can use the drug which seems to be the best given the data she already has (exploitation).
This sort of tradeoff is not just about casinos, but it shows up in a lot of places.
In the academic literature this is often referred to as a [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem.
This week I'm going to to define the problem more concretely and try to give some insight into how we to tackle this effectively.

**Note:** This was my first proper blog post and the length/mathematics has definitely gotten out of hand.
I'll be following up this post with some code and fun demos within a month, which should make this more interesting!
If you find something weird, post your questions to the forum at the bottom and I'll do my best to sort it out asap.
Cheers.

## Multi-armed bandit

What is a multi-armed bandit? Well... here are a few types:

<center>
<div>
 <img src="http://www.rmtracking.com/blog/wp-content/uploads/2011/11/iStock_000014773673XSmall.jpg" alt="thief" style="height:170px">
 <img src="http://epguides.com/Zorro_1957/cast.jpg" alt="zorro" style="height:170px">
 <img src="http://research.microsoft.com/en-us/projects/bandits/MAB-2.jpg" alt="mab" style="height:170px">
</div>
</center>

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
At each timestep there's a tradeoff between trying an arm which appears to give high rewards (exploitation) and trying out an arm which you haven't seen many times in the hope of learning more (exploration).
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

<center>
<img src="http://homepages.inf.ed.ac.uk/amos/figures/gpprior.png" alt="family" style="height:250px">
</center>


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

### The Markov assumption

Although the multi-armed bandit (or Bayesian optimization) as stated above is a very flexible tool for solving sequential decision problems, it does rely very heavily on the so-called [Markov assumption](https://en.wikipedia.org/wiki/Markov_property).

In short, this means that the unknown function $ f^* $ we are trying to optimize stays constant through time and is not influenced by our previous attempts to optimize it.
In other words, even though we don't assume we know the probability of payout of each arm, **we do assume there is a fixed probability of success independent of our choices**.
In many practical settings this is a *huge* oversimplication... if you're a doctor treating a patient and you give them a treatment that kills them, then they'll pretty much stay dead no matter what you try!

My own research actually focuses on settings like this, where this may be some long term consequences for your actions.
This field is called [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) and it's really cool.
I'm definitely intending to follow up on this extension in a future blog post, but before we do that it really helps to understand some of the issues in the bandit setting.
So, for now let's just keep things simple and assume the Markov assumption is a reasonable approximation to the truth... and for many examples it is!


### Some notation

The last thing before we dive in we'll need just a little bit more notation:

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

<center>
<div>
 <img src="http://img.humorsharing.com/media/images/1212/i_weird_families_007_50e050e7c7542.jpg" alt="family" style="height:200px">
 <img src="https://s-media-cache-ak0.pinimg.com/236x/7c/55/ab/7c55ab425cf2cd1c264277a8792235d8.jpg" alt="mullet" style="height:200px">
  <img src="https://fbcdn-sphotos-e-a.akamaihd.net/hphotos-ak-xpf1/v/t1.0-9/11021099_10203352168477639_168498448054596353_n.jpg?oh=e2e9c5cc3cae9b1cbf6c1b1d88011aa5&oe=5648ABAB&__gda__=1448696212_3640086f4dd495c1a11438b99b32f626" alt="ian" style="height:200px">
 <img src="https://s-media-cache-ak0.pinimg.com/736x/e0/6c/88/e06c880bcbe59e9d1faeca1019858f85.jpg" alt="cats" style="height:200px">
</div>
</center>


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

<center>
<img src="http://cdn.baekdal.com/2011/scrooge.jpg" alt="scrooge" style="height:200px">
</center>

**For each timestep $t$**:

$$\text{Estimate } \hat{f}_t = \mathbb{E} [f^* | H_t]$$

$$\text{Choose }  x_t \in \arg \max_x \hat{f}_t $$

Intuitively it might seem like this is the best thing you can do... but over the long run it's not.
The problem is that may be choosing actions which do not increase your future understanding of the system.
By replacing the unknown $ f^* $ with your point estimate $\hat{f}_t$ you overstate your knowledge.
In fact, you cannot even guarantee that this algorithm will *ever* learn the correct policy.

I think that this is one of the most commonly overlooked points in machine learning at the moment.

>If you implement a system that makes *optimal* decisions based on the data it has, even if it's with a super-fancy [deep neural network](https://en.wikipedia.org/wiki/Deep_learning) trained **to perfection**.
>The resulting decision making system might be very very far from optimal in the long run.

If you want to make good decisions for life, you need to take actions that will help to increase your understanding... and this isn't specific to machine learning - it applies to all types of learning!

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
Simply stated, the agent should formulate their posterior beliefs and then solve for the action with the highest expected long term rewards.

<center>
<div>
 <img src="http://www.theironsamurai.com/wp-content/uploads/2014/11/thomas-bayes.png" alt="" style="height:200px">
 <img src="https://upload.wikimedia.org/wikipedia/commons/1/18/Bayes'_Theorem_MMB_01.jpg" alt="" style="height:200px">
</div>
</center>
**For each timestep $t$**:

$$\text{Evaluate posterior } \phi(\cdot | H_t) \text{ for } f^* $$

$$\text{Solve for } x_t \in \max_x \mathbb{E}_\phi \mathbb{E} \left[ \sum_{j=t}^T f^*(x_j) \right]$$

In words, the agent works out exactly how profitable they expect each action to be **in the long run** and then picks the action with the highest expected value.
This characterises the optimal solution very concisely and in the case of independent arms this can even be computed effectively by the results of [Gittins indices](https://www.google.co.uk/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=gittins%20index).
However, the general problem of computing Bayes optimal solutions in settings *with generalization* is NP-hard...
that means really really hard!

> Why is computing the Bayes optimal solution so difficult?

The problem with the Bayes optimal solution is that you need to consider not only your understanding of the action you choose.
You also need to consider what you might learn about the function from this action and how that will affect *all* of your subsequent choices.
To do this properly requires building up a huge exponential search tree under all possible outcomes.
Even for simple problems it is generally just not possible to compute this exactly.

Now, there are some well known computational approximations to the Bayes-optimal solution, most notably those involving [Monte-carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).
However, these approximations typically do not maintain strong regret bounds for finite computation approximations and may fail spectacularly badly when they are implemented with finite resources.

Finally, even if we were able to use infinite compute power, there is a fundamental limit even on the independent $K$-armed bandit that we can never get a regret bound better than:

$$ {\rm Regret}(T, \pi) = \Omega(\sqrt{KT}) $$

for all possible configurations of the arms $1,..,K$.
There are similar lower bounds on the regret for systems with generalization.
For example, in the case of an unknown $d$-dimensional linear function $ f^* : \mathbb{R}^d \rightarrow \mathbb{R} $ you can never guarantee better than

$$ {\rm Regret}(T, \pi) = \Omega(\sqrt{dT}) $$

with similar sorts of results for other more general function classes $ f^* \in \mathcal{F} $.


I don't want to get too bogged down in the mathematics here,  but there is a big picture point to take away.
The Bayes optimal solution acheives good performance through **prioritizing exploration to arms which are either informative or promising or both**, unlike $\epsilon$-greedy which is extremely wasteful in its exploration.
We will now look at a family of algorithms that seek to direct their exploration more efficiently than $\epsilon$-greedy but with far lower computational cost than the Bayes-optimal solution.


### Algorithm 4: Be optimistic

Our goal is to optimize some function $ f^* $, the problem is that we don't know what this function actually *is*.
Now, something like $\epsilon$-greedy will *eventually* learn the function $ f^* $ through random measurements and therefore *eventually* we'll learn the optimal action.
However, we don't actually care about the whole function $ f^* $, we only care about finding its maximum.
The idea behind this next algorithm is that we should focus our efforts only on actions which *could potentially* be the best action.

<center>
<img src="http://mygreenfuture.org/wp-content/uploads/2015/06/optimism_0.png" alt="family" style="height:250px">
</center>

The idea is that, at each timestep you can use the data you've seen to narrow down the possible $ f^* $ you might consider.
For example, if you're in the casino and you've received a reward from arm one in the last go, you know that it's reward probability $ p_1 > 0 $.
Now, once you've formed this set of all possible functions $ \mathcal{F}_t $, you choose the action which is the *most optimistic* over any of the possible functions.
This means that you pick the action which would have the highest expected return in the best possible case...

**For each timestep $t$**:

$$\text{Form a set } \mathcal{F}_t \text{ which contains the true } f^* \text{ with high probability given } H_t $$

$$\text{Choose the most optimistic outcome } x \in \arg_x \max_{x \in \mathcal{X}, f \in \mathcal{F}_t} f(x) $$

The reason this algorithm is good is because, if you do it correctly, you'll never discount any action which *could* have been the best action.
At the same time, you will focus your energy on the most promising possible actions, so that you should be able to learn more quickly than if you just shared your exploration randomly.

This approach has gotten the lion's share of the attention in the academic literature and with good reason.
In the case of independent arms the [UCB algorithm](http://link.springer.com/article/10.1007/s10998-010-3055-6) gave a simple and tractable solution with some great theoretical guarantees.
More or less, **instead of picking the arm with the highest *expected* return, you pick the arm with the highest mean PLUS three standard deviations**.
This incentivizes you to go and explore promising actions which may have been unlucky on your first few attempts.
However, pretty quickly you should concentrate on only the best arms.

The general strategy for dozens of papers on modified UCB-style algorithms is to find new function classes $ \mathcal{F} $ and design some efficient optimistic strategy that has good regret bounds.
To show the regret bound, you consider the regret at any one step.
The hard part about this proof is that you never actually *know* what the optimal $ f^*( x^*) $ is... if you did it wouldn't be a bandit problem.
However, the optimism allows you to bound this difference by the different to your *imagined* optimistic optimum, which must be at least as large since $f^* \in \mathcal{F}_t$.

$$ f^*(x^*) - f^*(x_t) = \underbrace{\max_{x_t \in \mathcal{X}, f \in \mathcal{F}_t} f(x_t) - f^*(x_t)}_{\text{Concentrates with data}} + \underbrace{f^*(x^*) - \max_{x_t \in \mathcal{X}, f \in \mathcal{F}_t} f(x_t)}_{\le 0 \text{ by optimism}} $$

Working through the maths and choosing your algorithm correctly, in several settings some smart people have designed optimistic algorithms that get a good regret bound.
For independent $K$-armed bandit [UCB](http://link.springer.com/article/10.1007/s10998-010-3055-6):

$$ {\rm Regret}(T, \pi) = O(\sqrt{KT}) $$

And in $d$-dimensional [linear bandits](http://papers.nips.cc/paper/4417-improved-algorithms-for-linear-stochastic-bandits):

$$ {\rm Regret}(T, \pi) = O(d\sqrt{T}) $$

which is pretty great - we recovered something that we can guarantee is *close* to the optimal solution and it's a computationally tractable algorithm.
We can be confident if we run this algorithm on any problem that we will end up doing almost as well as we could have done... even though we didn't need to know the true underlying function - awesome.
Unfortunately, there is just one small problem...

Both of the steps in our algorithm depend on some maths-type to come along and tell us:

- How do we construct the optimistic set $\mathcal{F}_t$ given data $H_t$
- How do we solve the **double** maximization $ \max_{x \in \mathcal{X}, f \in \mathcal{F}_t} f(x) $?

In general, one or *both* of these steps might be computationally intractable!
If you're an expert in bandit theory, every year you might be able to generate a new algorithm for a new function class $ f^* \in \mathcal{F} $... but it's a lot of hard work and it's unclear whether anyone would even be able to use it in practice.

Fortunately, there is another simple algorithm that shares a lot of benefits of well-designed optimistic exploration... but with several benefits.

### Algorithm 5: Probability matching (aka Thompson sampling)

The final algorithm I'm going to introduce is my personal favourite: *probability matching*.
Simply stated, **the algorithm choose an action randomly, according to the probability it is optimal**.

<center>
<img src="http://www.betonhumour.com/wp-content/uploads/2015/05/dice_roll_img.jpg" alt="" style="height:200px">
</center>

This takes some of the ideas from the Bayesian solution, we should quantify our beliefs about the system in terms of a *distribution* rather than just a point estimate.
However, the computational cost of this algorithm will be so much less!

**For each timestep $t$**:

$$\text{Take a single sample } f_t \sim \phi(\cdot | H_t) \text{ posterior distribution for } f^* $$

$$\text{Solve for } x_t \in \max_x f_t(x) \text{ maximum for that sample. }$$

Now, because the algorithm is implemented by taking a single posterior sample and optimizing that sample it has been called [posterior sampling](http://arxiv.org/abs/1301.2609) in the literature.
However, because this sort of randomization was originally proposed by Thompson in 1933 the name [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) is most commonly used.

Personally, I don't like the name Thompson sampling, I think everything makes a lot more sense if we name things after what they *are* rather than the *people* who invented them...
I've had several conversations with [Rich Sutton](http://webdocs.cs.ualberta.ca/~sutton/) where we have lamented this convention... nevertheless I think it's time to admit the "Thompson sampling" name has stuck.

Now why is Thompson sampling such a good algorithm for bandit problems:

1. It's simple to explain.
2. You don't need to be a mathematician to implement it on *your* specific problem.
3. It is computationally cheap.
4. It works well [in practice](http://www.research.rutgers.edu/~lihong/pub/Chapelle12Empirical.pdf).
5. It applies naturally to systems with generalization.
6. It shares many of the *analytical guarantees* of the best possible optimistic algorithm!

I think the most surprising out of these observations is point (6), which has only been realised in the last few years.
People had tried Thompson sampling because it was easy and seemed to work well... but now we're starting to understand *why* it works and also provide some really solid **guarantees**.

To see why this is the case, the key observation is that conditional on any data $H_t$ the true function $ f^* $ and the *imagined* function $ f_t $ are drawn from exactly the same distribution.
This means that:

$$ \mathbb{E}[ f^*(x^*)] = \mathbb{E} \left[ \mathbb{E}[ f^*(x^*) | H_t] \right]=
   \mathbb{E} \left[ \mathbb{E}[ f_t(x_t) | H_t ] \right] = \mathbb{E}[ f_t(x_t)] $$

This relationship allows us to convert a statement about an unknown optimal reward of an action we'll never know into a standard estimation problem at the action we do actually choose $ f_t(x_t) $.

$$ f^*(x^*) - f^*(x_t) = \underbrace{f_t(x_t) - f^*(x_t)}_{\text{Concentrates with data}} + \underbrace{f^*(x^*) - f(x_t)}_{\mathbb{E} = 0 \text{ by construction}} $$

Following through exactly the same type of analysis as for the UCB-style we recover the same regret bounds for independent $K$-armed bandit:

$$ {\rm Regret}(T, \pi) = O(\sqrt{KT}) $$

And in $d$-dimensional linear bandits:

$$ {\rm Regret}(T, \pi) = O(d\sqrt{T}) $$

The cool thing here is that we've been able to deal with things at a higher level of abstraction when compared to the optimistic algorithm.
This means that our proofs will go through for *any* type of prior structure.
These results above are actually just special cases that for *any* type of function class $ f^* \in \mathcal{F} $ we can get an elegant regret analysis in terms of the **dimension of the function class** $ {\rm dim}(\mathcal{F}) $.


$$ {\rm Regret}(T, \pi) = O(\sqrt{ {\rm dim} (\mathcal{F}) T}) $$


What is this "dimension"... ahh well that's a topic for another time...
This result and several others is proven in the great paper by my friend [Dan Russo](http://web.stanford.edu/~djrusso/):

- [Learning to optimize via posterior sampling](http://web.stanford.edu/~djrusso/docs/Learning_to_Optimize.pdf)

If you've taken the time to read this far I can really recommend going through the paper, which covers a lot of the material in this post but in a more precise way.
I've already gone into way more maths/proof than I was hoping to cover...

## Time to wrap it up!

In this post I've tried to introduce the problem of the multi-armed bandit and also give some popular approaches to solving the problem.
I think the key point to take away is that if you want to do learning *with actions* then the problem can become substantially different to standard statistics or supervised learning.

>To learn well in the long run, you need to consider your uncertainty and take actions that explore efficiently.

Even if you have the most awesome deep neural network architecture imagineable, if you use this to make decisions in your learning system without efficient exploration you might never learn to behave well.
This isn't a case of fiddling with $\epsilon$ or the learning rate, it's a fundamental problem in decision making.

- **If you want to make good decisions from data, you need good data.**
- **If the decisions you take affect the data you get, you need to choose your actions to get good data.**

This is the reason that exploration and *reinforcement learning* are so important, they're not just a simple extension of results from supervised learning, but something fundamentally more challenging.

In my next post I'm planning some fun demos (with accompanying code) so you can try these things out for yourself!
Also, I'll hope to show more clearly the cases where [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) outperforms naive methods like $\epsilon$-greedy but also some examples where it is falls way short of the Bayes-optimal solution.
Actually, I was planning on doing that this time around, but made the classic mistake of writing too much on your first post...
This has already become something of a monster.
If you have any questions/comments hit me up below and we can go from there.
See you later.





