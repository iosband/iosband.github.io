---
layout: post
title: Randomized Prior Functions for Deep Reinforcement Learning
date: 2018-11-30
---


So it turns out that keeping up full time job, a blog, [twitter](https://twitter.com/ianosband?lang=en) and writing papers takes a lot of time... so much so that we're now three years on from my previous post... wow... well I guess they do say time flies when you're having fun.

### NeurIPS spotlight

As part of the run up to [NeurIPS 2018](https://neurips.cc/) I thought I'd dust off the cobwebs and put a little something up about our latest submission [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335) (with John Aslanides and Albin Cassirer).
Now I am quite biased, but I think this is a **nice little paper that identifies a key shortcoming and then presents a simple + effective solution.**
To summarize:

- Our previous work suggested using [bootstrapped ensembles](https://arxiv.org/abs/1602.04621) to estimate uncertainty with neural networks.
- But if rewards are very sparse (e.g. all observed rewards zero), then bootstrapping the targets will not generate any diversity... and so how is the algorithm going to learn anything?
- This makes a lot of sense... but if you want your agent to do anything useful in zero-reward regimes then you need some kind of [prior effect](https://arxiv.org/abs/1507.00300): an intrinsic motivation that tells the agent that there *might* be something good out there!
- **Our solution is simple: add a random untrainable "prior" function to each member of the ensemble.**

$$ \color{red}{\overbrace{Q_{\theta_k}(x)}^{\text{model k}}} 
		= \overbrace{f_{\theta_k}(x)}^{\text{trainable}} 
		+ \color{blue}{\overbrace{p_k(x)}^{\text{prior function}}} $$


In order to make this work more accessible we have actually made a dedicated website at [https://bit.ly/rpf_nips](https://bit.ly/rpf_nips).
This includes all the links to the paper, poster, presentations, videos and even a demo [colab](https://colab.sandbox.google.com/drive/1hOZeHjSdoel_-UoeLfg1aRC4uV_6gX1g) where you can run the code direct in the browser!

**We've put a lot of work into the paper and [accompanying website](https://bit.ly/rpf_nips), so if you are interested I hope you check it out...**

---------------------
---------------------


### Building intuition

OK so, first things first, I suggest you just skip this entire blog and head right over to the [colab](https://colab.sandbox.google.com/drive/1hOZeHjSdoel_-UoeLfg1aRC4uV_6gX1g) - that way you can follow along with the *actual* code and even tweak it yourself in the browser!
That said, in case you stumbled onto my site and just wanted a little summary, then I'm going to reproduce a little piece of that work here in the hope it will entice you to head over and find out more.

"Prior networks" break down the function you are learning into:

$$ \color{red}{\overbrace{Q_{\theta_k}(x)}^{\text{model k}}} 
		= \overbrace{f_{\theta_k}(x)}^{\text{trainable}} 
		+ \color{blue}{\overbrace{p_k(x)}^{\text{prior function}}} $$

In this section we visualize the effects of training $f_\theta$ = (20, 20)-MLP  but with different choices of prior function $p$.

We will see that, given a flexible enough architecture $f$, for any $p$:
- The resultant $Q=f+p$ will be able to fit all of the observed data $(x,y)$.
- Away from the training data, generalization will be impacted by the interaction of $f$ with $p$.

As such, we say that $p$ plays a role similar to a "prior" in Bayesian inference: it controls how the agent generalizes in areas *without* data... whereas regions *with* data are still perfectly able to fit their observations.
In fact, this superficial similarity can be made much more rigorous for the simple case of linear regression.
For more on this connection please see Section 3 of [our paper](https://arxiv.org/abs/1806.03335)!


#### 1D regression experiment

We now  perform a simple experiment:
1. Fix a dataset of training data $(x,y)$
2. Initialize a prior function $p$.
3. Fit $Q_\theta = f_\theta + p$ to this data by optimizing $\theta$.

The following plots demonstrate how the prior function alters behaviour:
- Training data in **black** points $(x,y)$.
- Trainable function $f$ is the **dashed** line.
- Fixed (untrainable) prior function $p$ in <font color='blue'>**blue**</font>.
- Resultant prediction $Q = f + p$ in <font color='red'>**red**</font>.

<center>
<div>
 <img src="https://i.imgur.com/2llwS1z.png" alt="1d_regression" style="width:750px">
</div>
</center>

The plots show that, for different choices of prior function:
- The neural network can still fit the data at observed data points $(x,y)$.
- The generalization behaviour away from the data is different for different choices of $p$.


### Approximate posterior distribution via ensemble

The section above outlines our approach to training a network with prior function.

In [our paper](https://arxiv.org/abs/1806.03335) we make the connection between:
1. Generating a single posterior sample
2. Training a network with randomized prior function

In fact, for linear networks, the two are precisely equivalent.


However, in many settings, a single posterior sample is not enough... we want to be able to approximate the *entire distribution* of posterior beliefs.
In this setting, our approach is very simple... just take multiple samples and use the resulting ensemble as an approximation to your posterior.

To demonstrate this effect, we perform a very simple experiment:
- Random prior $p_k$ is a (20,20)-MLP according to standard Glorot initialization for $k=1,..,10$.
- Train $Q^k_\theta = f^k_\theta + p_k$ for $k=1,..,10$
- Use the resultant $\{Q^k\}$ as an approximation to the posterior.

The plots below visualize the resultant $Q, f, p$ for $k=1,..,10$.

You can see that all the functions fit the training data, but they do not generalize in exactly the same way.

<center>
<div>
 <img src="https://i.imgur.com/nQVs73Z.png" alt="1d_ensemble" style="width:750px">
</div>
</center>


We can combine the ensemble predictions by taking the mean and standard deviation of predictions $\{Q^k(x)\}$ at each $x$.
- Training data $(x,y)$ are black dots.
- Blue line represents the mean prediction
- Shaded grey bands at $\pm 1, 2$ standard deviations.

<center>
<div>
 <img src="https://i.imgur.com/Tpxiay4.png" alt="ensemble_conf" style="width:750px">
</div>
</center>

We feel that, intuitively, this represents a reasonable approximation to posterior uncertainty.

#### Adding extra noise to the data

In problems with observation noise, the ensemble procedure above will fit to every data point exactly... and this may not be appropriate!

Analogous to the Bayesian derivation of [Section 3](https://arxiv.org/abs/1806.03335), we should add noise to the targets $y$ similar to the type of noise we expect to see with our data.
- If we know the scale of the noise, we can simply add noise in that family.
- If we do not know the noise distribution, bootstrapping offers a non-parametric way to estimate it.
https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

However, once you have enough data, the effects of bootstrapping (or adding noise) will wash out eventually.

Our next series of plots repeats the ensemble procedure with bootstrapping *and* prior functions.


<center>
<div>
 <img src="https://i.imgur.com/4q3meJq.png" alt="bootstrap" style="width:750px">
</div>
</center>

And visualizing the ensemble

<center>
<div>
 <img src="https://i.imgur.com/WDqhBuj.png" alt="bootstrap" style="width:750px">
</div>
</center>

Note that, as we gather more data the bootstrap ensemble will still eventually concentrate around the mean of observed data.

------------
------------

## So what?

Well, if you've made it this far I really am going to have to point you to our [paper website](https://sites.google.com/corp/view/randomized-prior-nips-2018/).
There, you'll be able to find more details on the algorithm, and more motivation for why we really need something like this in Deep RL 2018.


**We even have some videos that explain why this simple modification can lead to exponentially faster reinforecement learning!**
<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/J6I0GXyFaUk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>


So, with that final flurry I will end this little teaser.
If you end up looking at any of this and you have any questions, please either write in the comments below or reach out to me on twitter [@ianosband](https://twitter.com/ianosband?lang=en).
This is particularly interesting if you think I'm missing something in my work, or you have some suggestions on how we can make it better!
Let's hope it's not 3 more years until my next blog post...