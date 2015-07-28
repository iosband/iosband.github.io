---
layout: post
title: Can you beat the bandit?
date: 2015-07-30
---

[Last time](http://iosband.github.io/2015/07/19/Efficient-experimentation-and-multi-armed-bandits.html) we went through the basic problem known as the "multi-armed bandit".
If you're not totally familiar with the subject, or just want a refresher then I'd suggest you start by reading that post.
I know, I know... ain't nobody got time to be reading old posts.
The main idea is:

> If you want to design a good learning algorithm you must balance **exploration** (learning more about the world around you) with **exploitation** (making good decisions based on what you already know).

This good news this week is that we're going to cut out the maths and you're going to get a chance to try these algorithms out for yourselves; *very* exciting.
Many thanks to [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) who is an absolute superstar and helped me get up and running with Javascript for this demo.

## Your mission

Imagine yourself three months from now:
you've subscribed to my blog and even read all the maths parts that you skipped the first time through.
Well, now that you're now an expert in sequential decision making (and updated your [LinkedIn profile](https://www.linkedin.com/in/iosband) to say so too) **you are hired as head of drug development at a prestigious research hospital**.
Life's going great, the sign on bonus was *insane*, but now you find yourself in the middle of a catastrophic medical emergency.

<center>
<img src="https://cbskearth101.files.wordpress.com/2014/01/457763331.jpg?w=620&h=349&crop=1" alt="bieber" style="height:300px">
</center>

With the release of his new album [Bieber fever](http://www.bieberfever.com/) has mutated to become lethal.
Far and wide people are bopping to ["baby, baby, baby"](https://www.youtube.com/watch?v=kffacxfA7G4) until their hearts give out in a modern day [dancing plague](https://en.wikipedia.org/wiki/Dancing_Plague_of_1518)...
The world's foremost medical researchers have developed several drugs to combat the disease.
Unfortunately, nobody knows how well these treatments work.
**Now it's up to you to save as many lives as possible**.

Luckily, you recognise this as nothing more than an [independent bernoulli bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)... no problem!

In the simulation below you'll choose how many drugs you have to choose from *and* how many patients you'll be dealing with.
Once you press **Play** you'll be able to simulate this sequential experiment problem for yourself.
At the end (or whenever you press **Plot**) you will be able to see the true success probabilities of each drug and compare your performance to that of several basic algorithms:

- Greedy
- Epsilon-Greedy
- UCB
- Posterior Sampling

<center>
    <h2> Can you beat the bandit algorithms? </h2>
</center>



<div id="wrap">
    <h3>Configuration</h3>
<div class="button-container">
<style media="screen" type="text/css">
.button-container form,
.button-container form div {
    display: inline;
}

.button-container button {
    display: inline;
    vertical-align: middle;
}
#wrap {
    border: 1px solid black;
    padding: 10px;
    margin: 10px;
}
</style>

    <strong>Drugs:</strong>
    <form>
        <input type="range" name="arm_form" min="0" max="50" value="4" oninput="this.form.arm_input.value=this.value" id="drugs_slider" />
        <input type="number" name="arm_input" min="0" max="50" value="4" oninput="this.form.arm_range.value=this.value" />
    </form>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Patients:</strong>
    <form>
        <input type="range" name="step_form" min="0" max="10000" value="40" oninput="this.form.step_input.value=this.value" id="patients_slider"/>
        <input type="number" name="step_input" min="0" max="10000" value="40" oninput="this.form.step_form.value=this.value" />
    </form>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <button type="button" class="button" onclick="gen_bandit_problem($('#drugs_slider').val())">Play</button>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <button type="button" class="button" onclick="gen_plot()">Plot</button>
</div>

</div>


<div id="wrap">
<style media="screen" type="text/css">
.butt {
    width:50px;
    height:50px;
    margin: 10px;
}
#wrap {
    border: 1px solid black;
    padding: 10px;
    margin: 10px;
}
#wrap2 {
    padding: 10px;
    margin: 10px;
}
</style>

<script>
//------------------------------------------------------------------------------
// Generating random Beta in Javascript (taken from StackExchange)

// javascript shim for Python's built-in 'sum'
function sum(nums) {
  var accumulator = 0;
  for (var i = 0, l = nums.length; i < l; i++)
    accumulator += nums[i];
  return accumulator;
}


// From Python source, so I guess it's PSF Licensed
var SG_MAGICCONST = 1 + Math.log(4.5);
var LOG4 = Math.log(4.0);
var SQRT2PI = Math.sqrt(Math.PI*2);
function pgamma(z) {
  // Reflection to right half of complex plane
  if (z < 0.5) {
    return Math.PI / Math.sin(Math.PI*z) / pgamma(1 - z);
  }
  // Lanczos approximation with g=7
  var az = z + 6.5;
  return Math.pow(az, (z - 0.5)) / Math.exp(az) * SQRT2PI * sum([
    0.9999999999995183,
    676.5203681218835 / z,
    -1259.139216722289 / (z+1.0),
    771.3234287757674 / (z+2.0),
    -176.6150291498386 / (z+3.0),
    12.50734324009056 / (z+4.0),
    -0.1385710331296526 / (z+5.0),
    0.9934937113930748e-05 / (z+6.0),
    0.1659470187408462e-06 / (z+7.0)
  ]);
}

function rgamma(alpha, beta) {
  // does not check that alpha > 0 && beta > 0
  if (alpha > 1) {
    // Uses R.C.H. Cheng, "The generation of Gamma variables with non-integral
    // shape parameters", Applied Statistics, (1977), 26, No. 1, p71-74
    var ainv = Math.sqrt(2.0 * alpha - 1.0);
    var bbb = alpha - LOG4;
    var ccc = alpha + ainv;

    while (true) {
      var u1 = Math.random();
      if (!((1e-7 < u1) && (u1 < 0.9999999))) {
        continue;
      }
      var u2 = 1.0 - Math.random();
      v = Math.log(u1/(1.0-u1))/ainv;
      x = alpha*Math.exp(v);
      var z = u1*u1*u2;
      var r = bbb+ccc*v-x;
      if (r + SG_MAGICCONST - 4.5*z >= 0.0 || r >= Math.log(z)) {
        return x * beta;
      }
    }
  }
  else if (alpha == 1.0) {
    var u = Math.random();
    while (u <= 1e-7) {
      u = Math.random();
    }
    return -Math.log(u) * beta;
  }
  else { // 0 < alpha < 1
    // Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
    while (true) {
      var u3 = Math.random();
      var b = (Math.E + alpha)/Math.E;
      var p = b*u3;
      if (p <= 1.0) {
        x = Math.pow(p, (1.0/alpha));
      }
      else {
        x = -Math.log((b-p)/alpha);
      }
      var u4 = Math.random();
      if (p > 1.0) {
        if (u4 <= Math.pow(x, (alpha - 1.0))) {
          break;
        }
      }
      else if (u4 <= Math.exp(-x)) {
        break;
      }
    }
    return x * beta;
  }
}

// like betavariate, but more like R's name
function rbeta(alpha, beta) {
  var alpha_gamma = rgamma(alpha, 1);
  return alpha_gamma / (alpha_gamma + rgamma(beta, 1));
}

//------------------------------------------------------------------------------
// Basic bandit functions
// All of this is specific to independent bernoulli bandits

function init_counts(n_arms) {
    // init_counts is a function to initialize empty counts for the bandit
    // algorithm of with n_arms independent arms.
    //
    // Args:
    //  n_arms - int - number of arms for bandit
    //
    // Returns:
    //  arm_counts - n_arms x 2 - array of counts set to zero
    var arm_counts = [];
    for (i = 0; i < n_arms; i++) {
        arm_counts.push([0, 0]);
    }
    return arm_counts;
}

function init_arms(n_arms) {
    // init_arms is a function to initialize random uniform probalities for
    // the arms in the bandit problem
    //
    // Args:
    //  n_arms - int - number of arms for bandit
    //
    // Returns:
    //  p_true - n_arms x 1 - array of success probabilities (unknown to agent)
    var p_true = [];
    for (i = 0; i < n_arms; i++) {
        p_true.push(Math.random());
    }
    return p_true;
}

function pull_arm(p_true, action) {
    // pull_arm is a function to evaluate one step of the bandit algorithm
    //
    // Args:
    //  p_true - n_arm x 1 - array of success probabilities
    //  action - int - choice of arm to pull
    //
    // Returns:
    //  reward - int - binary outcome of slot machine
    if (Math.random() < p_true[action]) {
        return 1;
    } else {
        return 0;
    }
}

function run_bandit(p_true, bandit_alg, t_steps) {
    // run_bandit runs a bandit algorithm for t_steps
    //
    // Args:
    //  p_true - n_arm x 1 - array of success probabilities
    //  bandit_alg - funciton - bandit algorithm to use
    //
    // Returns:
    //  cum_rewards - t_step x 2 - (cumulative rewards, choices) through time.
    var cum_rewards = [];
    var arm_counts = init_counts(p_true.length);

    for (t = 0; t < t_steps; t++) {
        choice = bandit_alg(arm_counts, t+1);
        reward = pull_arm(p_true, choice);
        if (t === 0) {
            cum_rewards.push([reward, choice]);
        } else {
            cum_rewards.push([reward + cum_rewards[t - 1][0], choice]);
        }
        arm_counts[choice][1 - reward] += 1;
    }
    return cum_rewards;
}

//------------------------------------------------------------------------------
// Bandit algorithms (e-greedy, posterior sampling and UCB)
// Note that for ease of transfer we have added the dummy timestep variable
// for ease of transfer in the run_bandit algorithm

function ps_choice(arm_counts) {
    // ps_choice uses posterior sampling to make a choice of arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    var prior_a = 1
    var prior_b = 1
    for (i = 0; i < arm_counts.length; i++) {
        var p_sample = rbeta(arm_counts[i][0] + prior_a,
                             arm_counts[i][1] + prior_b)
        if (p_sample > p_max) {
            p_max = p_sample
            choice = i
        }
    }
    return choice
}

function greedy_choice(arm_counts) {
    // greedy_choice uses the greedy empirical estimate to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    for (i = 0; i < arm_counts.length; i++) {
        var n_pull = (arm_counts[i][0] + arm_counts[i][1])
        if (n_pull < 0.5) {
            n_pull = 1
        }
        var p_hat = arm_counts[i][0] / n_pull
        if (p_hat > p_max) {
            p_max = p_hat
            choice = i
        }
    }
    return choice
}

function egreedy_choice(arm_counts) {
    // egreedy_choice uses epsilon-greedy to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //  epsilon - double - probability of random action - now fixed within
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var epsilon = 0.1
    var choice = -1
    if (Math.random() < epsilon) {
        choice = Math.floor((Math.random() * arm_counts.length));
    } else {
        choice = greedy_choice(arm_counts)
    }
    return choice
}

function ucb_choice(arm_counts, timestep) {
    // ucb_choice uses the UCB algoirthm to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //  timestep - int - number of timesteps elapsed
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    for (i = 0; i < arm_counts.length; i++) {
        var n_pull = (arm_counts[i][0] + arm_counts[i][1])
        if (n_pull < 0.5) {
            n_pull = 1
        }
        var p_hat = arm_counts[i][0] / n_pull
        var p_upper = p_hat + Math.sqrt(Math.log(timestep) / n_pull)
        if (p_upper > p_max) {
            p_max = p_upper
            choice = i
        }
    }
    return choice
}

//------------------------------------------------------------------------------
// Now let's do a simple implementation of the bandit problem

// function gen_bandit(n_arms)

n_arms = $("#drugs_slider").val()
p_true = init_arms(n_arms)
t_steps = $("patients_slider").val()

ps_rewards = run_bandit(p_true, ps_choice, t_steps)
ucb_rewards = run_bandit(p_true, ucb_choice, t_steps)
egreedy_rewards = run_bandit(p_true, egreedy_choice, t_steps)
greedy_rewards = run_bandit(p_true, greedy_choice, t_steps)



function gen_arm_str(arm){
    // gen_button_str is a function to generate the hmtl for a single arm
    var button_str = '<button type="button" class="butt" onclick="clicked(' +
                      arm + ')"> Drug ' + (arm + 1) + '</button>';
    return button_str;
}

function clear_plt() {
    // clear_plt is a function to clear the output of the previous simulation

}

function gen_bandit_problem(n_arms) {
    // gen_bandit_problem generates a new random instance of the bandit
    // problem. Generates HTML for the arms and also simulates performance
    // of the other algorithms.

    // Generate the HMTL
    var html_str = '<h3>Medical testing </h3>'
    for (i = 0; i < n_arms; i++) {
        html_str += gen_arm_str(i)
    }
    $("#medical_testing").html(html_str);
    make_guide()
    // $("#click_output").html('Trial results will display here.');
    // $("#rewards_header").html('</br><h3>Performance (lives saved)</h3>');
    // $("#ptrue_header").html('</br><h3>True drug efficacy</h3>');
    // $("#flot_sum").html('Press "Plot" to see output algorithm performance.');
    // $("#legend_holder").html('');
    // $("#flot_p").html('Press "Plot" to reveal true drug efficacy.');

    // Generate the new probabilities
    p_true = init_arms(n_arms)

    // Update the timestep from the patients slider
    t_steps = $("#patients_slider").val()

    // Run the experiments with all the algorithms
    ps_rewards = run_bandit(p_true, ps_choice, t_steps)
    ucb_rewards = run_bandit(p_true, ucb_choice, t_steps)
    egreedy_rewards = run_bandit(p_true, egreedy_choice, t_steps)
    greedy_rewards = run_bandit(p_true, greedy_choice, t_steps)

    // Reset the user variables
    user_t = 0
    user_rewards = [[1,1]]
    user_str = ''
}

// Call this function to populate the screen
gen_bandit_problem($('#drugs_slider').val())

function convert_cumsum(cum_rewards) {
    // convert_cumsum is a helper function to convert cumulative sums
    // of rewards into cumulative rewards for the plot.
    plt_dt = []
    for (i = 0; i < cum_rewards.length; i++) {
        plt_dt.push([i + 1, cum_rewards[i][0]])
    }
    return plt_dt
}

function convert_regret(cum_rewards) {
    // convert_cumsum is a helper function to convert cumulative sums
    // of rewards into regret for the plot.
    p_max = -1
    for (j = 0; j < p_true.length; j++) {
        if (p_true[j] > p_max) {
            p_max = p_true[j]
        }
    }
    plt_dt = [[1, p_max - p_true[cum_rewards[0][1]]]]
    for (i = 1; i < cum_rewards.length; i++) {
        var step_gap = p_max - p_true[cum_rewards[i][1]]
        plt_dt.push([i + 1, plt_dt[i-1][1] + step_gap])
    }
    return plt_dt
}

function make_guide() {
    // make_guide creates the helper text for the blank screens
    $("#click_output").html('Experiment results will show up here once you select a drug to trial.')
    $("#flot_p").html('Press "Plot" to reveal the true drug efficacy.')
    $("#flot_sum").html('Press "Plot" to reveal bandit algorithm performance.')
}

function clear_guide() {
    // clear_guide gets rid of the helper text
    $("#flot_p").html('')
    $("#flot_sum").html('')
    $("#flot_regret").html('')
}


function gen_plot() {
    // clear the helper text
    clear_guide()

    // Plot p_true
    p_plot = []
    for (i = 0; i < p_true.length; i++) {
        p_plot.push([i + 1, p_true[i]])
    }
    data = [{ data:p_plot, label:"Cure probability", bars:{show:true}}]
    options = {legend:{show: false},
               xaxis:{min:1, tickSize:1},
               yaxis:{min:0, max:1, tickSize:0.1}}
    $(document).ready(function(){
    chart = $.plot($("#flot_p"), data, options);
    });

    // Convert performance data
    var ps_plt = convert_cumsum(ps_rewards)
    var ucb_plt = convert_cumsum(ucb_rewards)
    var egreedy_plt = convert_cumsum(egreedy_rewards)
    var greedy_plt = convert_cumsum(greedy_rewards)
    console.log(user_rewards[0])
    if (user_rewards.length > 1) {
        var user_plt = convert_cumsum(user_rewards)
    } else {
        user_plt = [[1, 0]]
    }


    // Plot the data
    data = [{ data:ps_plt, label:"Posterior sampling", lines:{show:true}},
            { data:ucb_plt, label:"UCB", lines:{show:true}},
            { data:egreedy_plt, label:"epsilon-greedy", lines:{show:true}},
            { data:greedy_plt, label:"greedy", lines:{show:true}},
            { data:user_plt, label:"user", lines:{show:true}}];
    options = {legend:{show: true, placement: 'outsideGrid',
                       container:$('#legend_holder')},
               axisLabels: {show: true},
               xaxes: [{axisLabel: 'foo'}],
               yaxes: [{axisLabel: 'bar'}]};
    $(document).ready(function(){
    chart = $.plot($("#flot_sum"),data,options);
    });

    // Convert performance data
    var ps_plt = convert_regret(ps_rewards)
    var ucb_plt = convert_regret(ucb_rewards)
    var egreedy_plt = convert_regret(egreedy_rewards)
    var greedy_plt = convert_regret(greedy_rewards)

    console.log(user_rewards)
    var user_plt = convert_regret(user_rewards)
    console.log(user_plt)

    // Plot the regret
    data = [{ data:ps_plt, label:"Posterior sampling", lines:{show:true}},
        { data:ucb_plt, label:"UCB", lines:{show:true}},
        { data:egreedy_plt, label:"epsilon-greedy", lines:{show:true}},
        { data:greedy_plt, label:"greedy", lines:{show:true}},
        { data:user_plt, label:"user", lines:{show:true}}];
        options = {legend:{show: false}};
    chart = $.plot($("#flot_regret"), data, options);

}


function clicked(choice) {
    // When the arm is clicked a choice is given.
    reward = pull_arm(p_true, choice)
    if (reward == 1) {
        outcome = 'Live'
    } else {
        outcome = 'Die'
    }

    if (user_t === 0) {
        user_rewards = []
        user_rewards.push([reward, choice]);
    } else {
        user_rewards.push([reward + user_rewards[user_t - 1][0], choice]);
    }
    console.log(user_rewards[user_t])

    var extra_output = ''
    if (user_t == t_steps) {
        extra_output = ('Trial complete, please see plot below or start again. <br/>')
        gen_plot()
    } else if (user_t < t_steps) {
        extra_output = ('Timestep: ' + (user_t + 1) +
                        ', Drug: ' + (choice + 1) +
                        ', Patient: ' + outcome + '<br/>')
    }
    user_str = extra_output + user_str
    user_t += 1
    $("#click_output").html(user_str);
}

</script>

<div id="medical_testing">
    <h3>Medical testing</h3>
    Press "Play" to try for yourself or "Plot" to the bandit algorithms.
</div>

</div>


<div id="wrap">
<h3>Trial results</h3>
<div style="overflow: auto; height:150px;" id="click_output">
    Experiment results will show up here once you select a drug to trial.
</div>
</div>

<div id="wrap2">
<div id="ptrue_header"><br><h3>True drug efficacy</h3></div>
<div id="flot_p" style="width:760px;height:300px">
    Press "Plot" to reveal true drug efficacy.
</div>
<div id="rewards_header"><br><h3>Lives saved (higher is better)</h3></div>
<div id="flot_sum" style="width:760px;height:400px"></br>
    Press "Plot" to see algorithm performance.
</div>
<div id="legend_holder" style="width:800px;height:220px"></div>
<div id="regret_header"><br><br><h3>Regret (lower is better)</h3></div>
<div id="flot_regret" style="width:800px;height:400px"></br>
    Press "Plot" to see algorithm performance.
</div>
</div>


-----------------------------------------------------------------------------

## Results

Hopefully you'll have a play around with the simulation above and get a bit of a feel for the performance of these algorithms.
Here are my first takeaways:

- **Being greedy really sucks.**
- Human intuition and $epsilon$-greedy (here $epsilon = 0.1$) aren't bad on the small problems (drugs and/or patients).
- **Once you get into the large-scale problems you start to see the benefits of efficient experimentation**.
- Posterior sampling is consistently one of the best algorithms in simulation.

Isn't it nice when the experiment matches the theory?

<center>
<img src="http://i.huffpost.com/gen/1632851/images/o-DOCTOR-THUMBS-UP-facebook.jpg" alt="doctor" style="height:250px">
</center>

Now, if you're interested in learning more about this I can recommend the paper "[An emprical evaluation of Thompson sampling](http://www.research.rutgers.edu/~lihong/pub/Chapelle12Empirical.pdf)" that really kicked off the recent revival in Thompson sampling research.

Just a few more things before I check out on this post.

1. This simulation is a multi-armed bandit **with independent arms**.
Problems with *similarity* between drug treatments are much more interesting. In these settings efficient exploration is *even more* important.

2. In the case of finite independent Bernoulli arms **you can actually compute the Bayes-optimal solution**, which would do better than all of these algorithms. This method of solution is usually called a [Gittins index](https://en.wikipedia.org/wiki/Gittins_index) but I'll leave that for you to dig into another time...

## Code

If you fancy having a peak under the hood then just right click anywhere on this page and click "View Page Source".
The only really interesting part of the code is how the different bandit algorithms select actions.
Hopefully this should look exactly like the descriptions in my [last post](http://iosband.github.io/2015/07/19/Efficient-experimentation-and-multi-armed-bandits.html).

### Algorithm 1: Greedy
~~~~ js
function greedy_choice(arm_counts) {
    // greedy_choice uses the greedy empirical estimate to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    for (i = 0; i < arm_counts.length; i++) {
        var n_pull = (arm_counts[i][0] + arm_counts[i][1])
        if (n_pull < 0.5) {
            n_pull = 1
        }
        var p_hat = arm_counts[i][0] / n_pull
        if (p_hat > p_max) {
            p_max = p_hat
            choice = i
        }
    }
    return choice
}
~~~~


### Algorithm 2: Epsilon-Greedy

~~~~
function egreedy_choice(arm_counts) {
    // egreedy_choice uses epsilon-greedy to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //  epsilon - double - probability of random action - now fixed within
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var epsilon = 0.1
    var choice = -1
    if (Math.random() < epsilon) {
        choice = Math.floor((Math.random() * arm_counts.length));
    } else {
        choice = greedy_choice(arm_counts)
    }
    return choice
}
~~~~


### Algorithm 3: Optimism (UCB)

~~~~
function ucb_choice(arm_counts, timestep) {
    // ucb_choice uses the UCB algoirthm to choose an arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //  timestep - int - number of timesteps elapsed
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    for (i = 0; i < arm_counts.length; i++) {
        var n_pull = (arm_counts[i][0] + arm_counts[i][1])
        if (n_pull < 0.5) {
            n_pull = 1
        }
        var p_hat = arm_counts[i][0] / n_pull
        var p_upper = p_hat + Math.sqrt(Math.log(timestep) / n_pull)
        if (p_upper > p_max) {
            p_max = p_upper
            choice = i
        }
    }
    return choice
}
~~~~

### Algorithm 4: Posterior sampling

~~~~
function ps_choice(arm_counts) {
    // ps_choice uses posterior sampling to make a choice of arm.
    //
    // Args:
    //  arm_counts - n_arms x 2 - array of observed counts
    //
    // Returns:
    //  choice - int - arm to be pulled at the next timestep (0 index)
    var p_max = -1
    var choice = -1
    var prior_a = 1
    var prior_b = 1
    for (i = 0; i < arm_counts.length; i++) {
        var p_sample = rbeta(arm_counts[i][0] + prior_a,
                             arm_counts[i][1] + prior_b)
        if (p_sample > p_max) {
            p_max = p_sample
            choice = i
        }
    }
    return choice
}
~~~~

Until next time...

<center>
<img src="http://www.etfroundup.com/wp-content/uploads/2014/11/thats_all_folks_wallpaper.jpg" alt="looney tunes" style="height:300px">
</center>














