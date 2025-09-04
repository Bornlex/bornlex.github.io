+++
title = 'Log Derivation Trick'
date = 2025-09-03T21:48:23+02:00
draft = false
+++

## Introduction

Today, let’s talk about reinforcement learning, and more specifically policy-based reinforcement learning.

Policy-based reinforcement learning is when we directly parametrize the policy, meaning we are looking for a policy such as :

$$
\pi_{\theta}(s, a) = p(a | s, \theta)
$$

In other words, we are looking for a function that represents the probability of our agent taking a specific action $a$ in a state $s$. Think about a state as the position on the chess board for instance and the action as the move to be played next.

Here, we consider a neural network and we want it to learn the best policy. In order to quantify how good is the policy, we have to have an *objective function*. Let’s call it $J$. $J$ depends on $\theta$, the parameters of our model. We are looking for the parameters that maximize this function.

Because we are looking to maximize the objective function, we are going to perform *gradient ascent* on it :

$$
\theta_{i + 1} \leftarrow \theta_i + \alpha \nabla_{\theta}J(\theta)
$$

Where $\alpha$ is the learning rate.

## Computing the gradient

When training a model for reinforcement learning, what we are doing is basically trying to maximize some kind of reward function $f(x)$ (which is a scalar function) under some probability distribution $p(x | \theta)$ which can be seem as the policy function.

Our goal is to compute

$$
\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)]
$$

By definition we can write :

$$
\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)] = \nabla_{\theta} \sum_x p(x|\theta)f(x)
$$

Because the gradient of a sum is the sum of the gradients :

$$
\nabla_{\theta} \sum_x p(x|\theta)f(x) = \sum_x \nabla_{\theta} [p(x|\theta)f(x)]
$$

$f(x)$ does not depend on $\theta$ but only on the state $x$ so we can get it out of the gradient and have :

$$
\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)] = \sum_x f(x)\nabla_{\theta} p(x|\theta)
$$

The problem we have here is that we cannot easily compute this expression. Indeed, we cannot sum over the whole state space, because we have no idea what it looks like. It might be following uniform but it might follow a very exotic manifold as well. We need to rework a bit our equation.

### The gradient trick

We are going to multiply by 1, which might seems strange at first but this will allow us to get a tractable formula :

$$
\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)] = \sum_x f(x) \nabla_{\theta} p(x|\theta) \frac{p(x|\theta)}{p(x|\theta)}
$$

Then considering the sub term :

$$
\nabla_{\theta} p(x|\theta) \frac{p(x|\theta)}{p(x|\theta)} = \frac{\nabla_{\theta} p(x|\theta)}{p(x|\theta)} p(x|\theta)
$$

And we know that the derivative of the logarithm of a function is written as follow :

$$
(\log u(x))' = \frac{u'(x)}{u(x)}
$$

It looks like the first part of the previous term, so we can inject it inside our formula :

$$
\sum_x f(x) \nabla_{\theta} p(x|\theta) \frac{p(x|\theta)}{p(x|\theta)} = \sum_x f(x) \frac{\nabla_{\theta} p(x|\theta)}{p(x|\theta)} p(x|\theta)
$$

And 

$$
\sum_x f(x) \frac{\nabla_{\theta} p(x|\theta)}{p(x|\theta)} p(x|\theta) = \sum_x f(x) \nabla_{\theta} \log p(x|\theta) p(x|\theta)
$$

Which can be written as an expectancy :

$$
\sum_x f(x) \nabla_{\theta} \log p(x|\theta) p(x|\theta) = E_{x \sim p(x | \theta)}[f(x) \nabla_{\theta} \log p(x|\theta)]
$$

And now we have a distribution $p(x | \theta)$ we can sample from. For each sample, we just have to evaluate the reward function $f(x)$ and the gradient of the log of the probability $\nabla_{\theta} \log p(x|\theta)$ to have the overall gradient.

The final formula is :

$$
\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)] = E_{x \sim p(x | \theta)}[f(x) \nabla_{\theta} \log p(x|\theta)]
$$

This gives us in what direction to shift the parameters in order to maximize the objective function (as judged by $f$).

## Conclusion

Let’s think a bit about what we did here. Basically, we had a first formula and we reworked it a bit to get another formula. But why ? This is very subtle, and the key is **computability**.

Using the first formula $\nabla_{\theta} E_{x \sim p(x | \theta)} [f(x)]$ is not enough because it would require us to evaluate the entire state space, which is usually enormous in reinforcement learning. It is practically impossible to enumerate all possible states.

> But what is the difference with the second equation ?? It still is an expectancy over a distribution !
> 

Indeed it is, but with a big difference : there is a term inside the sum that depends on $\theta$. And this makes the whole thing works, because now we can sample using any sampling method such as Monte-Carlo, and have a term that depends on the parameters, meaning that it is possible to know how much the current parameters impacted the reward.

Because it is possible to just sample a trajectory and compute how much this trajectory affected the reward without having to enumerate the whole space, the method works.

We went from

> I need to know how changing $\theta$ affects the probability of every possible state. (impossible)
> 

To

> I need to know how changing $\theta$ affects the probability of just the states I actually sampled. (doable)
>