+++
title = 'Chain-of-Thought is LLMs prompting themselves'
date = 2025-07-11T20:55:05+01:00
draft = false
+++


Let’s take the following notations:

- $f_{\theta}: X \to Y$ the LLM parametrized by its weights $\theta$
- $X$ the set of tasks (prompts, made of tokens)
- $Y$ the set of answers to those tasks (made of tokens as well)

The best parameters for the model are given by:

$$
\theta^* = \argmax_{\theta} f_{\theta}(y | x) \text{ with } x, y \in X, Y
$$

When fine tuning a model to think, the model is trained to answer a sequence a tokens in between the prompt and the answer that can be manually curated instead of outputing the final answer straight away. Let’s call this sequence of tokens $c \in C$. The optimal weights are now given by:

$$
\theta^* = \argmax_{\theta} f_{\theta}(y, c | x)
$$

But we know that:

$$
p(y, c | x) = p(c | x) \times p(y | x, c)
$$

Here, $f$ is actually trained to be the density probability of the next token to appear knowing the previous tokens so $f \sim p$ if we allow these aggressive notations:

- $f(c | x)$ is the probability of the model generating the intermediary sequence based on the prompt
- $f(y | x, c)$  is the probability of the model generating the right answer based on the prompt and the intermediary sequence

So if we rename $x' \leftarrow  x \circ c$ (the concatenation of both the input tokens and the intermediary tokens), then we indeed have:

$$
f(y | x')
$$

that is the probability of the model to output the right answer based on a prompt it crafted itself.