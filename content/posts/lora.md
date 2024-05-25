+++
title = 'LoRA: Low Rank Adaptation'
date = 2024-05-24T14:53:18+02:00
draft = false
+++

![LoRA: Low Rank Adaptation](/lora/roses.jpg)

# Introduction

Whenever we need to use a deep learning model for image or text generation, classification
or any other task, we have two possibilities:
- Train the model from scratch
- Use a pre-trained model and fine tune it for the task we need.

Training a model from scratch can be challenging and requires computational resources, time, and sometimes large quantity
of data. On the other hand, using a pre-trained model is easier but might require some adaptation to the new task.

Let's say we choose the second option, but we only have a small dataset. In this case, we can use a technique called
LoRA which stands for Low-Rank Adaptation.

This is a powerful and popular technique.

# How does it work?

## Training a model

When we train a model, we need to learn the weights of the model. These weights are learned by minimizing a loss function
which measures the difference between the predicted output and the true output.

$$
l(\hat{y}, y) = l(f_{\theta}(x), y)
$$

Where:
- $x$ is the input
- $\hat{y}$ is the predicted output
- $y$ is the true output
- $f_{\theta}(x)$ is the model parameterized by $\theta$, the weights of the model

Once we have the loss function, we will simply compute the gradient of the loss function with respect to the weights and
increase or decrease the weights accordingly.

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta} l(\hat{y}, y)
$$

Where $\alpha$ is the learning rate.

## Fine-tuning a model

Once we trained a model on a large dataset (or we managed to find a pre-trained model), we might want to adapt it to a new task for which we have a small dataset.
From what we said before about training, it seems straightforward to just repeat the previous procedure on the new dataset.

We would just initialize the weights of the model with the weights of the pre-trained model and use the previous formula
to update the weights of our model on the new dataset.

But it might not be that easy.

## Why?

Let's keep in mind the following considerations:
- A pre-trained model might contain hundreds of millions of parameters if not billions of parameters, making it very difficult
  to train on a regular machine
- If we need to do many fine-tuning on many different tasks, we end up storing many sets of parameters with relatively
  small differences between them (remember that one model can have billions of parameters)
- A neural network typically contains dense layers and performs matrix multiplication. Those matrices can be very large,
  but they might have a low rank, meaning that they can be approximated by a product of two smaller matrices.

# LoRA

During fine-tuning, LoRA add to weights matrices a low-rank matrix that is the product of two smaller matrices:
$$
W = W_0 + AB
$$

Where:
- $W_0$ is the original weight matrix of shape (n, m)
- $A$ is a matrix of shape (n, k) with $k$ the rank of the decomposition, $k < n$
- $B$ is a matrix of shape (k, m)

The point is that we can use a $k$ as small as we need ($k \ll n, k \ll m$).

## Forward pass

During the forward pass, we compute the output of the layer as follows:
$$
y = (W_0 + AB)x = W_0x + ABx
$$

## Backward pass

During the backward pass, we compute the gradient of the loss function with respect to the weights as follows:
$$
\nabla_{AB}l = \nabla_{y}l \text{   }  \nabla_{AB}y
$$

Where:
- $\nabla_{y}l$ is the gradient of the loss function with respect to the output of the layer
- $\nabla_{AB}y$ is the gradient of the output of the layer with respect to the low-rank matrix
- $\nabla_{AB}l$ is the gradient of the loss function with respect to the low-rank matrix, the value we need to update our LoRA weights

## Gains

The main advantage of LoRA is that it allows to store smaller matrices and that it does not require to store the full set of parameters.

Let us compare how smaller the matrices are with LoRA compared to the original matrices:
- The original matrix $W_0$ has a size of $n \times m$
- The low-rank matrices $A$ and $B$ have a size of $n \times k$ and $k \times m$ respectively

As an example, we choose $n = m$, $n = 200$ and $k = \frac{n}{10}$, we have:
- The original matrix $W_0$ has a size of $n \times n = 40000$
- The low-rank matrix $A$ has a size of $n \times k = 4000$
- The low-rank matrix $B$ has a size of $k \times n = 4000$
- The product $AB$ has a size of the sum of the two previous matrices, $4000 + 4000 = 8000$
- The overall gain is $40000 - 8000 = 32000$ which represents a gain of $80%$

# The code

Now that we understand the principle behind LoRA, let's see how we can implement it in PyTorch.

First let's start with a regular Attention layer:
```python
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, embed_size):
        super(SimpleAttention, self).__init__()
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        # Apply linear transformations to get queries, keys, and values
        keys = self.keys(keys)
        queries = self.queries(query)
        values = self.values(values)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-1, -2)) / (self.embed_size ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)

        out = self.fc_out(out)
        return out
```

And now let's implement LoRA:
```python
import torch
import torch.nn as nn

class LoRAAttention(nn.Module):
    def __init__(self, embed_size, r=8, alpha=16):
        super(LoRAAttention, self).__init__()
        self.embed_size = embed_size
        self.r = r
        self.alpha = alpha

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

        # LoRA-specific parameters
        self.lora_A_k = nn.Linear(embed_size, r, bias=False)
        self.lora_B_k = nn.Linear(r, embed_size, bias=False)
        self.lora_A_v = nn.Linear(embed_size, r, bias=False)
        self.lora_B_v = nn.Linear(r, embed_size, bias=False)
        
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        # Apply linear transformations to get queries, keys, and values
        keys = self.keys(keys)
        queries = self.queries(query)
        values = self.values(values)

        # LoRA adapted keys and values
        lora_k = self.lora_B_k(self.lora_A_k(keys)) * (self.alpha / self.r)
        lora_v = self.lora_B_v(self.lora_A_v(values)) * (self.alpha / self.r)

        keys += lora_k
        values += lora_v

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-1, -2)) / (self.embed_size ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)

        out = self.fc_out(out)
        return out
```

We can observe the sum of the weights with the low-rank matrix for the linear layers we talked about earlier.

Note that there is also an alpha hyperparameter that we can tune to scale the values of the low-rank matrix.

During fine-tuning, we would initialize the weights of the values, keys and queries to the weights of the pre-trained model.
Those layers would then be frozen and we would only update the weights of the low-rank matrices.

# Resources

The link to the original paper can be found [here](https://arxiv.org/abs/2106.09685).
