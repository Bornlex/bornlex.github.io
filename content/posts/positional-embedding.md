+++
title = 'GPT Series - Positional Embedding'
date = 2025-09-16T13:33:36+02:00
draft = false
+++


## Positional embedding

### Motivation

As we saw earlier, multi-head self attention layer assigns the same output for every identical token, regardless of their position. This can cause obvious problems in sentences where the same word is used multiple times to represent different entities such as :

> The red car turned left where the yellow card turned left.
> 

The two occurrences of the “car” word represent different actual cars. They cannot be treated the same way.

To make sure that the model treats both words depending on their position, we need to somehow pass it a sense of locality. And this is where positional embedding comes into play.

### Properties

As Huggingface described in their article linked below (in the resources section), there are a few properties that we would like our embedding to have :

1. **Unique encoding for each position** : the value of the encoding should not depend on the length of the sequence, for example for both the following sentences, the word “car” should have the same positional encoding.
    1. The red car turned left
    2. The yellow car parked near the building
2. **Simple relationship between encoded positions** : to help the model learn patterns, we want to make sure the relationships between positions are simple, meaning if we know the encoding for position $p$, the encoding for position $p + k$ should be straightforward to compute.
3. **Works with longer sequences than those in the training dataset** : obviously, we want the model to learn an embedding that works for real world scenarios, so the function should be adaptable enough to work with unexpected input lengths.
4. **Extensible to multiple dimensions** : a sentence in natural language is a 1 dimensional sequence, but we might want our positional embedding to work with more dimensions to make it useful for multimodal models.

### Naive mechanism

Our positional embedding should add to each vector in the input sequence information about its position.

So for an input sequence $x \in \mathbb{R}^{n \times d}$ we want to have an embedding sequence of the same shape $e \in \mathbb{R}^{n \times d}$, and add the two together.

The most straightforward positional embedding we could have is simply a sequence of vectors containing $i$, $i$ being the position in the sequence.

Considering the $i$th vector in the sequence, we would perform the following operation :

$$
x[i, j] \leftarrow x[i, j] + i
$$

If we actually performed this operation, we would end up losing information about the actual embedding. The output value of the regular embedding layer is a tensor whose values are not too far from zero.

Let’s see with a concrete example where we simply create a random tensor respecting the vocabulary size and embedding dimension of GPT2, then use an embedding layer on this tensor :

```python
import torch
from torch import nn

random_tensor = torch.randint(low=0, high=50257, size=(10,))
embedding_layer = nn.Embedding(50257, 768)
output_tensor = embedding_layer(random_tensor)

print(output_tensor)
```

Would give something like :

```python
>>> out
tensor([[ 1.0138,  0.3207,  0.6852,  ...,  0.0805,  0.1603,  0.6530],
        [ 0.7767,  1.2861, -1.4658,  ..., -0.1355,  0.0973,  1.5500],
        [-0.8326, -1.2292, -1.4575,  ...,  0.6372, -0.5633, -1.3124],
        ...,
        [ 0.3862,  1.9091, -1.8795,  ..., -0.4065, -1.2252,  0.1061],
        [ 0.9495,  0.7224,  1.0682,  ...,  1.1046,  0.4380, -2.0438],
        [-1.5241, -0.8368, -1.1846,  ..., -0.5213,  0.2089,  0.3634]],
       grad_fn=<EmbeddingBackward0>)
```

As we can see, the values revolve around 0. Keep in mind that this layer has not been trained of course, but still, the distribution of values won’t be too far from what we see here.

So for small position in the sequence, our naive way of computing the positional embedding might work, but sooner or later, we will have the problem where the position is much higher than the values inside the tensor :

$$
i >> x[i, j]
$$

The text embedding will be noise around the position, which is not what we want.

### Bit smarter mechanism

Enjoy the word game here, we will actually be using bits…

Instead of directly adding the position to the embedding, we will first represent the position as an array of bits and then add this array to the embedding vector :

$$
x[i, j] \leftarrow x[i, j] + \text{bin}(i)[j]
$$

Which will look like this :

![Binary addition](/positional-embedding/bin.png)

Now, the position is indeed included inside the final embedding and it does not erase the information contained in the text embedding computed previously.

The issue with this still is that when going from position $i$ to position $i + 1$, a lot of the bits can jump from 0 to 1 or from 1 to 0, which is not ideal for an optimization process like training a model.

### Sinusoidal positional encoding

This is why we arrived at the sinusoidal positional embedding. The idea is to keep what we had with the bits, but smooth it using trigonometry. It is the embedding used in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762)!

Here is the formula the paper uses for even indices (inside an embedding vector) :

$$
\text{PE}(p)_{even} = \sin(\frac{p}{10000^{2i/d}})
$$

And the one for odd indices :

$$
\text{PE}(p)_{odd} = \cos(\frac{p}{10000^{2i/d}})
$$

First, $p$ is the position of the token in the sequence and $d$ is the dimension of the embedding. Second, the “10000” seems to have been chosen empirically.

The choice for cosine and sine for even and odd indices are motivated by the fact that the shape of the result looks like the binary representation we had before. Taken from the [HuggingFace article](https://huggingface.co/blog/designing-positional-encoding) :

![Binary waves](/positional-embedding/sin1.png)

For the binary representation.

![Sinusoidal waves](/positional-embedding/sin2.png)

For the sinusoidal representation.

Of course the values are not the same, but the alternation is similar which is something our model can learn.