+++
title = 'GPT Series - Multi-head Self Attention'
date = 2025-09-25T14:25:38+02:00
draft = false
+++

### Motivations

Attention is now a key component for most AI systems, wether they are working with images, sequences of tokens in language processing. It has been introduced by one of the most famous papers in deep learning : [Attention is All You Need](https://arxiv.org/pdf/1706.03762).

The idea behind attention is to map two sequences (or the same sequence to itself, called cross attention) and learn how items in the sequences are related to each other. Whether it is to map two sequences of two different languages in the case of translation, or to map tokens from the same sequence to identify links between words such as :

> The cat has entered the house. It is eating now.
> 

The pronoun “it” in the sequence is related to “cat” in the first sentence. Making sure the model understands the link is key to consistent text generation.

### Attention

The simplest attention layer is basically computing the following formula :

$$
\text{attention(Q, K, V)} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Where :

- $Q = x \times q$ is the queries, the first projection of the input of the layer into a space of dimension $d_k$
- $K = x \times k$ is the keys, the second projection of the input of the layer into a space of dimension $d_k$ as well
- $V = x \times v$ is the values, the first projection of the input of the layer into a space of dimension $d_v$ which is usually the same as the embedding dimension $d$

Those 3 components are basically the results of the input tensor multiplied by 3 different linear layers.

Considering that $b$ is the batch size, $n$ is the sequence size, $d$ is the embedding dimension and $d_k$ dimension of the key, let’s have a look about the shapes of the matrices :

- $q : (b, n, d_k)$
- $k : (b, n, d_k)$
- $v : (b, n, d)$

The result of the first matrix multiplication is then normalized by the square root of the dimension. Let’s talk about it.

The $QK^T$ is often called the score. It gives a matrix of shape $(b, n, n)$.

The square root is used to normalize the elements of the matrix. Indeed, the higher the number of elements involved in the dot product, the higher the variance of the product. To get each element of the score, we have to multiply vectors of size $d_k$. If we think that each vector is sampled from a $\mathcal{N}(0, 1)$ distribution, then the variance of the product is the sum of the variances, which means the number of elements $d_k$. The standard deviation is then the square root of this $\sqrt{d_k}$, which is used to normalize the dot product.

Once we have the normalized score, we compute the softmax and it gives us some kind of probability of each token being related to all the other tokens.

The final step is to multiply this score matrix by the values, to get the output of the Attention layer.

Here is my implementation of the Attention layer :

```python
class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

        self.mask = self.register_buffer('mask', None)

    def create_causal_mask(self, inputs: torch.Tensor):
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)

        return mask

    def forward(self, keys, queries, values):
        x = torch.matmul(queries, keys.transpose(-1, -2))  # (b, n, n)

        if self.mask is None or self.mask.shape[-1] != x.shape[-1]:
            self.mask = self.create_causal_mask(x)

        x = torch.masked_fill(x, mask=self.mask, value=-torch.inf)
        x /= keys.shape[-1] ** 0.5
        x = torch.softmax(x, -1)
        x = torch.matmul(x, values)

        return x

class Attention(nn.Module):
    def __init__(self, embedding_size: int, dk: int):
        super().__init__()

        self.__d = embedding_size
        self.__dk = dk

        self.__qkv = nn.Linear(self.__d, 2 * self.__dk + self.__d)
        self.__scaled_dot_product = ScaledDotProduct()

    def forward(self, x, *args, **kwargs):
        """
        inputs (n, d)
        """
        qkv = self.__qkv(x)  # (n, 2 * dk + d)
        queries, keys, values = torch.split(qkv, [self.__dk, self.__dk, self.__d], -1)
        result = self.__scaled_dot_product(keys, queries, values)

        return result
```

### Variations around Attention

Attention in its simplest form is not the one we use the most nowadays. Many variations around Attention exist today, the most common one being Multi-head Self Attention (MHA).

Multi-head Self Attention is not very different from Attention except that instead of computing one score, we compute $h$ different scores, $h$ being the number of heads.

The MHA layer is not too difficult to implement, the struggle comes from the fact that we want our code not to use any loops for efficiency, so we have to use the Pytorch framework and our tensors in a smart way. Here is my implementation :

```python
class ScaledDotProductMHA(nn.Module):
    def __init__(self):
        super().__init__()

        self.mask = self.register_buffer('mask', None)

    def create_causal_mask(self, inputs: torch.Tensor):
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)

        return mask

    def forward(self, keys, queries, values):
        b, n, h, d2 = keys.shape
        x = torch.matmul(
            queries.reshape((b, h, n, d2)),
            keys.reshape((b, h, n, d2)).transpose(-1, -2)
        )

        if self.mask is None or self.mask.shape[-1] != x.shape[-1]:
            self.mask = self.create_causal_mask(x)

        x = torch.masked_fill(x, mask=self.mask, value=-torch.inf)
        x /= keys.shape[-1] ** 0.5
        x = torch.softmax(x, -1)
        x = torch.matmul(x, values.reshape((b, h, n, d2)))
        x = x.reshape((b, n, h, d2))

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, masked: bool = False):
        assert embedding_size % n_heads == 0

        super().__init__()

        self.__embedding_size = embedding_size
        self.__number_heads = n_heads
        self.__masked = masked

        self.__qkv = nn.Linear(
            self.__embedding_size,
            self.__embedding_size * 3
        )
        self.__scaled_dot_product = ScaledDotProductMHA()
        self.__fc = nn.Linear(self.__embedding_size, self.__embedding_size)

    def forward(self, x, *args, **kwargs):
        qkv = self.__qkv(x)
        queries, keys, values = torch.split(qkv, [self.__embedding_size] * 3, -1)

        head_size = self.__embedding_size // self.__number_heads
        queries = queries.view(x.shape[0], x.shape[1], self.__number_heads, head_size)
        keys = keys.view(x.shape[0], x.shape[1], self.__number_heads, head_size)
        values = values.view(x.shape[0], x.shape[1], self.__number_heads, head_size)

        scaled = self.__scaled_dot_product(keys, queries, values)
        scaled = scaled.view(x.shape[0], x.shape[1], self.__embedding_size)

        result = self.__fc(scaled)

        return result
```

There exist many other variations of Attention such as :

- Flash Attention (often used for efficiency)
- Multi Query Attention
- Grouped Query Attention

Those are not discussed in this article, but will be in further articles.

### Number of Parameters

For the Attention and the Multi-head Attention layers, the numbers of parameters are the following :

$$
d \times (2 * d_k + d_v)
$$
