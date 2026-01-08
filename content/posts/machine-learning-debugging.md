+++
title = 'Integrated Gradients - Debugging ML Part 2'
date = 2026-01-08T15:33:18+01:00
draft = false
+++

If you did not read the first part, it is available here https://bornlex.github.io/posts/markuplm/ ! The main content can be understood on its own, but for the practical examples, reading the first part will make things easier.

---


Now that the inner workings of the model are clearer, let’s actually talk about debugging it.

Debugging a machine learning model is not easy, but there are a few things we can do. In this article, we are going to talk about two of them :

- Understanding if the data that we are training the model on is actually fitting, this is easy
- Understanding the features of the input the model is relying on when classifying a specific example, this is slightly more complicated

## Truncation

As I said, BERT and thus MarkupLM (whether it is the base or large versions) accept at most 512 tokens, which is not that much when we are dealing with real-world web pages.

One easy way to know whether our page has been truncated or not is to look at the first dimension of the `input_ids` inside the return value of the processor call method. Indeed, if the page is larger than 512 tokens, the processor is going to batch with elements in the batch being chunks of the page :

```python
encoding = processor(
    html_string,
    return_tensors="pt",
    max_length=max_length,
    truncation=True, # We enable this to see what gets kept
    padding="max_length",
    return_overflowing_tokens=True, # Critical: tells us what was cut off
    stride=0
)
input_ids = encoding["input_ids"]

if input_ids.shape[0] > 1:
		print(f"⚠️ TRUNCATION DETECTED: This page is too long.")
else:
		print(f"✅ Safe: The entire page fits within the context window.")
```

To get how many tokens are used per element in batch, the simplest method is to count the number of 1 in the attention masks. Indeed, when a page is shorter than the context window, it is filled with padding tokens, and those tokens are masked in the attention layer.

So the very simple :

```python
num_tokens = torch.sum(encoding["attention_mask"][0]).item()
```

Will do the trick just fine !

<aside>
💡

Keep in mind that a batch of size > 1 means multiple predictions ! So each chunk of the page is going to be classified, which means that you might have to postprocess the result of the prediction to average it for example.

</aside>

## Integrated Gradients

Integrated gradients is a method that has been developed by people from Google in 2017. The idea is to attribute a score for each feature of the input vector so that the score represents how important this feature is for the prediction.

Let us define this formally.

Suppose we have a function :

$$
f : \mathbb{R}^n \to [0, 1]
$$

that represents a neural network and an input $x \in \mathbb{R}^n = (x_1, ..., x_n)$. Our target is to understand what are the most important features of the input $x$ when predicting its label.

We define an attribution of the prediction at input $x$ relative to a baseline input $x'$ being a vector :

$$
A_f(x, x') = (a_1, ..., a_n) \in \mathbb{R}^n
$$

where $a_i$ is the contribution of the feature $x_i$ to the prediction $f(x)$.

### Method

An intuitive thought would be to compute the gradient of the prediction with respect to the input vector. Indeed, the derivative is basically how much the prediction would change if we change the input vector. But the integrated gradient methods does it slightly differently, and we will explain why later in the article.

The integrated gradient method computes the attribution as follows :

$$
A(x\_i) = (x\_i - x'\_i) \int^1\_{\alpha = 0} \frac{\partial f(x' + (x - x') \alpha)}{\partial x\_i} d \alpha
$$

We are basically computing the integral of the gradient from the baseline to the actual output. It is easy to see that :

$$
\\left\\{\\begin{matrix}
f(x' + (x - x')\alpha) = f(x') & \text{if } \alpha = 0 \\\\
\\\\
f(x' + (x - x')\alpha) = f(x) & \text{if } \alpha = 1
\\end{matrix}\\right.
$$

### Approximation

In practice, we are not going to compute the integral over the whole path, we are going to approximate the integral by a summation. We simply sum the gradients at small intervals from the baseline to the output.

So instead of compute the continuous integral defined above, we approximate it using the following sum :

$$
A\_{\text{approx}}(x\_i) = (x\_i - x'\_i) \sum_{k = 1}^m \frac{\partial f(x' + \frac{k}{m}(x - x'))}{\partial x\_i} \frac{1}{m}
$$

The higher the value of $m$, the closer the approximation to the actual integral.

### Baseline

The first question that comes to mind (at least that came to mine) is what is the baseline ? The baseline is the reference vector (for example an image or a sequence of embeddings) that is used to compute the attribution against. Practically, it can be a vector full of 0s, a whole black image in the case of image processing, or a sequence of embeddings set to 0 for text.

But why choose a baseline ? Couldn’t we just compute the gradient of the prediction with respect to the input and see how changing a feature would change the prediction ?

Indeed we could, but this would sometimes create problems. The authors of the article give this interesting example. Consider a simple network that has one variable and only calls the ReLU function :

$$
f(x) = 1 - \text{ReLU}(1 - x)
$$

Now suppose the baseline is $x = 0$ and the input is $x = 2$. The function changes from 0 to 1 and then becomes flat. So the gradient would be 0 at $x = 2$ :

$$
\frac{df}{dx}(2) = -\frac{d\text{ ReLU}}{dx}(-1) = 0
$$

So the simple gradient method would attribute an importance of 0 to the only variable we have in the function ! Which is not exactly what we are looking for obviously.

### Properties

Understanding the limitation of the simple gradient method leads us to the main property that we want our attribution method to have : **sensitivity**.

Sensitivity is :

> If an input and a baseline differ in one feature but have different predictions, then this differing feature should be given a non-zero attribution.
> 

The gradient method defined just above violates this definition, justifying the need for a baseline.

Later in the paper, the authors give an additional definition of sensitivity :

> If the function implemented by the network does not depend on some variable, the attribution of that variable should be 0.
> 

Which is a natural complement to the first definition.

---

The second property that the integrated gradient method has is **implementation invariance**.

> If two models are functionally equivalent (their outputs are equal for all inputs despite having very different implementations), the attributions should be identical.
> 

---

On top of those two main axioms, the authors give two more that matter a bit less :

- completeness : the attributions add up to the differences between $f(x)$ and $f(x')$
- linearity : if the prediction is the weighted sum of two networks, the attribution should respect the linear combination

### Paths

A careful reader might note that the method used here follows a straight path from the baseline to the output, but we could be following any kind of path. So why the straight line ?

It is possible to extend the method to work with any path :

$$
A(x_i) = \int^1_{\alpha = 0} \frac{\partial f(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \times \frac{\partial\gamma_i(\alpha)}{\partial \alpha} d \alpha
$$

Let’s break it down :

- $\alpha$ is still the small move we make from the baseline towards the input
- $\gamma(\alpha)$ is the path we take to go from the baseline to the input, it is a function that gives a vector $\gamma(\alpha) = x_{\alpha}$ for every value of $\alpha$
- $\frac{\partial \gamma_i(\alpha)}{\partial \alpha}$ is how changing $\alpha$ changes the $i$-th coordinate of the path
- $\frac{\partial f(\gamma(\alpha))}{\partial \gamma_i(\alpha)}$ is how a change of the $i$-th coordinate of the path changes the prediction

The path function can be many functions that at least respect the following properties :

- $\gamma(\alpha = 0) = x'$
- $\gamma(\alpha = 1) = x$

If we follow a straight line, meaning that :

$$
\gamma(\alpha) = \alpha (x - x') + x'
$$

It is obvious that it respects the two conditions.

The terms in the integral become :

$$
\frac{\partial\gamma_i(\alpha)}{\partial \alpha} = \frac{\partial}{\partial \alpha}[\alpha(x_i - x_i') + x_i'] = x_i - x_i'
$$

Which is the term that we see outside the integral, we can get it out because it does not depend on $\alpha$ any more. The second term is only a replacement of the expression of $\gamma_i(\alpha)$ in the formula. The bottom term might seem different than the one we had in the straight line integral but it is not, it is merely a matter of notation.

$\gamma_i(\alpha)$ is the value of the $i$-th coordinate of the input evaluated at $\alpha$.

$$
\frac{\partial f(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \equiv \left. \frac{\partial f(z)}{\partial z_i} \right|_{z=\gamma(\alpha)} \equiv \frac{\partial f(x' + (x - x') \alpha)}{\partial x_i}
$$

The point of following a straight path is that it preserves symmetry, meaning that if two features contributed the same, they should have the same attribution, which is the case only for the straight line. The authors prove that it is possible to find a function that does not preserve symmetry if it is not a straight line in the annex.

### Code

Now that we understand the method, let’s talk code.

The code is not going to be very difficult to write since any framework can compute the gradient very easily.

```python
# Import the required functions and classes
from transformers import MarkupLMProcessor, MarkupLMForSequenceClassification

# Load the pretrained model and processor from Huggingface
model = MarkupLMForSequenceClassification.from_pretrained('microsoft/markuplm-base')
processor = MarkupLMProcessor.from_pretrained('microsoft/markuplm-base')

# Compute the input sequences, ready to be used as input of the model
inputs = processor(
    html_string,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding="max_length",
)

# Now because the model sums the token, positional, tag and subscript embeddings
# we are not going to compute the derivative of the prediction wrt the input
# but instead wrt the embedding, so we need to select the embedding layer
embeddings_layer = model.markuplm.embeddings

# Compute the embedding for our input
with torch.no_grad():
    input_embeddings = embeddings_layer(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"],
        xpath_tags_seq=inputs["xpath_tags_seq"],
        xpath_subs_seq=inputs["xpath_subs_seq"]
    )

# Create a full zero embedding which is our baseline
baseline_embeddings = torch.zeros_like(input_embeddings)

# Create the placeholder for the accumulated gradient
total_gradients = torch.zeros_like(input_embeddings)
```

Now we are ready to compute the gradient alongside the path between the baseline and the embedding.

We are going to use the sum formula and transform it into a nice loop :

```python
for i in range(steps + 1):
    alpha = i / steps

		# This is exactly the formula of the straight line path
    interpolated_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
    interpolated_embeddings.requires_grad = True

    # Forward pass using 'inputs_embeds' to bypass the standard embedding layer
    # This allows us to feed our interpolated vectors directly into the encoder
    outputs = model(
        inputs_embeds=interpolated_embeddings,
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )

		# Retrieve the prediction of the class we want to inspect
    score = outputs.logits[0, target_class_index]

    model.zero_grad()
    
    # Backpropagate the gradient of the prediction through the network
    score.backward()

		# Extract the gradient of prediction wrt the interpolate embeddings
		# Accumulate it
    total_gradients += interpolated_embeddings.grad

# Average it, this is the "1/m" term in the formula
avg_gradients = total_gradients / (steps + 1)
attributions = (input_embeddings - baseline_embeddings) * avg_gradients
```

Then we are going to perform two small operations that are for convenience and ease :

```python
# Sum across the embedding dimension (768) to get one score per token
attributions_sum = attributions.sum(dim=-1).squeeze(0)

# Normalize for readability
attributions_sum = attributions_sum / torch.norm(attributions_sum)
```

And we are ready to inspect it !

### Results

Let’s display the first 50 tokens that are the most important based on their attribution scores :

```python
def visualize_top_tokens(attributions, input_ids, top_k=10):
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    # Pair tokens with their scores
    token_scores = list(zip(tokens, attributions.tolist()))

    # Sort by absolute importance (magnitude)
    sorted_tokens = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)

    print(f"--- Top {top_k} Most Important Tokens ---")
    for token, score in sorted_tokens[:top_k]:
        impact = "Positive (+)" if score > 0 else "Negative (-)"
        print(f"Token: {token:<15} | Score: {score:.4f} | Impact: {impact}")
```

Which gives :

```python
Token:  is             | Score: 0.7826 | Impact: Positive (+)
Token:  a              | Score: 0.2153 | Impact: Positive (+)
Token:  article        | Score: -0.2038 | Impact: Negative (-)
Token:  article        | Score: -0.1970 | Impact: Negative (-)
Token: This            | Score: 0.1924 | Impact: Positive (+)
Token:  is             | Score: -0.1733 | Impact: Negative (-)
Token:  real           | Score: -0.1607 | Impact: Negative (-)
Token:  testing        | Score: -0.1457 | Impact: Negative (-)
Token:  here           | Score: -0.1391 | Impact: Negative (-)
Token: This            | Score: -0.1277 | Impact: Negative (-)
Token:  article        | Score: -0.1172 | Impact: Negative (-)
Token:  multiple       | Score: -0.1146 | Impact: Negative (-)
Token:  article        | Score: -0.1080 | Impact: Negative (-)
Token:  is             | Score: -0.1076 | Impact: Negative (-)
Token:  meant          | Score: -0.0882 | Impact: Negative (-)
Token: It              | Score: -0.0785 | Impact: Negative (-)
Token:  paragraphs     | Score: -0.0711 | Impact: Negative (-)
Token:  test           | Score: 0.0687 | Impact: Positive (+)
Token: </s>            | Score: -0.0669 | Impact: Negative (-)
Token: This            | Score: -0.0640 | Impact: Negative (-)
Token:  for            | Score: -0.0635 | Impact: Negative (-)
Token:  content        | Score: -0.0613 | Impact: Negative (-)
Token:  contains       | Score: -0.0596 | Impact: Negative (-)
Token:  a              | Score: -0.0578 | Impact: Negative (-)
Token:  test           | Score: 0.0543 | Impact: Positive (+)
Token:  purposes       | Score: -0.0536 | Impact: Negative (-)
Token: .               | Score: 0.0481 | Impact: Positive (+)
Token: .               | Score: 0.0369 | Impact: Positive (+)
Token:  and            | Score: -0.0352 | Impact: Negative (-)
Token:  to             | Score: -0.0252 | Impact: Negative (-)
Token:  demonstration  | Score: 0.0231 | Impact: Positive (+)
Token:  purely         | Score: -0.0203 | Impact: Negative (-)
Token: The             | Score: 0.0188 | Impact: Positive (+)
Token:  a              | Score: 0.0184 | Impact: Positive (+)
Token:  for            | Score: 0.0182 | Impact: Positive (+)
Token: .               | Score: -0.0174 | Impact: Negative (-)
Token:  is             | Score: 0.0152 | Impact: Positive (+)
Token:  fictional      | Score: -0.0057 | Impact: Negative (-)
Token: <s>             | Score: 0.0045 | Impact: Positive (+)
Token:  simulate       | Score: -0.0009 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
Token: <pad>           | Score: 0.0000 | Impact: Negative (-)
```

The text of our toy page is smaller than 50 tokens, so we see the padding tokens at the end.

So this is a bit strange because we have an attribution score for each individual text token, which is not exactly helpful in the case of HTML. Indeed, we might be more interested in understanding how a specific node affects the classification than how a words affect it.

Let’s perform some aggregation over the nodes :

```python
def aggregate_attributions_to_nodes(attributions_sum, inputs):
    input_ids = inputs["input_ids"][0]
    xpath_tags = inputs["xpath_tags_seq"][0]
    xpath_subs = inputs["xpath_subs_seq"][0]

		# A dictionary to store the node data, both score and content
    node_data = collections.defaultdict(lambda: {'score': 0.0, 'tokens': []})

		# We want to skip the special tokens
    special_ids = set(processor.tokenizer.all_special_ids)

    for i, token_id in enumerate(input_ids):
        if token_id.item() in special_ids:
            continue

				# This will act as a signature for the current node, so that all
				# text tokens that have the same xpath and subscript are going to be
				# aggregator together
        tags = xpath_tags[i].tolist()
        subs = xpath_subs[i].tolist()
        node_signature = tuple(zip(tags, subs))

				# Aggregate score
        score = attributions_sum[i].item()
        node_data[node_signature]['score'] += score

        # Decode text tokens and append them to the current node
        word = processor.tokenizer.decode([token_id])
        node_data[node_signature]['tokens'].append(word)

    return node_data
```

Let’s pass the node_data return value to a function that is going to display it and have a look !

```python
#1 [POSITIVE (+)] Score: 0.7419
   📝 Text:  "This is a test article"
   📍 XPath: /div/title/p
   ------------------------------------------------------------
#2 [NEGATIVE (-)] Score: -0.6394
   📝 Text:  "It contains multiple paragraphs to simulate a real article."
   📍 XPath: /div/body/div/p[2]
   ------------------------------------------------------------
#3 [NEGATIVE (-)] Score: -0.4964
   📝 Text:  "The content here is purely fictional and meant for demonstration."
   📍 XPath: /div/body/div/p[3]
   ------------------------------------------------------------
#4 [NEGATIVE (-)] Score: -0.4875
   📝 Text:  "This article is for testing purposes."
   📍 XPath: /div/body/div/p[1]
   ------------------------------------------------------------
#5 [POSITIVE (+)] Score: 0.0766
   📝 Text:  "This is a test article"
   📍 XPath: /div/body/div/h1
   ------------------------------------------------------------
```

Here, it seems that the 1st and 5th nodes are pushing towards the class I’ve chosen to run the attribution against. Both xpath are titles, maybe this tells us something valuable. The 3 other nodes, that are paragraphs, are pushing away from this class.

Obviously the class is internal to what I was doing, so it won’t make sense to you, but this could work exactly the same for a blog, article, e-commerce… kind of classification. And we would be able to understand what are the nodes that push towards a specific class or away from a specific class.

# Resources

- https://medium.com/@Oxagile/building-an-llm-crawler-11260af207f3
- https://huggingface.co/docs/transformers/model_doc/markuplm#transformers.MarkupLMProcessor
- https://arxiv.org/abs/1703.01365
- https://neptune.ai/blog/ml-model-interpretation-tools
- https://github.com/microsoft/unilm/tree/master/markuplm
