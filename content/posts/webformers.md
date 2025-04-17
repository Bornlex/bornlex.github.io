+++
title = 'Webformers'
date = 2025-04-16
draft = false
+++

# Webformers: Structured Information Extraction from Web Documents

## Introduction

Extracting structured information from web pages remains a challenging task in natural language processing. The latest innovation in this field comes in the form of Webformers, a novel approach that uses sequence modeling to achieve state-of-the-art performance in web extraction tasks. This article explores how Webformers works, its architecture, and its implications for the future of structured information extraction.

## Understanding Webformers

Webformers stands apart from traditional approaches by encoding HTML, target fields, and text sequences within a unified transformer model. The key innovation lies in how it designs specialized HTML tokens for each DOM node, embedding representations from neighboring tokens through graph attention mechanisms, while constructing rich attention patterns between HTML and text tokens.

![Webformers Architecture](/webformers/architecture.png)

## Key Contributions

The Webformers approach makes two significant contributions to the field:

1. **Structure-aware information extraction** that integrates web HTML layout through graph attention mechanisms
2. **Rich attention mechanisms** for embedding representations among different token types, enabling efficient encoding of long sequences and empowering zero-shot extraction capabilities

## Problem Definition

To understand how Webformers functions, we need to define the problem it aims to solve. The web document is processed into:

- **Text sequence**: T = (t₁, t₂, ..., tₖ) where each tᵢ represents the i-th text node containing nᵢ words/tokens
- **DOM tree**: G = (V, E) where V is the set of DOM nodes and E is the set of edges

The objective is to extract the corresponding text information for a set of target fields F = (f₁, ..., fₘ) from the web document. Mathematically, this means finding the best text span s̄ⱼ for each field fⱼ:

$$\bar{s_j} = \argmax_{b_j, e_j} Pr(w_{b_j}, w_{e_j} | f_j, T, G)$$

Where bⱼ and eⱼ are the beginning and ending offsets of the extracted text span.

## Architecture

Webformers consists of three main components:

### 1. Input Layer

The input layer introduces three new token types:

- **Field tokens**: Represent the text fields to be extracted (title, price, etc.)
- **HTML tokens**: Each node in the DOM tree corresponds to an HTML token
- **Text tokens**: Standard word representations used in natural language models

Each token is converted into a d-dimensional embedding vector:
- Field and text embeddings combine word embedding and segment embedding
- HTML token embeddings concatenate tag embedding and segment embedding

The segment embedding indicates which type the token belongs to (field, HTML, or text), while tag embedding represents different HTML tags of the DOM nodes.

### 2. Webformers Encoder

The encoder consists of L identical contextual layers plus one feed-forward layer. It defines several attention patterns:

- **HTML-to-HTML (H2H)**: Computes attention weights between HTML tokens using graph attention
- **HTML-to-Text (H2T)**: Connects HTML tokens to their contained text tokens
- **Text-to-HTML (T2H)**: Connects text tokens to all HTML tokens
- **Text-to-Text (T2T)**: Standard attention between neighboring text tokens
- **Field token attention**: Connects field tokens to all HTML tokens

These attention patterns are combined to generate the final token representations.

### 3. Output Layer

The output layer extracts the final text span for each field from the text tokens using a softmax function on the output embeddings to generate probabilities for the beginning and ending indices.

## Limitations

Despite its advancements, Webformers has two main limitations:

1. **Single object focus**: The model is designed for pages where each target field has a single text value. For multiple objects, the authors recommend using repeated patterns methods.

2. **No rendered page consideration**: The model does not take into account the visually rendered page (OCR), focusing solely on the HTML structure.

## Results and Implications

![Benchmark](/webformers/results.png)

Webformers achieves state-of-the-art performance in structured information extraction from web documents. Its ability to leverage the HTML structure through graph attention mechanisms and its rich attention patterns between different token types enable more accurate and efficient extraction compared to previous approaches.

The model's architecture allows for zero-shot extraction capabilities, making it particularly valuable for applications requiring information extraction from previously unseen web page structures.

## Conclusion

Webformers represents a significant advancement in structured information extraction from web documents. By integrating HTML structure through graph attention and implementing rich attention patterns, it achieves superior performance in web extraction tasks. While limitations exist, the approach opens new possibilities for more accurate and efficient information extraction from the web.

As web content continues to grow exponentially, technologies like Webformers will become increasingly important for automated information extraction and knowledge building from web sources.
