# Building a GPT-Style Large Language Model from Scratch

> A complete, ground-up implementation of a GPT-2–class large language model in **pure PyTorch** — from raw text tokenization all the way to a pretrained, fine-tuned, instruction-following model. No `transformers`, no `nn.Transformer`, no hidden abstractions. Every component is implemented and reasoned about from first principles.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Model" src="https://img.shields.io/badge/Architecture-GPT--2%20124M-blue">
  <img alt="Status" src="https://img.shields.io/badge/Status-Complete%20Pipeline-success">
</p>

---

## Why this project

Most people use LLMs through a one-line `from_pretrained()` call. This project does the opposite: it **rebuilds the entire stack** that sits underneath that call, so that every line — the attention math, the causal mask, the KQV projections, the weight-tying in the output head, the cross-entropy objective, the sampling loop — is implemented, tested, and understood.

The end result is a GPT-2 Small (124M) architecture (~163M total parameters) that:

- is trained from random initialization on raw text,
- can **load and run OpenAI's official GPT-2 pretrained weights** through a custom weight-mapping layer, and
- is **fine-tuned in two distinct paradigms**: supervised text classification and instruction following.

Every component — attention, normalization, the training loop, and the weight-mapping layer — is implemented directly rather than imported, making the internals fully inspectable.

---

## What's inside

The repository is organized as a progressive, three-stage curriculum. Each stage builds directly on the components implemented in the previous one.

### Stage 01 — Foundations: Data, Attention, and Architecture

**Data preparation & tokenization**
- A regex-based word-level tokenizer (`SimpleTokenizerV1` / `V2`) built from scratch, including a vocabulary mapping and special context tokens (`<|endoftext|>`, `<|unk|>`).
- **Byte-Pair Encoding (BPE)** via `tiktoken` (the GPT-2 tokenizer, 50,257-token vocabulary).
- Sliding-window **input–target pair** generation for next-token prediction.
- **Token embeddings** and learned **absolute positional embeddings**.

**Attention mechanism — implemented in four escalating steps**
1. Simplified self-attention (no trainable weights) to expose the core dot-product intuition.
2. Self-attention with trainable **Query / Key / Value** projection matrices and scaled dot-product.
3. **Causal (masked) attention** with an upper-triangular mask and attention dropout.
4. **Multi-head attention** — both an intuitive `MultiHeadAttentionWrapper` and an efficient single-matrix `MultiHeadAttention` implementation using tensor reshaping and batched head computation.

**The GPT architecture — built block by block**
- **Layer Normalization** implemented manually (scale/shift parameters, no `nn.LayerNorm`).
- **GELU** activation implemented from its mathematical definition.
- A position-wise **Feed-Forward network** with a 4× hidden expansion.
- **Shortcut (residual) connections** with a demonstration of how they mitigate vanishing gradients.
- A complete **Transformer block** assembling pre-LayerNorm, multi-head attention, feed-forward, dropout, and residuals.
- The full **`GPTModel`** — token + positional embeddings, a stack of 12 transformer blocks, final norm, and a weight-tied output head — verified to instantiate **163M parameters** (~124M non-embedding, GPT-2 small configuration).

### Stage 02 — Pretraining and Text Generation

- **Loss measurement done right**: manual cross-entropy, PyTorch cross-entropy, and **perplexity** as an interpretable training signal.
- A reusable **training loop** (`train_model_simple`) with gradient zeroing, backpropagation, periodic train/validation evaluation, and token-throughput tracking.
- Train/validation split and per-batch loss aggregation over a `DataLoader`.
- **Decoding strategies** implemented from scratch:
  - greedy / probabilistic (multinomial) sampling,
  - **temperature scaling**,
  - **top-k sampling**, and
  - combined temperature + top-k generation.
- **Checkpointing**: saving and loading model and optimizer state.
- **Loading OpenAI's pretrained GPT-2 (124M) weights** into the from-scratch architecture via a custom `load_weights_into_gpt` routine that maps every tensor — splitting the fused QKV matrix, transposing convolution-style weights, and aligning layer-norm scale/shift — to produce coherent generated text.

### Stage 03 — Fine-Tuning

**Classification fine-tuning (spam detection)**
- Downloads and balances the UCI SMS Spam Collection dataset, then builds padded, batched data loaders.
- Replaces the language-modeling head with a **2-class classification head** and fine-tunes the final transformer block + head.
- Evaluates with classification **accuracy** computed from the last-token logits, alongside cross-entropy loss.

**Instruction fine-tuning**
- Formats a 1,100-example instruction dataset into the **Alpaca prompt template**.
- Implements **custom collation with padding and loss masking** for variable-length instruction/response pairs.
- Loads pretrained GPT-2 weights as the base model and fine-tunes it to follow instructions.
- Generates and evaluates responses on held-out instructions.

**Fine-tuning architecture — one backbone, two adaptations**

Both paradigms reuse the *same* pretrained GPT-2 backbone and diverge only at the head and objective:

```
                  Pretrained GPT-2 backbone
            (token + positional emb → 12 transformer blocks → final LayerNorm)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   Classification fine-tuning        Instruction fine-tuning
   ─────────────────────────        ────────────────────────
   • Swap LM head → Linear(768→2)   • Keep LM head (50,257 vocab)
   • Unfreeze final block + head    • Alpaca-formatted prompts
   • Last-token logits → class      • Padded collation + loss masking
   • Objective: cross-entropy        • Objective: next-token cross-entropy
              │                               │
              ▼                               ▼
      ham / spam prediction           instruction-following text
```

---

## Architecture at a glance

```
                       ┌─────────────────────────────┐
   Raw text  ──►  BPE Tokenizer ──►  Token IDs
                       │
                       ▼
            Token Embeddings  +  Positional Embeddings
                       │
                       ▼
        ┌───────────────────────────────────────────┐
        │   Transformer Block  × 12                  │
        │   ┌─────────────────────────────────────┐ │
        │   │ LayerNorm → Multi-Head Causal Attn  │ │
        │   │           → Dropout → + Residual    │ │
        │   │ LayerNorm → FeedForward (GELU, 4×)  │ │
        │   │           → Dropout → + Residual    │ │
        │   └─────────────────────────────────────┘ │
        └───────────────────────────────────────────┘
                       │
                       ▼
              Final LayerNorm ──►  Output Head (weight-tied)
                       │
                       ▼
               Logits over 50,257 vocab  ──►  Sampling
```

**GPT-2 Small (124M) configuration**

| Hyperparameter | Value |
| --- | --- |
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Dropout | 0.1 |
| Parameters | ~163M total |

---

## Repository structure

```
llm-from-scratch/
├── Stage - 01/
│   ├── Data Preparation & Sampling/   # Tokenizers, BPE, embeddings, positional encoding
│   ├── Attention Mechanism/           # Self-, causal, and multi-head attention
│   └── LLM Architecture/              # LayerNorm, GELU, FFN, transformer block, GPTModel
├── Stage - 02/
│   ├── 01-Measuring LLM Loss Function/ # Cross-entropy, perplexity
│   ├── 02-Training LLM/                # Training loop, sampling, checkpointing
│   └── 03-Loading PreTrained GPT2 Weights.py
├── Stage - 03 (FineTuning)/
│   ├── Classification FineTuning/      # Spam classification head + training
│   └── Instruction Set FineTuning/     # Alpaca-format instruction tuning
└── the-verdict.txt                     # Sample training corpus
```

The files are numbered in learning order, so the repository reads top to bottom as a coherent narrative from tokens to a working, fine-tuned model.

---

## Tech stack

- **PyTorch** — all model components, training, and inference
- **tiktoken** — GPT-2 byte-pair encoding
- **TensorFlow** — only to read OpenAI's original GPT-2 weight checkpoints
- **NumPy / pandas** — data handling and weight manipulation

---

## Getting started

```bash
git clone https://github.com/anshgautam-github/llm-from-scratch.git
cd llm-from-scratch

pip install torch tiktoken numpy pandas tensorflow tqdm
```

Each script is self-contained and runnable. Because the files are written as a teaching progression, the recommended path is to follow them in numeric order within each stage — starting with `Stage - 01/Data Preparation & Sampling/01-Creating_Tokens_to_TokenID.py` and ending with instruction fine-tuning in Stage 03.

> Training and weight-loading scripts will automatically download the required datasets and the GPT-2 checkpoints on first run.

---

## Key concepts demonstrated

This project is a practical demonstration of the core competencies behind modern LLM engineering:

- **Transformer internals** — scaled dot-product attention, causal masking, multi-head projection and reshaping, residual streams, pre-LayerNorm.
- **Numerical foundations** — manual LayerNorm and GELU, cross-entropy and perplexity, weight tying.
- **Training engineering** — data loaders, train/val loss tracking, optimizer/checkpoint management, throughput measurement.
- **Inference & decoding** — temperature, top-k, and probabilistic sampling.
- **Transfer learning** — mapping external pretrained weights into a custom architecture, plus classification and instruction fine-tuning.

---

## Acknowledgements

The learning path and reference implementation are inspired by Sebastian Raschka's *Build a Large Language Model (From Scratch)*. The code here is re-implemented and annotated as a hands-on study of LLM internals.

---

## Author

**Ansh Gautam** · [GitHub](https://github.com/anshgautam-github)

If this project is useful or interesting to you, a ⭐ on the repository is appreciated.
