# Model Card: miklium-lm-mini

## Model Details
| Field | Value |
|---|---|
| **Developer** | OpenAGI for the MIKLIUM ecosystem |
| **Model Type** | Decoder-only Transformer Language Model |
| **Architecture** | Custom, built entirely in pure Python / NumPy |
| **Total Parameters** | **1,734,912 (~1.73M)** |
| **Number of Layers** | 4 |
| **Embedding Dimension** | 128 |
| **Attention Heads** | 4 |
| **Head Dimension** | 32 (128 / 4) |
| **MLP Hidden Dim** | 512 (4 × n_embd) |
| **Context Length** | 256 tokens |
| **Vocabulary Size** | 3,577 tokens |
| **Tokenization** | Word-level |
| **License** | MIT |

### Parameter Breakdown (from saved weights)

| Layer / Matrix | Shape | Parameters |
|---|---|---|
| `wte` (token embeddings) | (3577, 128) | 457,856 |
| `wpe` (position embeddings) | (256, 128) | 32,768 |
| `lm_head` | (3577, 128) | 457,856 |
| `layerN.attn_wq` × 4 | (128, 128) × 4 | 65,536 |
| `layerN.attn_wk` × 4 | (128, 128) × 4 | 65,536 |
| `layerN.attn_wv` × 4 | (128, 128) × 4 | 65,536 |
| `layerN.attn_wo` × 4 | (128, 128) × 4 | 65,536 |
| `layerN.mlp_fc1` × 4 | (512, 128) × 4 | 262,144 |
| `layerN.mlp_fc2` × 4 | (128, 512) × 4 | 262,144 |
| **Total** | | **1,734,912** |

---

## Intended Use
`miklium-lm-mini` is a lightweight model within the MIKLIUM LM family. It is intended as a foundation model designed for rapid deployment, educational purposes, and experimentation. The model demonstrates foundational reasoning and basic conversational abilities using a minimal NumPy-only stack — no PyTorch or TensorFlow required.

---

## Architecture & Infrastructure

| Component | Detail |
|---|---|
| **Framework** | Pure NumPy — no external deep learning libraries |
| **Normalization** | Root Mean Square Normalization (RMSNorm) |
| **Weight Init** | `randn(nout, nin) * sqrt(1/nin)` |
| **Activation** | ReLU (in MLP blocks) |
| **Attention** | Causal (masked) multi-head self-attention |
| **Positional Encoding** | Learned absolute position embeddings (`wpe`) |
| **Tokenizer** | Custom word-level regex tokenizer |
| **Special tokens** | `<user>`, `<ai>`, `<think>`, `</think>`, `<eos>`, `<\|endoftext\|>` (BOS) |

---

## Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam |
| **β₁** | 0.9 |
| **β₂** | 0.95 |
| **ε** | 1e-8 |
| **Learning Rate** | 0.008 with linear decay → 0 |
| **LR Schedule** | `lr * (1 - step / num_steps)` |
| **Gradient Clipping** | ±5.0 |
| **Batch Size** | 4 sequences per step |
| **Training Steps** | 5,000 |
| **Sequence Length** | 256 tokens |
| **Objective** | Autoregressive next-token prediction (cross-entropy) |
| **Random Seed** | 42 |

---

## Capabilities & Limitations

`miklium-lm-mini` sits at **1.73M parameters** — a 2× larger embedding and 2× longer context than its predecessor — providing noticeably improved coherence and context tracking. It can answer general knowledge queries and participate in short conversational exchanges.

**Limitations:**
- Small scale limits factual depth and complex reasoning.
- No fine-tuning or RLHF applied — raw autoregressive pretraining only.
- Vocabulary (3,577 tokens) and knowledge are bounded by the training corpus.
