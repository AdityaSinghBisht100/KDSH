# Architecture & Implementation Comparison: Dragon Hatchling vs. Current System

This chart breaks down the system components, distinguishing between what strictly follows the *Dragon Hatchling* (Byte-Level Language Model) methodology and what are custom engineering adaptations tailored for the Fact-Consistency Checking task.

| Feature / Component | 🐉 **Dragon Hatchling Paper** (Core Research) | 🛠️ **Our Implementation** (Custom Adaptations) |
| :--- | :--- | :--- |
| **Tokenization** | **Byte-Level** (UTF-8 bytes). Direct input of raw bytes (0-255). No BPE/WordPiece. | **Identical**. We implemented a custom `ByteTokenizer` mapping 256 byte values directly to embeddings. |
| **Architecture** | **Transformer Decoder** (GPT-style). Uses standard self-attention blocks. | **Identical**. `BDH_GPU` is a standard simplified GPT architecture (Embedding → PosEnc → N Layers → Output). |
| **Vocabulary Size** | **256** (plus special tokens). Small, fixed vocabulary. | **Identical**. `vocab_size=256`. |
| **Context Window** | Short-to-Medium (Paper tests varying lengths). | **Extended Context**. We chunk large novels (up to 8,192 tokens) to capture broader backstory context. |
| **Pretraining Objective** | **Next-Byte Prediction** (Autoregressive). Minimizing Negative Log-Likelihood (NLL). | **Identical**. Phase 1 (`pretrain_on_novels`) trains purely on next-byte prediction on the raw novel text. |
| **Latent Representation** | Implicit **Hidden State** ($h_t$). The model compresses history into this vector. | **Explicit Extraction**. We extract the hidden state using `model.get_state_representation` to use as a feature vector. |
| **Consistency / Fact Checking** | *Not explicitly defined*. The paper focuses on generation and compression capabilities. | **Perplexity Difference Method**. We devised the logic: If `PPL(Statement | Context) < PPL(Statement)`, the statement is Consistent. |
| **Semantic Understanding** | Implicitly learned through byte patterns. | **Hybrid Integration (SBERT/E5)**. We explicitly inject high-level semantic signals using a frozen E5-Base model (`SemanticEncoder`). |
| **Loss Function (Fine-Tuning)** | N/A (Paper is unsupervised/self-supervised). | **Dual-Loss Interaction**:<br>1. **Margin Ranking Loss** (on Perplexity Diff).<br>2. **Cosine Embedding Loss** (on Learned Projection). |
| **Dimensionality** | Fixed hidden size (e.g., $d_{model}$). | **Learned Projection**. We map the external SBERT embedding (768-dim) into a higher-dimensional space (2048-dim) to align with BDH complexity. |
| **Inference Strategy** | Generative (Sampling next token). | **Discriminative / Contrastive**. We don't generate text; we score the *likelihood* of the provided text (Backstory vs. Statement). |

## Summary of Innovations

1.  **Direct Gradient Flow from Perplexity**: Instead of just using the model as a feature extractor for a classifier, we use the *uncertainty* (perplexity) of the model itself as a differentiable signal.
2.  **Semantic Guardrails**: The paper relies purely on the model's internal world. We added an external "Semantic Judge" (E5-Base) to catch obvious contradictions that might be statistically plausible at the byte level but semantically opposite.
3.  **Learned Projection Layer**: Bridging the gap between a frozen, pretrained sentence encoder and a custom byte-level generative model by training a specific linear layer to align their vector spaces.
