# KDSH: Narrative Consistency Classification System

## Architecture Overview

The KDSH (Kernel Data Science Hackathon) system is a **Narrative Consistency Classifier** that determines whether character statements are **CONSISTENT** or **CONTRADICT** with their backstory. The system is built on top of **BDH (Baby Dragon Hatchling)**, a biologically-inspired language model architecture.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    KDSH NARRATIVE CONSISTENCY SYSTEM             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────┐                      ┌───────────────┐
│   TRAINING    │                      │   INFERENCE   │
│   PIPELINE    │                      │   PIPELINE    │
└───────────────┘                      └───────────────┘
        │                                       │
        │                                       │
        ▼                                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKSTORY PROCESSING (One-time)                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Text Loading                                     │   │
│  │ 2. Entity Extraction                                │   │
│  │ 3. Entity-Aware State Building                      │   │
│  │    - Global State (all narrative context)           │   │
│  │    - Entity States (per-character context)          │   │
│  │ 4. Infinite Context Encoding via BDH                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WorldState Storage
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STATEMENT CLASSIFICATION                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Statement Encoding (Isolated)                    │   │
│  │ 2. Context Encoding (with Backstory State)          │   │
│  │ 3. Counterfactual Consistency Checking              │   │
│  │    - Compute Surprise(S)                            │   │
│  │    - Compute Surprise(¬S)                           │   │
│  │    - Energy-based Decision:                         │   │
│  │      if E(S) > E(¬S) → CONTRADICT                   │   │
│  │      if E(S) < E(¬S) → CONSISTENT                   │   │
│  │ 4. CoherenceClassifier (Trained)                    │   │
│  │ 5. Final Prediction                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   PREDICTION   │
                    │  (consistent/  │
                    │  contradict)   │
                    └───────────────┘
```

## Core Components

### 1. BDH (Baby Dragon Hatchling) Model

**Location**: `bdh.py`

**Purpose**: Base language model architecture that provides:

- **Infinite Context State**: Compressed representation of entire narrative history
- **Temporal-Conditioned Attention**: Time-aware fact storage and retrieval
- **Entity-Aware Encoding**: Byte-level tokenization and embedding

**Key Classes**:

- `BDHConfig`: Model configuration (layers, embedding dim, heads, etc.)
- `BDH`: Main model class
- `TemporalLinearAttention`: Temporal-conditioned attention mechanism
- `LinearAttention`: State-space kernel for infinite context

**Key Features**:

- **State-Space Formulation**: O(1) inference with infinite history
- **Temporal Gating**: Erase/write gates for fact-aware updates
- **Byte-Level Vocabulary**: Direct UTF-8 encoding (256 tokens)

### 2. Entity-Aware World State

**Location**: `narrative_consistency.py`

**Purpose**: Maintains separate BDH states per entity for concentrated signals

**Key Classes**:

- `WorldState`: Entity-aware state container
  - `global_state`: Shared narrative context
  - `entity_states`: Per-character BDH states
  - `known_entities`: Set of detected entities

**Entity-Aware Ingestion Process**:

```
For each text chunk:
  1. ALWAYS update global_state
  2. Detect entities mentioned in chunk
  3. For each mentioned entity:
     - Compute gate value (EntityWriteGate)
     - GATED update to entity_state
     - Update entity timestamp
```

**Benefits**:

- Prevents entity-state pollution from irrelevant mentions
- Concentrates character-specific signals
- Enables query-adaptive state blending

### 3. Counterfactual Consistency Checker

**Location**: `narrative_consistency.py`

**Purpose**: Energy-based consistency checking using surprise measurement

**Key Classes**:

- `CounterfactualChecker`: Main consistency checking logic
- `ContrastiveEnergyLoss`: Training loss for energy-based model

**Algorithm**:

```
1. Generate negation of statement S → ¬S
2. Compute surprise(S, world_state) = E(S)
3. Compute surprise(¬S, world_state) = E(¬S)
4. Decision:
   - if E(S) > E(¬S): Statement resisted by world → CONTRADICT
   - if E(S) < E(¬S): Statement accepted by world → CONSISTENT
```

**Surprise Computation**:

- Measure state change after encoding statement
- Layer-weighted delta (later layers weighted more)
- Temporal decay applied
- Log-stabilized for numerical safety

### 4. Coherence Classifier

**Location**: `sentence_analyzer.py`

**Purpose**: Trained neural network for consistency classification

**Architecture**:

```
Input: [Content_Emb, Context_Emb, |Content-Context|, Content⊙Context]
  ↓
Linear(4*D → 256) + BatchNorm + ReLU + Dropout
  ↓
Linear(256 → 128) + ReLU + Dropout
  ↓
Linear(128 → 2)  [Contradict, Consistent]
```

**Training**:

- Binary cross-entropy loss
- Sparsity regularization (entropy penalty)
- Joint training with BDH feature extractor

### 5. Sentence Analyzer

**Location**: `sentence_analyzer.py`

**Purpose**: Sentence-level analysis and character substory extraction

**Key Classes**:

- `SentenceAnalyzer`: Main analyzer class
- `CoherenceScorer`: Multi-dimensional scoring wrapper
- `ViolationDetector`: Rule-based violation detection
- `DevelopmentTracker`: Character development tracking

**Features**:

- Character substory extraction from book files
- Context retrieval via attention
- Multi-dimensional coherence scoring

### 6. Data Utilities

**Location**: `data_utils.py`

**Purpose**: Text processing and narrative document building

**Key Classes**:

- `NarrativeDocument`: Structured narrative representation
- `Sentence`: Sentence with metadata (entities, temporal markers)
- `NarrativeEntity`: Entity representation
- `NarrativeDocumentBuilder`: Builder for narrative documents
- `EntityExtractor`: Rule-based entity extraction
- `SentenceTokenizer`: Sentence segmentation
- `ByteTokenizer`: Byte-level encoding/decoding

## Data Flow

### Training Pipeline

```
1. Load Training Data
   ├── train.csv: (book_name, char, content, label)
   └── Book text files

2. Precompute Backstory States (One-time)
   ├── For each book:
   │   ├── Load full text
   │   ├── Extract entities
   │   ├── Entity-aware ingestion:
   │   │   ├── Build global_state
   │   │   └── Build entity_states (per character)
   │   └── Store WorldState
   └── Map (book, char) → merged_state

3. Training Loop
   ├── For each batch:
   │   ├── Encode content (isolated) → v_iso
   │   ├── Encode content (with backstory_state) → v_ctx
   │   ├── Forward through CoherenceClassifier
   │   ├── Compute loss (CE + sparsity)
   │   └── Backpropagate & update
   └── Save best checkpoint

4. Evaluation
   ├── For each test sample:
   │   ├── Predict consistency
   │   └── Compute accuracy
   └── Generate confusion matrix
```

### Inference Pipeline

```
1. Load Model & Precomputed States
   ├── Load BDH weights
   ├── Load CoherenceClassifier weights
   └── Load WorldStates

2. For each test sample:
   ├── Get backstory_state for (book, char)
   │   └── Use AdaptiveMerge to blend global + entity states
   ├── Counterfactual Checking:
   │   ├── Negate statement
   │   ├── Compute E(S) = surprise(statement, state)
   │   ├── Compute E(¬S) = surprise(negation, state)
   │   └── Decision based on energy ratio
   ├── (Optional) CoherenceClassifier:
   │   ├── Encode isolated → v_iso
   │   ├── Encode contextual → v_ctx
   │   └── Classify (v_iso, v_ctx)
   └── Output: "consistent" or "contradict"
```

## Key Algorithms

### 1. Entity-Aware State Building

```python
def entity_aware_ingest(text, entities):
    global_state = initialize_state()
    entity_states = {e: initialize_state() for e in entities}

    for chunk in text:
        # 1. Always update global
        global_state = update_state(global_state, chunk)

        # 2. Detect entities in chunk
        mentioned = detect_entities(chunk)

        # 3. Gated update to entity states
        for entity in mentioned:
            gate = EntityWriteGate(old_state, chunk_emb, global_state)
            new_state = update_state(entity_states[entity], chunk)
            entity_states[entity] = old + gate * (new - old)

    return WorldState(global_state, entity_states)
```

### 2. Counterfactual Consistency Checking

```python
def check_consistency(statement, world_state):
    negated = negate(statement)

    # Compute surprise (energy)
    E_S = compute_surprise(statement, world_state)
    E_negS = compute_surprise(negated, world_state)

    # Energy-based decision
    if E_S > E_negS:
        return "contradict"  # Statement resisted
    else:
        return "consistent"  # Statement accepted
```

### 3. Surprise Computation

```python
def compute_surprise(text, world_state):
    state_before = world_state.clone()

    # Encode text with current state
    encode_with_state(text, state_before)
    state_after = get_state()

    # Layer-weighted delta
    delta = weighted_state_change(state_before, state_after)

    # Temporal decay
    delta *= temporal_decay(fact_time, query_time)

    # Stabilize
    surprise = log1p(delta)
    return min(surprise, SURPRISE_MAX)
```

### 4. Adaptive State Merging

```python
def adaptive_merge(statement_emb, global_state, entity_state):
    # Distributional summaries (mean + std)
    stmt_summary = [mean(stmt_emb), std(stmt_emb)]
    global_summary = [mean(global_state), std(global_state)]
    entity_summary = [mean(entity_state), std(entity_state)]

    # Learn alpha (weight for global)
    alpha = MLP([stmt_summary, global_summary, entity_summary])
    alpha = clamp(alpha, 0.05, 0.95)  # Prevent saturation

    # Blend states
    merged = alpha * global_state + (1 - alpha) * entity_state
    return merged
```

## System Integration

### Modal Deployment

The system is deployed on **Modal** for cloud execution:

```python
@app.cls(
    image=image,
    gpu="A100",
    timeout=12000,
    volumes={"/models": vol}
)
class NarrativeConsistencySystem:
    # System implementation
```

**Key Methods**:

- `train_and_evaluate()`: Full training pipeline
- `generate_submission()`: Train + generate predictions
- `predict_single()`: Single prediction
- `precompute_backstory_states()`: One-time state computation

### File Structure

```
KDSH/
├── bdh.py                      # BDH model implementation
├── narrative_consistency.py    # Main system logic
├── sentence_analyzer.py        # Sentence analysis
├── data_utils.py               # Data processing utilities
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── ARCHITECTURE.md            # This document
├── files/
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   └── *.txt                  # Book text files
└── examples/
    ├── example_1_backstory.txt
    └── example_1_current.txt
```

## Performance Optimizations

### 1. Infinite Context State

- **O(1) inference** with infinite history via state-space formulation
- Compressed representation: `[L, H, D, D]` tensor for entire book

### 2. Entity-Aware Routing

- **Gated updates** prevent state pollution
- Concentrated signals per character
- Reduces noise in entity representations

### 3. Precomputed States

- **One-time computation** of backstory states
- Cached for fast inference
- Frozen during training (no gradient flow)

### 4. Temporal Decay

- Facts decay with time distance
- Recent facts weighted more heavily
- Prevents stale information from dominating

### 5. Mixed Precision Training

- BF16/FP16 for memory efficiency
- Gradient scaling for stability

## Training Details

### Hyperparameters

- **BDH Config**:

  - Layers: 6
  - Embedding Dim: 256
  - Heads: 4
  - Dropout: 0.1

- **Training**:

  - Learning Rate: 2e-4
  - Batch Size: 16
  - Epochs: 15-20 (early stopping)
  - Optimizer: Adam
  - Weight Decay: 0.1

- **Loss Function**:
  - Cross-Entropy Loss (classification)
  - Sparsity Loss (entropy penalty, λ=1e-2)

### Training Strategy

1. **Precompute States**: One-time entity-aware state building
2. **Joint Training**: BDH feature extractor + Classifier
3. **Early Stopping**: Patience=2 epochs
4. **Best Checkpoint**: Save model with best test accuracy

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: TP, TN, FP, FN
- **Precision**: P = TP / (TP + FP)
- **Recall**: R = TP / (TP + FN)

## Key Innovations

1. **Entity-Aware World State**: Separate states per character
2. **Counterfactual Checking**: Energy-based consistency verification
3. **Infinite Context**: O(1) inference with compressed history
4. **Adaptive Merging**: Query-adaptive state blending
5. **Temporal Decay**: Time-aware fact weighting
6. **Gated Updates**: Selective state updates via EntityWriteGate

## Limitations & Future Work

### Current Limitations

- Rule-based entity extraction (could use NER)
- Rule-based negation (could use LLM)
- Simple temporal markers (keyword-based)
- Fixed book paths (hardcoded)

### Future Improvements

- Advanced NER models (spaCy, transformers)
- LLM-based negation generation
- Explicit time parsing
- Multi-document support
- Interactive visualization
- Domain-specific fine-tuning

## References

- **BDH Paper**: [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://doi.org/10.48550/arXiv.2509.26507)
- **Pathway Technology**: https://pathway.com
- **Modal**: https://modal.com

## License

Copyright 2025 Pathway Technology, Inc.
