# Narrative Coherence Analysis System

A comprehensive system built on top of the **BDH (Pathway) architecture** for analyzing narrative coherence between backstory and current story using temporal reasoning, causal analysis, thematic coherence, and Graph Neural Networks.

## Features

- **Thematic Coherence Module**: Encodes and understands entire backstory, tracking constraints and narrative elements
- **Temporal Reasoning**: Time-aware GNN for modeling temporal relationships between events
- **Causal Analysis**: Message passing networks for tracking causal dependencies
- **Sentence-Level Analysis**: Analyzes each sentence of current story for:
  - Relevance to backstory
  - Violation detection (contradictions, inconsistencies)
  - Character development tracking
  - Event consistency checking
- **Multi-Dimensional Scoring**: Temporal, causal, thematic, and character coherence scores
- **Optimization**: Handles very long text files with chunking, caching, and memory optimization

## Architecture

```
Backstory → BDH Encoder → Thematic Module → Constraint Tracker → Knowledge Graph
                                                                        ↓
Current Story → Sentence Tokenizer → BDH Encoder → Temporal-Causal GNN → Coherence Analyzer
                                                                               ↓
                                                        Violation Detection, Development Tracking
                                                                               ↓
                                                                        Analysis Report
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# If PyTorch Geometric installation fails, install manually:
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Usage

### Inference

Analyze narrative coherence between backstory and current story:

```bash
python inference.py backstory.txt current_story.txt output_report.json
```

**Example Output:**
```
======================================================================
NARRATIVE COHERENCE ANALYSIS REPORT
======================================================================

Total Sentences Analyzed: 45
Violations Detected: 3
Violation Rate: 6.67%

--- Average Coherence Scores ---
Temporal: 0.842
Causal: 0.756
Thematic: 0.891
Character: 0.823
Overall: 0.827
Relevance To Backstory: 0.734

--- Violations by Type ---

Temporal Violation: 2 occurrences
  Example: Temporal inconsistency detected (score: 0.42)

Character Violation: 1 occurrences
  Example: Character inconsistency detected (score: 0.38)
======================================================================
```

### Programmatic Usage

```python
from inference import NarrativeAnalyzer

# Initialize analyzer
analyzer = NarrativeAnalyzer(
    model_checkpoint="checkpoints/narrative_coherence_epoch_10.pt"
)

# Process backstory
analyzer.process_backstory("backstory.txt")

# Analyze current story
results = analyzer.analyze_current_story("current_story.txt")

# Generate report
report = analyzer.generate_report(results, "report.json")
analyzer.print_summary(report)
```

### Training

Train the system on your own narrative pairs:

```bash
# Create training data directory structure:
# training_data/
#   pair_1_backstory.txt
#   pair_1_current.txt
#   pair_2_backstory.txt
#   pair_2_current.txt
#   ...

python train_narrative_coherence.py
```

## Modules

### Core Components

- **`data_utils.py`**: Data loading, preprocessing, entity extraction, sentence tokenization
- **`temporal_causal_gnn.py`**: Graph Neural Networks for temporal and causal reasoning
  - `TemporalGNN`: Time-aware graph attention network
  - `CausalGNN`: Causal dependency modeling
  - `NarrativeGraph`: Dynamic graph structure for story elements
- **`narrative_coherence.py`**: Main system orchestrator
  - `ThematicCoherenceModule`: Backstory encoding and understanding
  - `ConstraintTracker`: Constraint management
  - `KnowledgeGraphBuilder`: Graph construction from backstory
- **`sentence_analyzer.py`**: Sentence-level analysis
  - `CoherenceScorer`: Multi-dimensional scoring
  - `ViolationDetector`: Contradiction detection
  - `DevelopmentTracker`: Character/plot development tracking
- **`optimization.py`**: Optimization utilities for large texts
  - Chunking strategies
  - Embedding caching
  - Mixed-precision training
  - Memory optimization

### Pipelines

- **`inference.py`**: Complete inference pipeline with report generation
- **`train_narrative_coherence.py`**: Training script with multi-task learning

## How It Works

### 1. Backstory Processing

The system first processes the backstory to extract:

- **Entities**: Characters, locations, events
- **Constraints**: Character attributes, world rules, temporal orderings
- **Relationships**: Temporal, causal, and thematic connections
- **Knowledge Graph**: Graph representation with nodes (entities, events) and edges (relationships)

### 2. Current Story Analysis

For each sentence in the current story:

1. **Encoding**: Convert to embedding using BDH model
2. **Context Retrieval**: Find most relevant backstory sentences
3. **Coherence Scoring**: Compute temporal, causal, thematic, and character scores
4. **Violation Detection**: Check for contradictions or inconsistencies
5. **Development Tracking**: Track character growth and plot evolution

### 3. Report Generation

Generates comprehensive analysis including:

- Summary statistics (total sentences, violations, rates)
- Average coherence scores across dimensions
- Detailed per-sentence analysis
- Violation explanations grouped by type

## Configuration

Modify `BDHConfig` in training/inference scripts:

```python
BDHConfig(
    n_layer=6,           # Number of BDH layers
    n_embd=256,          # Embedding dimension
    dropout=0.1,         # Dropout rate
    n_head=4,            # Number of attention heads
    mlp_internal_dim_multiplier=128,
    vocab_size=256       # Byte-level tokens
)
```

## Examples

See `examples/` directory for sample narratives:

```
examples/
  example_1_backstory.txt      # Fantasy backstory
  example_1_current.txt         # Corresponding current story
  example_1_analysis.json       # Sample analysis output
```

## Performance

- **GPU Recommended**: CUDA-enabled GPU for faster processing
- **Memory**: ~2GB for moderate-length texts (1000+ sentences)
- **Speed**: ~10-50 sentences/second depending on hardware

## Limitations

- Entity extraction uses rule-based patterns (can be improved with NER models)
- Temporal marker detection is keyword-based
- Requires training data for optimal performance
- Long documents (10k+ sentences) may require chunking

## Future Improvements

- Integration with advanced NER models (spaCy, transformers)
- Better temporal reasoning with explicit time parsing
- Support for multi-document backstories
- Interactive visualization of narrative graphs
- Fine-tuning on domain-specific narratives

## Citation

If you use this system in your research, please cite:

```
Narrative Coherence Analysis System
Built on BDH (Pathway) Architecture
Copyright 2025 Pathway Technology, Inc.
```

## License

Copyright 2025 Pathway Technology, Inc.
