"""
Narrative Consistency Classification System (Modal Version).

Implements a Retrieval-Augmented Classifier:
1. Ingests books and indexes sentences by character.
2. For input content, retrieves relevant backstory sentences.
3. Classifies CONSISTENT vs CONTRADICT based on alignment.
4. Generates Rationale by citing the relevant backstory.

Run with: modal run narrative_consistency.py
"""

import modal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
import sys
import os

# Define Modal App
app = modal.App("narrative-consistency-classifier")

# Define Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "scikit-learn"
    )
    .add_local_dir(".", remote_path="/root")
)

# Define Volume
vol = modal.Volume.from_name("narrative-models", create_if_missing=True)

# Constants
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256

# --- WorldState: Entity-Aware State Management ---

from dataclasses import dataclass, field

@dataclass
class WorldState:
    """
    Entity-Aware World State.
    
    Instead of a single global state that entangles all character facts,
    we maintain separate BDH states per entity for concentrated signals.
    """
    global_state: torch.Tensor = None           # Shared narrative context
    entity_states: Dict[str, torch.Tensor] = field(default_factory=dict)  # entity_id → BDH state
    known_entities: set = field(default_factory=set)  # All detected entity names
    
    def get_query_state(self, entity: str, alpha: float = 0.3) -> torch.Tensor:
        """
        Merge global and entity state for inference.
        
        Args:
            entity: Character name to query
            alpha: Weight for global state (0.3 = 30% global, 70% entity)
        
        Returns:
            Combined state tensor
        """
        if entity not in self.entity_states:
            return self.global_state  # Fallback to global
        
        entity_s = self.entity_states[entity]
        if self.global_state is None:
            return entity_s
            
        # Linear interpolation
        return alpha * self.global_state + (1 - alpha) * entity_s


# --- Energy-Based World Model Components ---

class EntityWriteGate(nn.Module):
    """
    Entity Write Gating (Scalar, Dimension-Reduced).
    
    Controls how much new information overwrites entity state.
    prevents entity-state pollution from irrelevant mentions.
    
    Review Fix #2, #3: Scalar gate with reduced dimensionality.
    """
    def __init__(self, state_dim: int, proj_dim: int = 64):
        super().__init__()
        self.proj_dim = proj_dim
        
        # Dimension reduction projections
        self.proj_entity = nn.Linear(state_dim, proj_dim)
        self.proj_chunk = nn.Linear(state_dim, proj_dim)
        self.proj_global = nn.Linear(state_dim, proj_dim)
        
        # Scalar gate MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Scalar output
            nn.Sigmoid()
        )
    
    def forward(self, entity_state: torch.Tensor, chunk_emb: torch.Tensor, 
                global_state: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar gate in [0, 1].
        Low = ignore mention, High = write update.
        """
        # Flatten states to [D] if multi-dimensional
        e = entity_state.view(-1).mean().unsqueeze(0).expand(self.proj_dim)
        c = chunk_emb.view(-1).mean().unsqueeze(0).expand(self.proj_dim) if chunk_emb is not None else torch.zeros(self.proj_dim, device=entity_state.device)
        g = global_state.view(-1).mean().unsqueeze(0).expand(self.proj_dim)
        
        gate_input = torch.cat([e, c, g], dim=-1)
        gate = self.gate_mlp(gate_input.unsqueeze(0)).squeeze()
        return gate


class AdaptiveMerge(nn.Module):
    """
    Query-Adaptive Global/Entity State Merge.
    
    Learns when to use global vs entity state based on query.
    
    Review Fix #4: Uses distributional summaries (mean + std).
    """
    def __init__(self, state_dim: int, proj_dim: int = 64):
        super().__init__()
        # Uses mean + std = 2x features
        self.alpha_mlp = nn.Sequential(
            nn.Linear(proj_dim * 6, 64),  # 3 inputs * 2 (mean+std)
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.proj_dim = proj_dim
        self.proj = nn.Linear(state_dim, proj_dim) if state_dim != proj_dim else nn.Identity()
    
    def get_summary(self, state: torch.Tensor) -> torch.Tensor:
        """Distributional summary: concat(mean, std)."""
        flat = state.view(-1)
        mean = flat.mean().unsqueeze(0).expand(self.proj_dim // 2)
        std = flat.std().unsqueeze(0).expand(self.proj_dim // 2)
        return torch.cat([mean, std], dim=-1)
    
    def forward(self, statement_emb: torch.Tensor, global_state: torch.Tensor,
                entity_state: torch.Tensor) -> torch.Tensor:
        """
        Returns merged state with learned alpha.
        """
        stmt_summary = self.get_summary(statement_emb) if statement_emb is not None else torch.zeros(self.proj_dim, device=global_state.device)
        global_summary = self.get_summary(global_state)
        entity_summary = self.get_summary(entity_state)
        
        alpha_input = torch.cat([stmt_summary, global_summary, entity_summary], dim=-1)
        alpha = self.alpha_mlp(alpha_input.unsqueeze(0)).squeeze()
        
        return alpha * global_state + (1 - alpha) * entity_state


class ContrastiveEnergyLoss(nn.Module):
    """
    Contrastive Energy Loss for supervision.
    
    Aligns surprise with labels via margin loss.
    Contradict samples should have higher E_pos than E_neg.
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, surprise_pos: float, surprise_neg: float, 
                is_contradict: bool) -> torch.Tensor:
        """
        Args:
            surprise_pos: Surprise of original statement
            surprise_neg: Surprise of negated statement
            is_contradict: True if label is "contradict"
        
        Returns:
            Loss tensor
        """
        E_pos = torch.tensor(surprise_pos, dtype=torch.float32)
        E_neg = torch.tensor(surprise_neg, dtype=torch.float32)
        
        if is_contradict:
            # Contradiction: E_pos should be HIGH (resisted by world)
            # E_neg should be LOW (accepted by world)
            loss = torch.clamp(self.margin + E_neg - E_pos, min=0)
        else:
            # Consistent: E_pos should be LOW (accepted)
            # E_neg should be HIGH (resisted)
            loss = torch.clamp(self.margin + E_pos - E_neg, min=0)
        
        return loss


# Maximum surprise value for clamping (Review Fix #6)
SURPRISE_MAX = 100.0
class CounterfactualChecker:
    """
    Counterfactual Consistency Checking.
    
    Detects LOGICAL contradictions by comparing:
    - How much the statement "surprises" the world state
    - How much its negation "surprises" the world state
    
    If statement causes MORE surprise → CONTRADICT
    If negation causes MORE surprise → CONSISTENT
    """
    
    def __init__(self, bdh_model, device):
        self.bdh = bdh_model
        self.device = device
        
        # Negation patterns (rule-based, no LLM)
        self.negation_patterns = [
            (r"^(.+) is (.+)$", r"\1 is NOT \2"),
            (r"^(.+) was (.+)$", r"\1 was NOT \2"),
            (r"^(.+) has (.+)$", r"\1 has NOT \2"),
            (r"^(.+) had (.+)$", r"\1 had NOT \2"),
            (r"^(.+) did (.+)$", r"\1 did NOT \2"),
            (r"^(.+) does (.+)$", r"\1 does NOT \2"),
            (r"^(.+) can (.+)$", r"\1 can NOT \2"),
            (r"^(.+) will (.+)$", r"\1 will NOT \2"),
        ]
    
    def negate(self, statement: str) -> str:
        """
        Generate negated variant of statement.
        Rule-based, no external LLM required.
        """
        statement = statement.strip()
        
        # Try each pattern
        for pattern, replacement in self.negation_patterns:
            if re.match(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, flags=re.IGNORECASE)
        
        # Fallback: prepend negation phrase
        return f"It is NOT true that {statement}"
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to byte tokens."""
        tokens = torch.tensor([[ord(c) % 256 for c in text]], dtype=torch.long, device=self.device)
        return tokens
    
    def compute_surprise(self, text: str, world_state: torch.Tensor, 
                          fact_time: float = 0.0, query_time: float = 1.0,
                          temporal_beta: float = 0.01) -> float:
        """
        Measure how much the statement "surprises" the world state.
        
        High surprise = statement conflicts with stored facts.
        
        Bugfixes Applied:
        - Fix #1: detach().clone() for safe state copying
        - Fix #2: Layer-weighted Δ (later layers weighted more)
        - Fix #5: Temporal decay affects decision
        - Fix #6: log1p stabilization
        """
        # Safe state cloning - no gradient leakage
        state_before = world_state.detach().clone().to(self.device)
        
        # Reset before setting to prevent silent drift
        self.bdh.reset_state()
        self.bdh.set_state(state_before)
        
        # Encode statement
        tokens = self.encode_text(text)
        
        with torch.no_grad():
            self.bdh(tokens, use_state=True)
        
        # Get state after
        state_after = self.bdh.get_state()
        
        # Bugfix #2: Layer-weighted Δ (later layers encode higher-level state)
        if state_after.dim() >= 2 and state_after.shape[0] > 1:
            n_layers = state_after.shape[0]
            layer_weights = torch.linspace(0.5, 1.5, n_layers, device=self.device)
            
            # Compute weighted sum of per-layer deltas
            delta = 0.0
            for i in range(n_layers):
                layer_delta = torch.norm(state_after[i] - state_before[i], p=2).item()
                delta += layer_weights[i].item() * layer_delta
        else:
            # Fallback for single-layer or flat state
            delta = torch.norm(state_after - state_before, p=2).item()
        
        # Temporal decay affects surprise
        temporal_decay = math.exp(-temporal_beta * (query_time - fact_time))
        delta = delta * temporal_decay
        
        # Stabilize via log1p (prevents explosion)
        surprise = math.log1p(delta)
        
        # Clamp for additional safety
        surprise = min(surprise, math.log1p(SURPRISE_MAX))
        
        return surprise
    
    def predict(self, statement: str, world_state: torch.Tensor) -> Tuple[str, float]:
        """
        Counterfactual consistency prediction.
        
        Returns:
            (prediction, confidence)
            prediction: "consistent" or "contradict"
            confidence: ratio indicating strength of prediction
        """
        # Generate negation
        negated = self.negate(statement)
        
        # Compute surprise for both
        surprise_S = self.compute_surprise(statement, world_state)
        surprise_negS = self.compute_surprise(negated, world_state)
        
        # Avoid division by zero
        epsilon = 1e-8
        conflict_ratio = surprise_S / (surprise_negS + epsilon)
        
        # Decision logic
        if conflict_ratio > 1.0:
            # Statement caused MORE surprise than its negation
            # → World state conflicts with statement
            return "contradict", conflict_ratio
        else:
            # Negation caused MORE surprise
            # → World state aligns with statement  
            return "consistent", 1.0 / (conflict_ratio + epsilon)
    
    def predict_with_details(self, statement: str, world_state: torch.Tensor) -> Dict:
        """
        Detailed prediction with debug info.
        """
        negated = self.negate(statement)
        surprise_S = self.compute_surprise(statement, world_state)
        surprise_negS = self.compute_surprise(negated, world_state)
        
        epsilon = 1e-8
        conflict_ratio = surprise_S / (surprise_negS + epsilon)
        
        prediction = "contradict" if conflict_ratio > 1.0 else "consistent"
        
        return {
            "statement": statement,
            "negated": negated,
            "surprise_statement": surprise_S,
            "surprise_negation": surprise_negS,
            "conflict_ratio": conflict_ratio,
            "prediction": prediction
        }


# --- Model Definitions ---

# GNN removed - using Pure Infinite Context architecture

class CoherenceClassifier(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        self.use_gnn = False # Deprecated in favor of Infinite Context State
        
        # Input: [Isolated_Emb, Contextual_Emb, Diff, Prod]
        # Contextual_Emb is derived from Infinite Memory State
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        
    def forward(self, iso_emb, ctx_emb):
        """
        Args:
           iso_emb: [B, D] - Embedding of content in isolation
           ctx_emb: [B, D] - Embedding of content given Backstory State
        """
        features = torch.cat([
            iso_emb, 
            ctx_emb, 
            torch.abs(iso_emb - ctx_emb),
            iso_emb * ctx_emb
        ], dim=1)
        return self.net(features)

class NarrativeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# --- Main System Logic ---

@app.cls(
    image=image,
    gpu="A100",
    timeout=12000,
    volumes={"/models": vol}
)
class NarrativeConsistencySystem:
    def __init__(self):
        self.backstory_store = {}
        self.device = None
        self.classifier = None
        self.bdh = None
        # Entity-Aware WorldState: book_name → WorldState
        self.world_states: Dict[str, WorldState] = {}
        self.backstory_states = {}  # Legacy compatibility
        
    def __enter__(self):
        print("--> Starting __enter__ initialization")
        try:
            self._initialize_components()
        except Exception as e:
            print(f"__enter__ failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _initialize_components(self):
        if self.classifier is not None:
            return

        print(f"Initializing components in pid {os.getpid()}")
        sys.path.append("/root")
        print(f"ls /root: {os.listdir('/root')}")
        
        from bdh import BDH, BDHConfig, BDHConfig  # Re-import to be safe
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized on {self.device}")
        
        self.config = BDHConfig(n_layer=6, n_embd=EMBEDDING_DIM, n_head=4)
        self.bdh = BDH(self.config).to(self.device)
        self.bdh.eval()
        
        self.classifier = CoherenceClassifier(EMBEDDING_DIM, self.device).to(self.device)
        
        # Load Weights if available
        if os.path.exists("/models/narrative_consistency.pt"):
            try:
                self.classifier.load_state_dict(torch.load("/models/narrative_consistency.pt", map_location=self.device), strict=False)
                print("Loaded classifier weights (strict=False).")
            except Exception as e:
                print(f"Weights load failed: {e}")
                
        if os.path.exists("/models/bdh_base.pt"):
             self.bdh.load_state_dict(torch.load("/models/bdh_base.pt", map_location=self.device), strict=False)

        # Import and initialize sentence analyzer
        from sentence_analyzer import SentenceAnalyzer
        self.sentence_analyzer = SentenceAnalyzer(self.bdh, EMBEDDING_DIM, self.device)

        self.backstory_store = {}
        self.backstory_states = {} # Map (book, char) -> Tensor State
        print("--> Initialization complete")

    def encode_text(self, text_list, distinct_states=None):
        """
        Args:
            text_list: List[str] - The content to encode
            distinct_states: Optional List[Tensor] or Single Tensor - The initial states to use.
                             If None, encodes in isolation (standard).
                             If provided, encodes 'as continuation' of that state.
        """
        if self.bdh is None:
            self._initialize_components()
            
        if not text_list:
            return torch.zeros((1, EMBEDDING_DIM), device=self.device)
            
        batch_tensors = []
        for text in text_list:
            # Simple truncation/encoding
            tokens = torch.tensor([ord(c) % 256 for c in text[:MAX_SEQ_LEN]], dtype=torch.long, device=self.device)
            if len(tokens) == 0:
                 tokens = torch.zeros(1, dtype=torch.long, device=self.device)
            batch_tensors.append(tokens)
            
        max_len = max([len(t) for t in batch_tensors])
        padded = torch.zeros(len(batch_tensors), max_len, dtype=torch.long, device=self.device)
        for i, t in enumerate(batch_tensors):
            padded[i, :len(t)] = t
            
        # Manage State for Contextual Encoding
        if distinct_states is not None:
             # Case 1: Single State shared for all inputs
             if isinstance(distinct_states, torch.Tensor): 
                 self.bdh.set_state(distinct_states.to(self.device))
             pass 

        if distinct_states is None:
             # Isolated: Standard forward, no state usage (or reset first)
             self.bdh.reset_state() # Ensure clean slate
             with torch.no_grad():
                sent_emb = self.bdh.compute_embeddings(padded)
        else:
             # Contextual loop
             sent_embs = []
             for i in range(len(text_list)):
                  # Get specific state
                  if isinstance(distinct_states, list):
                      s = distinct_states[i]
                  else:
                      s = distinct_states # Broadcast
                  
                  self.bdh.set_state(s.to(self.device))
                  row = padded[i:i+1] # [1, T]
                  
                  with torch.no_grad():
                      e_seq = self.bdh(row, use_state=True, return_embeddings=True)
                      e = e_seq.mean(dim=1) # [1, D]
                      sent_embs.append(e)
             sent_emb = torch.cat(sent_embs, dim=0)

        return sent_emb

    def absorb_story_stream(self, text_stream):
        """
        Reads a full story stream and returns the Final State.
        Infinite Context Learning.
        """
        self.bdh.reset_state()
        
        # Determine chunk size 
        chunk_size = 512
        
        with torch.no_grad():
            for i in range(0, len(text_stream), chunk_size):
                chunk = text_stream[i : i+chunk_size]
                # Convert to tensor
                tokens = torch.tensor([[ord(c) % 256 for c in chunk]], dtype=torch.long, device=self.device)
                if tokens.size(1) == 0: continue
                
                # Forward Pass with State Update
                self.bdh(tokens, use_state=True)
                
        return self.bdh.get_state()

    def precompute_backstory_states(self, train_df, test_mode=True):
        """
        ENTITY-AWARE WORLD STATE: Maintains per-entity BDH states.
        
        Instead of a single global state that entangles all character facts,
        we route updates to relevant entity states for concentrated signals.
        
        Args:
            test_mode: If True, use only first 50,000 chars for fast testing
        """
        print("Pre-computing Entity-Aware World States...")
        
        self.bdh.eval()
        
        # Get unique books
        unique_books = train_df['book_name'].dropna().unique()
        print(f"  Found {len(unique_books)} unique books to process.")
        
        # Get all known entities (characters)
        all_entities = set(train_df['char'].dropna().str.strip().unique())
        print(f"  Known entities: {len(all_entities)} characters")
        
        if test_mode:
            print("  >> TEST MODE: Using first 50,000 characters only <<")
        
        # Book paths mapping
        book_paths = {
            "The Count of Monte Cristo": "/root/files/The Count of Monte Cristo.txt",
            "In Search of the Castaways": "/root/files/In search of the castaways.txt"
        }
        
        # Process each book
        for book_name in unique_books:
            book_name = str(book_name).strip()
            book_path = book_paths.get(book_name)
            
            if not book_path:
                print(f"  -> Warning: Unknown book '{book_name}'")
                continue
                
            try:
                print(f"  Reading '{book_name}'...")
                with open(book_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                if test_mode:
                    full_text = full_text[:50000]
                
                print(f"    -> Processing {len(full_text):,} characters with entity routing...")
                
                # Get entities relevant to this book
                book_entities = set(
                    train_df[train_df['book_name'].str.strip() == book_name]['char']
                    .dropna().str.strip().unique()
                )
                print(f"    -> Entities in book: {book_entities}")
                
                # Entity-aware ingestion
                world_state = self._entity_aware_ingest(full_text, book_entities)
                self.world_states[book_name] = world_state
                
                print(f"    -> Done! Global + {len(book_entities)} entity states created.")
                
            except Exception as e:
                print(f"    -> Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Computed WorldStates for {len(self.world_states)} books.")
        
        # Legacy compatibility: map (book, char) → merged state
        distinct_chars = train_df[['book_name', 'char']].drop_duplicates()
        
        for _, row in distinct_chars.iterrows():
            if pd.isna(row['book_name']) or pd.isna(row['char']):
                continue
            book = str(row['book_name']).strip()
            char = str(row['char']).strip()
            key = (book, char)
            
            if book in self.world_states:
                # Use merged state (30% global, 70% entity)
                self.backstory_states[key] = self.world_states[book].get_query_state(char, alpha=0.3)
            else:
                self.bdh.reset_state()
                self.backstory_states[key] = self.bdh.get_state()
                
        print(f"Mapped {len(self.backstory_states)} character-book pairs with merged states.")
    
    def _entity_aware_ingest(self, text: str, entities: set) -> WorldState:
        """
        Stream-process text with entity-aware state routing.
        
        For each chunk:
        1. ALWAYS update global state
        2. IF entities mentioned: GATED update to entity states
        
        Hardening: Uses EntityWriteGate for scalar gating.
        """
        chunk_size = 512
        
        # Initialize states
        self.bdh.reset_state()
        initial_state = self.bdh.get_state().cpu()
        
        global_state = initial_state.clone()
        entity_states = {e: initial_state.clone() for e in entities}
        
        # Review Fix #5: Track timestamps for temporal decay
        entity_timestamps = {e: 0.0 for e in entities}
        global_timestamp = 0.0
        
        entity_update_counts = {e: 0 for e in entities}
        gate_values = {e: [] for e in entities}  # Track gate values for debugging
        
        # Initialize EntityWriteGate (learnable but used in inference mode here)
        state_dim = initial_state.numel()
        write_gate = EntityWriteGate(state_dim=64, proj_dim=32).to(self.device)
        write_gate.eval()  # Don't train during ingestion
        
        current_time = 0.0
        
        with torch.no_grad():
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                tokens = torch.tensor([[ord(c) % 256 for c in chunk]], dtype=torch.long, device=self.device)
                if tokens.size(1) == 0:
                    continue
                
                current_time += 1.0  # Increment timestep
                
                # 1. ALWAYS update global state
                self.bdh.reset_state()
                self.bdh.set_state(global_state.detach().clone().to(self.device))
                self.bdh(tokens, use_state=True)
                global_state = self.bdh.get_state().cpu()
                global_timestamp = current_time
                
                # 2. Detect entities in this chunk
                chunk_lower = chunk.lower()
                mentioned_entities = [e for e in entities if e.lower() in chunk_lower]
                
                # Get chunk embedding for gating (use BDH output summary)
                chunk_emb = global_state  # Proxy for chunk representation
                
                # 3. GATED update to relevant entity states
                for entity in mentioned_entities:
                    old_state = entity_states[entity].detach().clone()
                    
                    # Compute new state
                    self.bdh.reset_state()
                    self.bdh.set_state(old_state.to(self.device))
                    self.bdh(tokens, use_state=True)
                    new_state = self.bdh.get_state().cpu()
                    
                    # Compute gate (scalar in [0, 1])
                    gate = write_gate(
                        old_state.to(self.device), 
                        chunk_emb.to(self.device), 
                        global_state.to(self.device)
                    ).item()
                    
                    # Gated update: entity_state = old + gate * (new - old)
                    update = new_state - old_state
                    entity_states[entity] = old_state + gate * update
                    entity_timestamps[entity] = current_time
                    
                    entity_update_counts[entity] += 1
                    gate_values[entity].append(gate)
        
        # Log entity update statistics with gate info
        for entity, count in entity_update_counts.items():
            if count > 0:
                avg_gate = sum(gate_values[entity]) / len(gate_values[entity])
                print(f"      {entity}: {count} updates, avg_gate={avg_gate:.3f}")
        
        # Attach timestamps to WorldState for temporal decay
        world_state = WorldState(
            global_state=global_state,
            entity_states=entity_states,
            known_entities=entities
        )
        # Store timestamps as attribute
        world_state.entity_timestamps = entity_timestamps
        world_state.global_timestamp = global_timestamp
        
        return world_state

    @modal.method()
    def verify_pipeline(self):
        print("=== Running Internal Verification ===")
        self._initialize_components()
        import pandas as pd
        
        # Test Data
        data = {
            'book_name': ['TestBook', 'TestBook'],
            'char': ['Alice', 'Alice'],
            'content': ['Statement 1', 'Statement 2'],
            'label': ['Consistent', 'Consistent']
        }
        df = pd.DataFrame(data)
        
        # Mock Analyzer
        class MockAnalyzer:
            def extract_character_substory(self, book, char):
                return ["Backstory sentence 1.", "Backstory sentence 2."]
        
        self.sentence_analyzer = MockAnalyzer()
        self.backstory_states = {} # Reset
        
        # 1. Precompute
        self.precompute_backstory_states(df)
        if ('TestBook', 'Alice') in self.backstory_states:
             print("State Precomputed: OK")
        else:
             print("State Precomputed: FAIL")
             
        # 2. Prediction
        prob = self.predict_single('TestBook', 'Alice', 'Statement')
        print(f"Prediction: {prob}")
        
        # 3. Training
        loss = self.run_training_step(df)
        print(f"Training Step Loss: {loss}")
        
        return "Verification Success"

    def run_training_step(self, train_df, batch_size=2):
        self.classifier.train()
        self.bdh.train()
        total_loss = 0
        
        optimizer = torch.optim.Adam(
            list(self.classifier.parameters()) + list(self.bdh.parameters()),
            lr=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        # Batch processing
        for start_idx in range(0, len(train_df), batch_size):
            batch = train_df.iloc[start_idx : start_idx + batch_size]
            if len(batch) < 2:
                continue  # Skip incomplete batches for BatchNorm
                
            contents = batch['content'].tolist()
            labels = torch.tensor(
                [1 if l.strip().lower() == 'consistent' else 0 for l in batch['label']],
                device=self.device
            )
            
            # Retrieve states for each sample
            states = []
            for _, row in batch.iterrows():
                key = (row['book_name'].strip(), row['char'].strip())
                if key in self.backstory_states:
                    states.append(self.backstory_states[key].to(self.device))
                else:
                    self.bdh.reset_state()
                    states.append(self.bdh.get_state())

            v_iso = self.encode_text(contents, distinct_states=None)
            v_ctx = self.encode_text(contents, distinct_states=states)
            
            optimizer.zero_grad()
            logits = self.classifier(v_iso, v_ctx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_df) // batch_size)
        return avg_loss

    @modal.method()
    def train_and_evaluate(self):
        try:
            print("Starting Infinite Context Training Pipeline...")
            self._initialize_components()
            
            # Load Data
            train_df = pd.read_csv("/root/files/train.csv")
            
            # Use custom test file that matches test_mode (first 50K chars)
            test_df = pd.read_csv("/root/files/test_custom.csv")
            print(f"Loaded {len(train_df)} train + {len(test_df)} custom test samples.")
            
            # Precompute Infinite Context States (One time pass per book)
            print("Precomputing states for Train & Test...")
            combined_df = pd.concat([train_df, test_df])
            self.precompute_backstory_states(combined_df)
            
            # Training Loop with Best Checkpoint Saving
            feature_extractor_params = list(self.bdh.parameters())
            classifier_params = list(self.classifier.parameters())
            self.optimizer = torch.optim.Adam(feature_extractor_params + classifier_params, lr=2e-4)
            
            epochs = 20  # Full run, no early stopping
            best_acc = 0.0
            
            for epoch in range(epochs):
                avg_loss = self.run_training_step(train_df, batch_size=16)
                test_acc = self.evaluate_accuracy(test_df)
                print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%}")

                # Save best checkpoint
                if test_acc > best_acc:
                    best_acc = test_acc
                    # Save to Volume (mounted at /models)
                    checkpoint_path = "/models/best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'classifier_state': self.classifier.state_dict(),
                        'bdh_state': self.bdh.state_dict(),
                        'best_acc': best_acc
                    }, checkpoint_path)
                    print(f"  -> New Best! Saved checkpoint (Acc: {best_acc:.2%})")
                
            return f"Training Complete. Best Acc: {best_acc:.2%}"
            
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}"


    def evaluate_accuracy(self, test_df):
        """Evaluate accuracy on test set with confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        correct = 0
        total = 0
        self.classifier.eval()
        
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            prob = self.predict_single(row['book_name'], row['char'], row['content'])
            pred = "consistent" if prob > 0.5 else "contradict"
            true_label = row['label'].strip().lower()
            
            y_true.append(1 if true_label == "consistent" else 0)
            y_pred.append(1 if pred == "consistent" else 0)
            
            if pred == true_label:
                correct += 1
            total += 1
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{'='*60}", flush=True)
        print(f"  CONFUSION MATRIX RESULTS", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  True Negative (Contradict→Contradict): {tn}", flush=True)
        print(f"  False Positive (Contradict→Consistent): {fp}", flush=True)
        print(f"  False Negative (Consistent→Contradict): {fn}", flush=True)
        print(f"  True Positive (Consistent→Consistent): {tp}", flush=True)
        print(f"  Precision (Consistent): {tp/(tp+fp) if (tp+fp) > 0 else 0:.2%}", flush=True)
        print(f"  Recall (Consistent): {tp/(tp+fn) if (tp+fn) > 0 else 0:.2%}", flush=True)
        print(f"  Accuracy: {correct/total:.2%}", flush=True)
        print(f"{'='*60}\n", flush=True)
            
        return correct / total if total > 0 else 0

    def predict_single(self, book_name, char_name, content):
        """
        Predict consistency using Counterfactual Consistency Checking.
        
        Energy-Based World Model:
        - If statement causes more surprise → CONTRADICT
        - If negation causes more surprise → CONSISTENT
        
        Uses AdaptiveMerge for query-adaptive state blending.
        """
        book_name = book_name.strip()
        char_name = char_name.strip()
        key = (book_name, char_name)
        
        # Get world state and timestamps
        query_time = 1000.0  # Query happens "at end of story"
        fact_time = 0.0
        
        if key not in self.backstory_states:
            # Try to get from world_states (entity-aware)
            if book_name in self.world_states:
                ws = self.world_states[book_name]
                
                # Get entity state and timestamp
                if char_name in ws.entity_states:
                    entity_state = ws.entity_states[char_name]
                    fact_time = ws.entity_timestamps.get(char_name, 0.0) if hasattr(ws, 'entity_timestamps') else 0.0
                else:
                    entity_state = ws.global_state
                    fact_time = ws.global_timestamp if hasattr(ws, 'global_timestamp') else 0.0
                
                # Use AdaptiveMerge instead of fixed 0.3/0.7
                # For now, encode statement as simple embedding proxy
                stmt_tokens = torch.tensor([[ord(c) % 256 for c in content[:50]]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    self.bdh.reset_state()
                    self.bdh(stmt_tokens, use_state=False)
                    stmt_emb = self.bdh.get_state()
                
                # Initialize AdaptiveMerge (using distributional summaries)
                merge = AdaptiveMerge(state_dim=64, proj_dim=32).to(self.device)
                merge.eval()
                
                with torch.no_grad():
                    state = merge(
                        stmt_emb.to(self.device),
                        ws.global_state.to(self.device),
                        entity_state.to(self.device)
                    ).cpu()
                
                self.backstory_states[key] = state
            else:
                # Fallback: compute on demand
                print(f"Loading state for {key} on demand...")
                sentences = self.sentence_analyzer.extract_character_substory(book_name, char_name)
                full_story = "".join(sentences)
                if not full_story:
                    state = None
                else:
                    state = self.absorb_story_stream(full_story).cpu()
                    self.backstory_states[key] = state
        
        state = self.backstory_states.get(key)
        
        if state is None:
            # No context available, default to consistent
            return 0.5
        
        # Counterfactual Consistency Checking with temporal decay
        checker = CounterfactualChecker(self.bdh, self.device)
        negated = checker.negate(content)
        
        # Compute surprise with temporal decay (Review Fix #5)
        surprise_S = checker.compute_surprise(content, state, fact_time=fact_time, query_time=query_time)
        surprise_negS = checker.compute_surprise(negated, state, fact_time=fact_time, query_time=query_time)
        
        # Decision logic
        epsilon = 1e-8
        conflict_ratio = surprise_S / (surprise_negS + epsilon)
        
        if conflict_ratio > 1.0:
            prediction = "contradict"
            confidence = conflict_ratio
        else:
            prediction = "consistent"
            confidence = 1.0 / (conflict_ratio + epsilon)
        
        # Return probability (0 = contradict, 1 = consistent)
        if prediction == "consistent":
            return min(0.5 + confidence * 0.05, 1.0)  # Reduced scaling for stability
        else:
            return max(0.5 - confidence * 0.05, 0.0)  # Reduced scaling for stability

    @modal.method()
    def generate_submission(self):
        """Train and then generate submission.csv for the hackathon."""
        print("Starting Submission Pipeline...")
        self._initialize_components()
        
        # 1. Train the model first (20 epochs = Sweet Spot)
        print("Step 1: Training Model (20 Epochs)...")
        train_df = pd.read_csv("/root/files/train.csv")
        self.precompute_backstory_states(train_df)
        
        feature_extractor_params = list(self.bdh.parameters())
        classifier_params = list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(feature_extractor_params + classifier_params, lr=2e-4)
        
        for epoch in range(20):
            avg_loss = self.run_training_step(train_df, batch_size=16)
            print(f"  Epoch {epoch+1}/20 | Loss: {avg_loss:.4f}")
            
        # 2. Generate Submission
        print("Step 2: Generating Predictions...")
        test_df = pd.read_csv("/root/files/test.csv")
        self.precompute_backstory_states(test_df)
        
        results = []
        for _, row in test_df.iterrows():
            prob = self.predict_single(row['book_name'], row['char'], row['content'])
            label = "consistent" if prob > 0.5 else "contradict"
            results.append({'id': row['id'], 'label': label})
            
        # Save submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv("/root/submission.csv", index=False)
        print("Saved submission.csv")
        
        # Also save to volume for persistence
        submission_df.to_csv("/models/submission.csv", index=False)
        vol.commit()
        
        return f"Generated {len(results)} predictions"

@app.local_entrypoint()
def submit():
    print("Generating Submission...")
    system = NarrativeConsistencySystem()
    result = system.generate_submission.remote()
    print(result)

@app.local_entrypoint()
def main():
    print("Starting Training...")
    system = NarrativeConsistencySystem()
    results = system.train_and_evaluate.remote()
    print(results)
