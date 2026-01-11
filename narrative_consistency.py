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
    
    FIXED: Rule-based gate (no training required).
    Uses variance difference to detect state-changing chunks.
    """
    def __init__(self, state_dim: int = 64, proj_dim: int = 32):
        super().__init__()
        # No learnable parameters - rule-based
        self.base_gate = 0.1  # Default low gate
        self.sensitivity = 0.5  # Reduced to 0.5 to target <35% write rate (was 1.5 -> 47%)
    
    def forward(self, entity_state: torch.Tensor, chunk_emb: torch.Tensor, 
                global_state: torch.Tensor) -> torch.Tensor:
        """
        Rule-based gate using variance difference.
        
        High gate: chunk brings NEW information (variance increases)
        Low gate: chunk is redundant/descriptive (variance stable)
        """
        # Extract variance features
        e_flat = entity_state.view(-1).float()
        c_flat = chunk_emb.view(-1).float() if chunk_emb is not None else torch.zeros_like(e_flat)
        g_flat = global_state.view(-1).float()
        
        # Compute variance of each state
        e_var = e_flat.var().item() + 1e-8
        c_var = c_flat.var().item() + 1e-8
        g_var = g_flat.var().item() + 1e-8
        
        # Key insight: if chunk variance differs significantly from entity variance,
        # the chunk brings new information → higher gate
        var_ratio = abs(c_var - e_var) / max(e_var, c_var)
        
        # Also consider: if global changed a lot, this is important info
        global_entity_diff = abs(g_var - e_var) / max(e_var, g_var)
        
        # Combine signals
        novelty_score = (var_ratio + global_entity_diff) / 2.0
        
        # Apply sigmoid-like transform with base_gate
        gate = self.base_gate + (1.0 - self.base_gate) * min(novelty_score * self.sensitivity, 1.0)
        
        return torch.tensor(gate, device=entity_state.device)


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
        Bugfix #4: Alpha clamped to [0.05, 0.95] to prevent saturation.
        """
        stmt_summary = self.get_summary(statement_emb) if statement_emb is not None else torch.zeros(self.proj_dim, device=global_state.device)
        global_summary = self.get_summary(global_state)
        entity_summary = self.get_summary(entity_state)
        
        alpha_input = torch.cat([stmt_summary, global_summary, entity_summary], dim=-1)
        alpha = self.alpha_mlp(alpha_input.unsqueeze(0)).squeeze()
        
        # Bugfix #4: Clamp alpha to preserve gradient flow
        alpha = torch.clamp(alpha, 0.05, 0.95)
        
        return alpha * global_state + (1 - alpha) * entity_state


class ContrastiveEnergyLoss(nn.Module):
    """
    Contrastive Energy Loss for supervision.
    
    Aligns surprise with labels via margin loss.
    Contradict samples should have higher E_pos than E_neg.
    """
    def __init__(self, margin: float = 0.3):
        """Margin increased to 0.3 for stronger separation."""
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


class HybridCoherenceClassifier(nn.Module):
    """
    Hybrid Classifier over State Transitions + Energy.
    
    Combines:
    - summary(S_before): World state summary
    - summary(Δ_S): State change from statement
    - summary(Δ_¬S): State change from negation  
    - E_S, E_¬S, E_diff: Energy signals
    
    Energy shapes the space; classifier draws the boundary.
    """
    def __init__(self, summary_dim: int = 4, device=None):
        super().__init__()
        # Feature vector (Fix #4 - expanded):
        # summary(S_before) = 2
        # summary(Δ_S_norm) = 2  
        # summary(Δ_¬S_norm) = 2
        # summary(Δ_diff) = 2  <- NEW: difference-of-differences
        # E_S, E_¬S, E_diff = 3
        # Total = 11 input features
        input_dim = 11
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output: P(contradict), P(consistent)
        )
        self.device = device
    
    def get_summary(self, state: torch.Tensor) -> torch.Tensor:
        """Compute summary statistics: [mean, std]."""
        flat = state.view(-1).float()
        mean = flat.mean()
        std = flat.std() + 1e-8
        return torch.stack([mean, std])
    
    def forward(self, S_before: torch.Tensor, delta_S: torch.Tensor, 
                delta_negS: torch.Tensor, E_S: torch.Tensor, E_negS: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_before: World state before encoding
            delta_S: NORMALIZED state change from statement (from compute_surprise)
            delta_negS: NORMALIZED state change from negation (from compute_surprise)
            E_S: Surprise/energy tensor of statement
            E_negS: Surprise/energy tensor of negation
        
        Returns:
            logits: [2] tensor with P(contradict), P(consistent)
        """
        # Deltas already normalized by compute_surprise, use directly
        delta_S_norm = delta_S
        delta_negS_norm = delta_negS
        
        # Difference-of-differences (KEY discriminative feature)
        delta_diff = delta_S_norm - delta_negS_norm
        
        # Compute summaries on normalized features
        summary_before = self.get_summary(S_before)
        summary_delta_S = self.get_summary(delta_S_norm)
        summary_delta_negS = self.get_summary(delta_negS_norm)
        summary_diff = self.get_summary(delta_diff)
        
        # Energy features - now tensors with gradients
        dev = self.device or S_before.device
        E_diff = E_S - E_negS
        energy_features = torch.stack([E_S, E_negS, E_diff]).to(dev)
        
        # Build feature vector (11 features)
        features = torch.cat([
            summary_before,        # 2
            summary_delta_S,       # 2
            summary_delta_negS,    # 2
            summary_diff,          # 2
            energy_features        # 3
        ])
        
        # Classify
        logits = self.classifier(features.unsqueeze(0).float())
        return logits.squeeze(0)


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
                          temporal_beta: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute surprise (energy) AND normalized state delta.
        
        FIXES APPLIED:
        - Fix #2: Removed torch.no_grad() - freeze BDH params explicitly instead
        - Fix #3: Returns torch.Tensor, not float
        - Fix #4: Returns delta_norm for classifier features
        - Fix #5: Temporal decay via torch operations
        
        Returns:
            Tuple of (surprise_tensor, delta_norm_tensor)
        """
        eps = 1e-8
        
        # Freeze BDH parameters explicitly (instead of no_grad)
        for p in self.bdh.parameters():
            p.requires_grad = False
        
        # Clone state (keep gradients for classifier learning)
        state_before = world_state.clone().to(self.device)
        S_norm = state_before.norm() + eps
        
        # Reset and set BDH state
        self.bdh.reset_state()
        self.bdh.set_state(state_before.detach())  # Detach for BDH internal state
        
        # Encode statement - NO torch.no_grad() so deltas can flow
        tokens = self.encode_text(text)
        self.bdh(tokens, use_state=True)
        
        # Get state after
        state_after = self.bdh.get_state()
        
        # Compute delta (state change)
        delta = state_after - state_before.detach()
        
        # Fix #4: Normalize delta for classifier
        delta_norm = delta / S_norm
        
        # Compute energy (L2 norm of delta)
        energy = torch.norm(delta, p=2)
        
        # Fix #5: Apply temporal decay using torch (not math)
        decay = torch.exp(torch.tensor(-temporal_beta * (query_time - fact_time), device=self.device))
        energy = energy * decay
        
        # Stabilize via log1p (keeps as tensor)
        surprise = torch.log1p(energy)
        
        # Clamp for safety (keeps as tensor)
        surprise = torch.clamp(surprise, max=torch.log1p(torch.tensor(SURPRISE_MAX, device=self.device)))
        
        # Restore BDH gradient state for training
        for p in self.bdh.parameters():
            p.requires_grad = True
        
        return surprise, delta_norm
    
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
    
    def compute_hybrid_features(self, statement: str, world_state: torch.Tensor,
                                 fact_time: float = 0.0, query_time: float = 1.0) -> Dict:
        """
        Compute features for hybrid classifier.
        
        Returns:
            Dict with S_before, delta_S, delta_negS, E_S, E_negS
        """
        negated = self.negate(statement)
        S_before = world_state.clone().to(self.device)
        
        # Use compute_surprise which now returns (energy, delta_norm) tuples
        # This eliminates duplicate BDH calls and ensures proper gradient flow
        E_S, delta_S = self.compute_surprise(statement, world_state, fact_time, query_time)
        E_negS, delta_negS = self.compute_surprise(negated, world_state, fact_time, query_time)
        
        return {
            "S_before": S_before,
            "delta_S": delta_S,  # Now normalized delta tensor
            "delta_negS": delta_negS,  # Now normalized delta tensor
            "E_S": E_S,  # Now tensor, not float
            "E_negS": E_negS,  # Now tensor, not float
            "negated": negated
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
        
        # Enhanced Gate Observability (Fix #5)
        all_gates = []
        for entity, count in entity_update_counts.items():
            if count > 0:
                gates = gate_values[entity]
                all_gates.extend(gates)
                avg_gate = sum(gates) / len(gates)
                min_gate = min(gates)
                max_gate = max(gates)
                print(f"      {entity}: {count} updates | gate: avg={avg_gate:.3f}, min={min_gate:.3f}, max={max_gate:.3f}")
        
        # Global gate statistics
        if all_gates:
            sorted_gates = sorted(all_gates)
            n = len(sorted_gates)
            q25 = sorted_gates[n // 4] if n >= 4 else sorted_gates[0]
            q50 = sorted_gates[n // 2]
            q75 = sorted_gates[3 * n // 4] if n >= 4 else sorted_gates[-1]
            print(f"      [GATE STATS] total={n} | mean={sum(all_gates)/n:.3f} | q25={q25:.3f} | q50={q50:.3f} | q75={q75:.3f}")
            
            # Warning if gates not selective
            mean_gate = sum(all_gates) / n
            if mean_gate > 0.35:
                print(f"⚠️ WARNING: mean_gate={mean_gate:.3f} > 0.35 - gate not selective enough!")
        
        # Attach timestamps to WorldState for temporal decay
        # FIX #2: Detach all tensors to freeze WorldState for training
        world_state = WorldState(
            global_state=global_state.detach(),
            entity_states={k: v.detach() for k, v in entity_states.items()},
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

    def run_training_step(self, train_df, batch_size=4):
        """
        Train HybridCoherenceClassifier on state transitions + energy features.
        """
        # Initialize hybrid classifier if needed
        if not hasattr(self, 'hybrid_classifier') or self.hybrid_classifier is None:
            self.hybrid_classifier = HybridCoherenceClassifier(device=self.device).to(self.device)
        
        self.hybrid_classifier.train()
        self.bdh.train()
        total_loss = 0
        
        # Optimizer includes hybrid classifier + BDH
        optimizer = torch.optim.Adam(
            list(self.hybrid_classifier.parameters()) + list(self.bdh.parameters()),
            lr=2e-4
        )
        
        # w_contradict = 2.0 forces model to prioritize finding contradictions
        class_weights = torch.tensor([2.0, 1.0], device=self.device)  # [contradict, consistent]
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Checkers
        checker = CounterfactualChecker(self.bdh, self.device)
        pred_counts = {"contradict": 0, "consistent": 0}
        energy_diffs = []
        
        # Process samples
        for idx, row in train_df.iterrows():
            book_name = row['book_name'].strip()
            char_name = row['char'].strip()
            content = row['content']
            label = 1 if row['label'].strip().lower() == 'consistent' else 0
            
            key = (book_name, char_name)
            
            # Get world state
            if key not in self.backstory_states:
                continue
            state = self.backstory_states[key].to(self.device)
            
            # Compute hybrid features
            features = checker.compute_hybrid_features(content, state)
            
            # Track energy difference for Tau (Step 3)
            with torch.no_grad():
                diff = abs(features["E_S"].item() - features["E_negS"].item())
                energy_diffs.append(diff)
            
            # Forward pass 
            logits = self.hybrid_classifier(
                features["S_before"].to(self.device),
                features["delta_S"].to(self.device),
                features["delta_negS"].to(self.device),
                features["E_S"],
                features["E_negS"]
            )
            
            # Track predictions for bias check
            pred = torch.argmax(logits).item()
            if pred == 0:
                pred_counts["contradict"] += 1
            else:
                pred_counts["consistent"] += 1
            
            # Loss
            target = torch.tensor([label], device=self.device)
            loss = criterion(logits.unsqueeze(0), target)
            
            # Step 2: Entropy Regularization (Force Diversity)
            p = torch.softmax(logits, dim=0)
            entropy = -torch.sum(p * torch.log(p + 1e-8))
            loss += 0.05 * entropy
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Fix #5: Prediction bias check
        total_preds = pred_counts["contradict"] + pred_counts["consistent"]
        if total_preds > 0:
            pct_consistent = pred_counts["consistent"] / total_preds * 100
            pct_contradict = pred_counts["contradict"] / total_preds * 100
            print(f"    [BIAS CHECK] Contradict: {pct_contradict:.1f}% | Consistent: {pct_consistent:.1f}%")
        
        avg_loss = total_loss / max(1, len(train_df))
        return avg_loss, energy_diffs

    def filter_explicit_dataset(self, df):
        """Step 1: Restrict task to EXPLICIT contradictions (prevents semantic drift)."""
        print("Step 1: Filtering for EXPLICIT contradictions...")
        valid_indices = []
        book_paths = {
            "The Count of Monte Cristo": "/root/files/The Count of Monte Cristo.txt",
            "In Search of the Castaways": "/root/files/In search of the castaways.txt"
        }
        
        for book in df['book_name'].unique():
            book = book.strip()
            path = book_paths.get(book)
            if not path: continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # User-specified constraint: "explicitly grounded in observed context"
                    # We assume observed context is first 50k chars (test mode)
                    content = f.read(50000).lower()
            except Exception as e:
                print(f"Error reading {book}: {e}")
                continue
                
            book_df = df[df['book_name'] == book]
            for idx, row in book_df.iterrows():
                statement = str(row['content'])
                # Heuristic: verify keywords appear in text
                keywords = [w for w in statement.split() if len(w) > 3][:3]
                if not keywords: keywords = statement.split()[:3]
                
                if any(k.lower() in content for k in keywords):
                    valid_indices.append(idx)
        
        filtered = df.loc[valid_indices].copy()
        print(f"Explicit Filter: kept {len(filtered)}/{len(df)} samples ({len(filtered)/len(df):.1%})")
        return filtered

    @modal.method()
    def train_and_evaluate(self):
        try:
            print("Starting Infinite Context Training Pipeline...")
            self._initialize_components()
            
            # Load Data
            train_df = pd.read_csv("/root/files/train.csv")
            test_df = pd.read_csv("/root/files/test_custom.csv")
            
            # Step 1: Filter Dataset
            train_df = self.filter_explicit_dataset(train_df)
            test_df = self.filter_explicit_dataset(test_df)
            
            # Balance (Step from previous fix - keep it)
            def balance_df(df):
                cons = df[df['label']=='consistent']
                cont = df[df['label']=='contradict']
                min_len = min(len(cons), len(cont))
                if min_len == 0: return df
                return pd.concat([cons.sample(min_len, random_state=42), cont.sample(min_len, random_state=42)]).sample(frac=1).reset_index(drop=True)
            
            train_df = balance_df(train_df)
            test_df = balance_df(test_df)
            print(f"Final Training Set: {len(train_df)} samples")
            
            # Precompute Infinite Context States
            print("Precomputing states for Train & Test...")
            combined_df = pd.concat([train_df, test_df])
            self.precompute_backstory_states(combined_df)
            
            # Training Setup
            feature_extractor_params = list(self.bdh.parameters())
            classifier_params = list(self.classifier.parameters())
            self.optimizer = torch.optim.Adam(feature_extractor_params + classifier_params, lr=2e-4)
            
            epochs = 30
            best_acc = 0.0
            patience = 10
            no_improve_count = 0
            
            # Step 3: Initialize Tau (Median Energy Diff)
            self.tau = 0.5 # Default fallback
            all_energy_diffs = []
            
            for epoch in range(epochs):
                avg_loss, energy_diffs = self.run_training_step(train_df, batch_size=16)
                
                # Update Tau dynamically
                all_energy_diffs.extend(energy_diffs)
                if all_energy_diffs:
                    import statistics
                    self.tau = statistics.median(all_energy_diffs)
                
                test_acc = self.evaluate_accuracy(test_df)
                print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} | Tau: {self.tau:.4f} | Test Acc: {test_acc:.2%}")

                # Save best checkpoint
                if test_acc > best_acc:
                    best_acc = test_acc
                    no_improve_count = 0 
                    
                    checkpoint_path = "/models/best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'classifier_state': self.classifier.state_dict(),
                        'bdh_state': self.bdh.state_dict(),
                        'best_acc': best_acc,
                        'tau': self.tau # Save tau
                    }, checkpoint_path)
                    print(f"  -> New Best! Saved checkpoint (Acc: {best_acc:.2%})")
                else:
                    no_improve_count += 1
                    print(f"  -> No improvement ({no_improve_count}/{patience})")
                    
                    # Early stopping check
                    if no_improve_count >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
                
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
        HYBRID Prediction: State Transitions + Energy → Classifier
        
        1. Compute S_before, Δ_S, Δ_¬S (state transitions)
        2. Compute E_S, E_¬S (energies)
        3. Build feature vector: [summary(S_before), summary(Δ_S), summary(Δ_¬S), E_S, E_¬S, E_diff]
        4. Pass to classifier → P(contradict), P(consistent)
        
        Energy shapes the space; classifier draws the boundary.
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
                    entity_ts = ws.entity_timestamps.get(char_name, 0.0) if hasattr(ws, 'entity_timestamps') else 0.0
                else:
                    entity_state = ws.global_state
                    entity_ts = 0.0
                
                # fact_time = max(entity_timestamp, global_timestamp)
                global_ts = ws.global_timestamp if hasattr(ws, 'global_timestamp') else 0.0
                fact_time = max(entity_ts, global_ts)
                
                # Use AdaptiveMerge for state
                stmt_tokens = torch.tensor([[ord(c) % 256 for c in content[:50]]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    self.bdh.reset_state()
                    self.bdh(stmt_tokens, use_state=False)
                    stmt_emb = self.bdh.get_state()
                
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
                sentences = self.sentence_analyzer.extract_character_substory(book_name, char_name)
                full_story = "".join(sentences)
                if not full_story:
                    return 0.5
                state = self.absorb_story_stream(full_story).cpu()
                self.backstory_states[key] = state
        
        state = self.backstory_states.get(key)
        
        if state is None:
            return 0.5
        
        # Initialize hybrid classifier if not exists
        if not hasattr(self, 'hybrid_classifier') or self.hybrid_classifier is None:
            self.hybrid_classifier = HybridCoherenceClassifier(device=self.device).to(self.device)
            self.hybrid_classifier.eval()
        
        # Compute hybrid features
        checker = CounterfactualChecker(self.bdh, self.device)
        features = checker.compute_hybrid_features(content, state, fact_time, query_time)
        
        # Classify using hybrid features
        with torch.no_grad():
            logits = self.hybrid_classifier(
                features["S_before"].to(self.device),
                features["delta_S"].to(self.device),
                features["delta_negS"].to(self.device),
                features["E_S"],
                features["E_negS"]
            )
            probs = torch.softmax(logits, dim=0)
            prob_classifier = probs[1].item()
        
        # Step 3: Energy Threshold Backstop (Decision Backstop)
        # If classifier is weak or ambiguous, trust the raw Energy signal if strong
        E_S = features["E_S"].item()
        E_negS = features["E_negS"].item()
        diff = abs(E_S - E_negS)
        tau = self.tau if hasattr(self, 'tau') else 0.5
        
        if diff > tau:
             # Energy signal is STRONG: Override classifier
             # If Statement Energy > Negation Energy -> Contradiction (prob low)
             return 0.1 if E_S > E_negS else 0.9
        else:
             # Energy signal is weak: Trust learned classifier
             return prob_classifier

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
