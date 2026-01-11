import modal
import torch
import torch.nn as nn
import pandas as pd
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Internal imports from the package
from .world_state import WorldState, EntityWriteGate, AdaptiveMerge
from .consistency import CounterfactualChecker, ContrastiveEnergyLoss, SURPRISE_MAX
from .models import CoherenceClassifier, NarrativeDataset

# Fix for import resolution when running inside Modal vs Local
# Assuming bdh.py and sentence_analyzer.py are in the parent directory (root of KDSH)
# We might need to add parent dir to path if not present
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from bdh import BDH, BDHConfig
    from sentence_analyzer import SentenceAnalyzer
except ImportError:
    # If standard import fails, try relative (though usually sys.path handles it)
    print("Warning: Could not import BDH/SentenceAnalyzer directly. Please ensure they are in the python path.")
    raise

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
        # Modal remote path adjustment
        if os.path.exists("/root"):
            sys.path.append("/root")
            
        print(f"ls /root: {os.listdir('/root')}" if os.path.exists("/root") else "Local execution")
        
        # Re-import BDH config to be sure
        from bdh import BDH, BDHConfig
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized on {self.device}")
        
        self.config = BDHConfig(n_layer=6, n_embd=EMBEDDING_DIM, n_head=4)
        self.bdh = BDH(self.config).to(self.device)
        self.bdh.eval()
        
        self.classifier = CoherenceClassifier(EMBEDDING_DIM, self.device).to(self.device)
        
        # Load Weights if available
        # Check both Modal path and local path
        weights_path = "/models/narrative_consistency.pt"
        if not os.path.exists(weights_path) and os.path.exists("./models/narrative_consistency.pt"):
             weights_path = "./models/narrative_consistency.pt"
             
        if os.path.exists(weights_path):
            try:
                self.classifier.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
                print("Loaded classifier weights (strict=False).")
            except Exception as e:
                print(f"Weights load failed: {e}")
                
        bdh_path = "/models/bdh_base.pt"
        if not os.path.exists(bdh_path) and os.path.exists("./models/bdh_base.pt"):
             bdh_path = "./models/bdh_base.pt"

        if os.path.exists(bdh_path):
             self.bdh.load_state_dict(torch.load(bdh_path, map_location=self.device), strict=False)

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
            
            # Local fallback for paths
            if not book_path:
                 # try local ./files
                 local_path = f"./files/{book_name}.txt"
                 if os.path.exists(local_path):
                      book_path = local_path
            
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
        """
        chunk_size = 512
        
        # Initialize states
        self.bdh.reset_state()
        initial_state = self.bdh.get_state().cpu()
        
        global_state = initial_state.clone()
        entity_states = {e: initial_state.clone() for e in entities}
        
        entity_timestamps = {e: 0.0 for e in entities}
        global_timestamp = 0.0
        
        entity_update_counts = {e: 0 for e in entities}
        gate_values = {e: [] for e in entities} 
        
        # Initialize EntityWriteGate (learnable but used in inference mode here)
        state_dim = initial_state.numel()
        write_gate = EntityWriteGate(state_dim=64, proj_dim=32).to(self.device)
        write_gate.eval() 
        
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
        
        # Enhanced Gate Observability
        all_gates = []
        for entity, count in entity_update_counts.items():
            if count > 0:
                gates = gate_values[entity]
                all_gates.extend(gates)
        
        # Attach timestamps to WorldState for temporal decay
        world_state = WorldState(
            global_state=global_state.detach(),
            entity_states={k: v.detach() for k, v in entity_states.items()},
            known_entities=entities
        )
        world_state.entity_timestamps = entity_timestamps
        world_state.global_timestamp = global_timestamp
        
        return world_state

    @modal.method()
    def verify_pipeline(self):
        print("=== Running Internal Verification ===")
        self._initialize_components()
        
        # Test Data
        data = {
            'book_name': ['TestBook', 'TestBook'],
            'char': ['Alice', 'Alice'],
            'content': ['Statement 1', 'Statement 2'],
            'label': ['Consistent', 'Consistent']
        }
        df = pd.DataFrame(data)
        # Mock precompute
        self.precompute_backstory_states(df, test_mode=True)
        print("Verification complete.")
