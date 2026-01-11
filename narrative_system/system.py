
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os

from pathlib import Path
from typing import List, Dict, Tuple

# Internal imports from the package
from .world_state import WorldState, EntityWriteGate, AdaptiveMerge
from .consistency import CounterfactualChecker, ContrastiveEnergyLoss, SURPRISE_MAX
from .models import CoherenceClassifier, NarrativeDataset

# Fix for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from bdh import BDH, BDHConfig
    from sentence_analyzer import SentenceAnalyzer
except ImportError:
    print("Warning: Could not import BDH/SentenceAnalyzer directly. Please ensure they are in the python path.")
    raise

# Constants
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256

class NarrativeConsistencySystem:
    def __init__(self, data_dir="./files", model_dir="./models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
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
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _initialize_components(self):
        if self.classifier is not None:
            return

        print(f"Initializing components in pid {os.getpid()}")
        
        # Re-import BDH config to be sure
        from bdh import BDH, BDHConfig
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized on {self.device}")
        
        self.config = BDHConfig(n_layer=6, n_embd=EMBEDDING_DIM, n_head=4)
        self.bdh = BDH(self.config).to(self.device)
        self.bdh.eval()
        
        self.classifier = CoherenceClassifier(EMBEDDING_DIM, self.device).to(self.device)
        
        # Load Weights if available
        weights_path = os.path.join(self.model_dir, "narrative_consistency.pt")
        if os.path.exists(weights_path):
            try:
                self.classifier.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
                print("Loaded classifier weights (strict=False).")
            except Exception as e:
                print(f"Weights load failed: {e}")
                
        bdh_path = os.path.join(self.model_dir, "bdh_base.pt")
        if os.path.exists(bdh_path):
             self.bdh.load_state_dict(torch.load(bdh_path, map_location=self.device), strict=False)

        # Import and initialize sentence analyzer
        from sentence_analyzer import SentenceAnalyzer
        self.sentence_analyzer = SentenceAnalyzer(self.bdh, EMBEDDING_DIM, self.device)

        self.backstory_store = {}
        self.backstory_states = {} # Map (book, char) -> Tensor State
        print("--> Initialization complete")

    def encode_text(self, text_list, distinct_states=None):
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
        Expected to process book content.
        """
        self.bdh.reset_state()
        
        # Determine chunk size 
        chunk_size = 512
        
        with torch.no_grad():
            for i in range(0, len(text_stream), chunk_size):
                chunk = text_stream[i : i+chunk_size]
                tokens = torch.tensor([[ord(c) % 256 for c in chunk]], dtype=torch.long, device=self.device)
                if tokens.size(1) == 0: continue
                
                # Forward Pass with State Update
                self.bdh(tokens, use_state=True)
                
        return self.bdh.get_state()

    def precompute_backstory_states(self, train_df, test_mode=True):
        """
        ENTITY-AWARE WORLD STATE: Maintains per-entity BDH states.
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
            "The Count of Monte Cristo": os.path.join(self.data_dir, "The Count of Monte Cristo.txt"),
            "In Search of the Castaways": os.path.join(self.data_dir, "In search of the castaways.txt")
        }
        
        # Process each book
        for book_name in unique_books:
            book_name = str(book_name).strip()
            # Try specific map first, then generic in data_dir
            book_path = book_paths.get(book_name)
            
            if not book_path or not os.path.exists(book_path):
                 # Try direct filename match
                 candidate = os.path.join(self.data_dir, f"{book_name}.txt")
                 if os.path.exists(candidate):
                     book_path = candidate
            
            if not book_path or not os.path.exists(book_path):
                print(f"  -> Warning: Unknown book or file not found '{book_name}' in {self.data_dir}")
                continue
                
            try:
                print(f"  Reading '{book_name}'...")
                with open(book_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                if test_mode:
                    full_text = full_text[:50000]
                
                print(f"    -> Processing {len(full_text):,} characters with entity routing...")
                
                book_entities = set(
                    train_df[train_df['book_name'].str.strip() == book_name]['char']
                    .dropna().str.strip().unique()
                )
                
                world_state = self._entity_aware_ingest(full_text, book_entities)
                self.world_states[book_name] = world_state
                print(f"    -> Done! Global + {len(book_entities)} entity states created.")
                
            except Exception as e:
                print(f"    -> Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Computed WorldStates for {len(self.world_states)} books.")
        
        distinct_chars = train_df[['book_name', 'char']].drop_duplicates()
        
        for _, row in distinct_chars.iterrows():
            if pd.isna(row['book_name']) or pd.isna(row['char']):
                continue
            book = str(row['book_name']).strip()
            char = str(row['char']).strip()
            key = (book, char)
            
            if book in self.world_states:
                self.backstory_states[key] = self.world_states[book].get_query_state(char, alpha=0.3)
            else:
                self.bdh.reset_state()
                self.backstory_states[key] = self.bdh.get_state()
                
        print(f"Mapped {len(self.backstory_states)} character-book pairs with merged states.")
    
    def _entity_aware_ingest(self, text: str, entities: set) -> WorldState:
        chunk_size = 512
        
        self.bdh.reset_state()
        initial_state = self.bdh.get_state().cpu()
        
        global_state = initial_state.clone()
        entity_states = {e: initial_state.clone() for e in entities}
        
        entity_timestamps = {e: 0.0 for e in entities}
        global_timestamp = 0.0
        
        entity_update_counts = {e: 0 for e in entities}
        gate_values = {e: [] for e in entities} 
        
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
                
                current_time += 1.0
                
                self.bdh.reset_state()
                self.bdh.set_state(global_state.detach().clone().to(self.device))
                self.bdh(tokens, use_state=True)
                global_state = self.bdh.get_state().cpu()
                global_timestamp = current_time
                
                chunk_lower = chunk.lower()
                mentioned_entities = [e for e in entities if e.lower() in chunk_lower]
                chunk_emb = global_state
                
                for entity in mentioned_entities:
                    old_state = entity_states[entity].detach().clone()
                    
                    self.bdh.reset_state()
                    self.bdh.set_state(old_state.to(self.device))
                    self.bdh(tokens, use_state=True)
                    new_state = self.bdh.get_state().cpu()
                    
                    gate = write_gate(
                        old_state.to(self.device), 
                        chunk_emb.to(self.device), 
                        global_state.to(self.device)
                    ).item()
                    
                    update = new_state - old_state
                    entity_states[entity] = old_state + gate * update
                    entity_timestamps[entity] = current_time
                    
                    entity_update_counts[entity] += 1
                    gate_values[entity].append(gate)
        
        world_state = WorldState(
            global_state=global_state.detach(),
            entity_states={k: v.detach() for k, v in entity_states.items()},
            known_entities=entities
        )
        world_state.entity_timestamps = entity_timestamps
        world_state.global_timestamp = global_timestamp
        
        return world_state

    def verify_pipeline(self):
        print("=== Running Internal Verification ===")
        self._initialize_components()
        
        data = {
            'book_name': ['TestBook', 'TestBook'],
            'char': ['Alice', 'Alice'],
            'content': ['Statement 1', 'Statement 2'],
            'label': ['Consistent', 'Consistent']
        }
        df = pd.DataFrame(data)
        self.precompute_backstory_states(df, test_mode=True)
        print("Verification complete.")

    def filter_explicit_dataset(self, df):
        """Step 1: Restrict task to EXPLICIT contradictions."""
        print("Step 1: Filtering for EXPLICIT contradictions...")
        valid_indices = []
        
        book_paths = {
            "The Count of Monte Cristo": os.path.join(self.data_dir, "The Count of Monte Cristo.txt"),
            "In Search of the Castaways": os.path.join(self.data_dir, "In search of the castaways.txt")
        }
        
        for book in df['book_name'].unique():
            book = book.strip()
            path = book_paths.get(book)
            
            if not path or not os.path.exists(path):
                 path = os.path.join(self.data_dir, f"{book}.txt")
                 
            if not path or not os.path.exists(path):
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(50000).lower() 
            except Exception as e:
                print(f"Error reading {book}: {e}")
                continue
                
            book_df = df[df['book_name'] == book]
            for idx, row in book_df.iterrows():
                statement = str(row['content'])
                keywords = [w for w in statement.split() if len(w) > 3][:3]
                if not keywords: keywords = statement.split()[:3]
                
                if any(k.lower() in content for k in keywords):
                    valid_indices.append(idx)
        
        filtered = df.loc[valid_indices].copy()
        print(f"Explicit Filter: kept {len(filtered)}/{len(df)} samples")
        return filtered

    def evaluate_accuracy(self, test_df):
        correct = 0
        total = 0
        self.bdh.eval()
        self.classifier.eval()
        
        if hasattr(self, 'hybrid_classifier') and self.hybrid_classifier is not None:
             self.hybrid_classifier.eval()
        
        with torch.no_grad():
            for _, row in test_df.iterrows():
                book = row['book_name']
                char = row['char']
                content = row['content']
                label = row['label'].strip().lower()
                
                score = self.predict_single(book, char, content)
                
                pred = "consistent" if score > 0.5 else "contradict"
                if pred == label:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0

    def run_training_step(self, train_df, batch_size=4):
        self.classifier.train()
        self.bdh.train()
        total_loss = 0
        
        optimizer = torch.optim.Adam(
            list(self.classifier.parameters()) + list(self.bdh.parameters()),
            lr=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        for start_idx in range(0, len(train_df), batch_size):
            batch = train_df.iloc[start_idx : start_idx + batch_size]
            if len(batch) < 2: continue
            
            contents = batch['content'].tolist()
            labels = torch.tensor(
                [1 if l.strip().lower() == 'consistent' else 0 for l in batch['label']],
                device=self.device
            )
            
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
            
            ce_loss = criterion(logits, labels)
            # Simple loss for now, removed complexity for basic cluster run reliability
            loss = ce_loss 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / max(1, len(train_df) // batch_size)

    def predict_single(self, book_name, char_name, content):
        book_name = book_name.strip()
        char_name = char_name.strip()
        key = (book_name, char_name)
        
        if self.bdh is None: self._initialize_components()
        
        if key not in self.backstory_states:
             if book_name in self.world_states:
                 ws = self.world_states[book_name]
                 state = ws.get_query_state(char_name)
                 self.backstory_states[key] = state.to(self.device)
             else:
                 book_path = os.path.join(self.data_dir, f"{book_name}.txt")
                 if not os.path.exists(book_path):
                      return 0.5 
                 
                 with open(book_path, 'r', encoding='utf-8') as f:
                      text = f.read()
                 self.bdh.reset_state()
                 self.absorb_story_stream(text[:10000])
                 state = self.bdh.get_state()
                 self.backstory_states[key] = state

        state = self.backstory_states[key].to(self.device)
        
        v_iso = self.encode_text([content], distinct_states=None)
        v_ctx = self.encode_text([content], distinct_states=state)
        
        with torch.no_grad():
             logits = self.classifier(v_iso, v_ctx)
             probs = torch.softmax(logits, dim=1)
             return probs[0, 1].item()

    def train(self, epochs=5):
        print(f"Starting Training on {self.device}...")
        self._initialize_components()
        
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")
        
        if not os.path.exists(train_path):
             print(f"Train file not found at {train_path}")
             return
             
        train_df = pd.read_csv(train_path)
        if os.path.exists(test_path):
             test_df = pd.read_csv(test_path)
        else:
             test_df = train_df.copy() # Mock
             
        combined = pd.concat([train_df, test_df])
        self.precompute_backstory_states(combined, test_mode=False)
        
        print("Training...")
        for epoch in range(epochs):
             loss = self.run_training_step(train_df)
             acc = self.evaluate_accuracy(test_df)
             print(f"Epoch {epoch+1}: Loss={loss:.4f} Acc={acc:.2%}")
             
             torch.save(self.classifier.state_dict(), os.path.join(self.model_dir, "narrative_consistency.pt"))
             torch.save(self.bdh.state_dict(), os.path.join(self.model_dir, "bdh_base.pt"))
