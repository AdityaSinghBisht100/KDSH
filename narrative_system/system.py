import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
import random
from sentence_transformers import SentenceTransformer

from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from .world_state import WorldState

# Constants
EMBEDDING_DIM = 256 # Internal memory dimension
SEMANTIC_DIM = 384  # Based on all-MiniLM-L6-v2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NarrativeConsistencySystem:
    def __init__(self, data_dir="./files", model_dir="./models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
        self.encoder = None
        self.bdh = None
        self.checker = None
        self.device = None
        
        # Entity-Aware WorldState: book_name → WorldState
        self.world_states = {}
        self.backstory_states = {} 

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
        if self.encoder is not None:
            return

        print(f"Initializing components in pid {os.getpid()}")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ CUDA NOT available. Using CPU.")
        
        # 1. Load Semantic Encoder
        print(f"  🧠 Loading Semantic Encoder (MiniLM)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.encoder.eval()

        # 2. Initialize GPT2-BDH Transformer (Vector Space Version)
        from bdh_transformer import GPT2BDHTransformer
        from dataclasses import dataclass
        
        @dataclass
        class Config:
            n_layer: int = 4
            n_embd: int = EMBEDDING_DIM
            input_dim: int = SEMANTIC_DIM # 384
            n_head: int = 4
            block_size: int = 1024
            dropout: float = 0.1
            
        set_seed(42)
        self.config = Config()
        self.bdh = GPT2BDHTransformer(self.config).to(self.device)
        self.bdh.eval()
        
        # 3. Load Trained Weights if exist
        bdh_path = os.path.abspath(os.path.join(self.model_dir, "bdh_transformer.pt"))
        if os.path.exists(bdh_path):
             try:
                 self.bdh.load_state_dict(torch.load(bdh_path, map_location=self.device, weights_only=False), strict=False)
                 print("  ✅ Semantic Transformer weights loaded.")
             except Exception as e:
                 print(f"  ⚠️ Weights incompatible ({e}). Using baseline.")

        # 4. Linkage Checker
        from .linkage_check import NarrativeLinkageChecker
        self.checker = NarrativeLinkageChecker(self.bdh, self.encoder, self.device)
        
        print("--> Initialization complete")

    def encode_text(self, text_list, distinct_states=None):
        """
        Convert list of strings into Semantic Vectors [B, 1, D].
        """
        if self.encoder is None:
            self._initialize_components()
            
        if not text_list:
            # Handle empty list case
            return torch.zeros((1, 1, SEMANTIC_DIM), device=self.device)
            
        # 1. Encode into LLM Space (MiniLM 384-dim)
        with torch.no_grad():
            vecs_np = self.encoder.encode(text_list, convert_to_numpy=True)
            
        # 2. Convert to Tensor [B, 1, D]
        vecs = torch.from_numpy(vecs_np).to(self.device).unsqueeze(1)
        
        # 3. Apply Context if States are provided
        if distinct_states is not None:
             sent_embs = []
             for i in range(len(text_list)):
                  s = distinct_states[i] if isinstance(distinct_states, list) else distinct_states
                  # s is a list of [nh, d, d] tensors (one per layer)
                  self.bdh.set_state(s)
                  
                  row = vecs[i:i+1] # [1, 1, D]
                  with torch.no_grad():
                      refined_vec, _, _ = self.bdh(row, use_state=True)
                      sent_embs.append(refined_vec)
             # Return [B, 1, D] refined vectors
             return torch.cat(sent_embs, dim=0).unsqueeze(1) 

        return vecs

    def absorb_story_stream(self, text_stream):
        """
        Reads a full story stream and returns the Final State.
        Uses sentence-level semantic chunks.
        """
        if self.encoder is None: self._initialize_components()
        self.bdh.reset_state()
        
        # Simple sentence-like chunking for story ingestion
        import re
        chunks = re.split(r'(?<=[.!?])\s+', text_stream)
        chunks = [c.strip() for c in chunks if len(c.strip()) > 5]
        
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            with torch.no_grad():
                # Get embeddings [B, 384]
                vecs_np = self.encoder.encode(batch, convert_to_numpy=True)
                vecs = torch.from_numpy(vecs_np).to(self.device).unsqueeze(1) # [B, 1, D]
                
                # Sequential update of the BDH persistent state
                for seq_idx in range(vecs.size(0)):
                    self.bdh(vecs[seq_idx:seq_idx+1], use_state=True)
                
        return self.bdh.get_state()

    def ingest_novel_knowledge(self, train_df, test_mode=True):
        from .ingestion import ingest_novel_knowledge
        ingest_novel_knowledge(self, train_df, test_mode)

    def _entity_aware_ingest(self, text: str, entities: set) -> WorldState:
        # This is now internal to ingestion.py, but if called by other methods, we proxy it
        from .ingestion import _entity_aware_ingest
        return _entity_aware_ingest(self, text, entities)

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
        self.ingest_novel_knowledge(df, test_mode=True)
        print("Verification complete.")

    def filter_explicit_dataset(self, df):
        # Kept locally or could be moved to utils
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
        from .training import evaluate_accuracy
        return evaluate_accuracy(self, test_df)

    def run_training_step(self, train_df, batch_size=4):
        from .training import run_training_step
        return run_training_step(self, train_df, batch_size)

    def predict_single(self, book_name, char_name, content):
        from .inference import predict_single
        return predict_single(self, book_name, char_name, content)

    def train(self, epochs=20):
        from .training import train
        train(self, epochs)

    def generate_predictions(self, input_file="test.csv", output_file="predictions.csv"):
        from .inference import generate_predictions
        generate_predictions(self, input_file, output_file)

    def interactive_session(self):
        from .inference import interactive_session
        interactive_session(self)
