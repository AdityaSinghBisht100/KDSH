import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os

from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Internal imports from the package
from .world_state import WorldState, EntityWriteGate, AdaptiveMerge
from .consistency import CounterfactualChecker, ContrastiveEnergyLoss, SURPRISE_MAX
from .models import NarrativeDataset

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
        # Import tiktoken for BPE
        import tiktoken
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ CUDA NOT available. Using CPU.")
        
        print(f"Initialized on {self.device}")
        
        # Initialize BPE Tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            vocab_size = self.tokenizer.n_vocab
            print(f"✅ Tokenizer initialized (Vocab: {vocab_size})")
        except Exception as e:
            print(f"❌ Failed to load tiktoken: {e}")
            raise

        self.config = BDHConfig(n_layer=6, n_embd=EMBEDDING_DIM, n_head=4, vocab_size=vocab_size)
        self.bdh = BDH(self.config).to(self.device)
        self.bdh.eval()
        
        # Classifier removed - we use direct Energy Inference
        self.classifier = None
                
        bdh_path = os.path.join(self.model_dir, "bdh_base.pt")
        if os.path.exists(bdh_path):
             try:
                 # Strict=False to allow shape mismatch if we are reloading old weights (though they won't work well)
                 # Actually, shape mismatch on Embedding will error out even with strict=False usually unless filtered.
                 # Ideally we should ignore mismatch keys.
                 state_dict = torch.load(bdh_path, map_location=self.device, weights_only=False)
                 if state_dict['embed.weight'].shape[0] != vocab_size:
                     print("⚠️  Vocab size mismatch (Old model). Starting fresh.")
                 else:
                     self.bdh.load_state_dict(state_dict, strict=False)
             except Exception as e:
                 print(f"⚠️  Weight load error: {e}")
        else:
             print(f"⚠️  BDH BASE NOT FOUND: {bdh_path}")

        # Import and initialize sentence analyzer
        from sentence_analyzer import SentenceAnalyzer
        self.sentence_analyzer = SentenceAnalyzer(self.bdh, EMBEDDING_DIM, self.device)

        # Better State Merging
        from .world_state import AdaptiveMerge
        self.adaptive_merge = AdaptiveMerge(EMBEDDING_DIM, num_heads=4).to(self.device).eval()

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
            # BPE Tokenization
            # tiktoken encodes to list of ints
            tokens_list = self.tokenizer.encode(text)
            
            # Truncate if too long (though BDH handles infinite, we batch here)
            if len(tokens_list) > MAX_SEQ_LEN:
                tokens_list = tokens_list[:MAX_SEQ_LEN]
            
            tokens = torch.tensor(tokens_list, dtype=torch.long, device=self.device)
                
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
                # BPE Fix
                ids = self.tokenizer.encode(chunk)
                tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
                if tokens.size(1) == 0: continue
                
                # Forward Pass with State Update
                self.bdh(tokens, use_state=True)
                
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
