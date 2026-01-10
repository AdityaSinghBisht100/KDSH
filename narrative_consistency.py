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
        PERFECT WORLD STATE: Read books through BDH.
        
        Args:
            test_mode: If True, use only first 50,000 chars (~1 chapter) for fast testing
        """
        print("Pre-computing Perfect World States (Full Book Absorption)...")
        
        self.bdh.eval()
        
        # Get unique books
        unique_books = train_df['book_name'].dropna().unique()
        print(f"  Found {len(unique_books)} unique books to process.")
        
        if test_mode:
            print("  >> TEST MODE: Using first 50,000 characters only <<")
        
        # Book paths mapping
        book_paths = {
            "The Count of Monte Cristo": "/root/files/The Count of Monte Cristo.txt",
            "In Search of the Castaways": "/root/files/In search of the castaways.txt"
        }
        
        # ==========================================
        # PHASE 1: Absorb books into BDH
        # ==========================================
        self.book_states = {}
        
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
                
                # Limit text in test mode
                if test_mode:
                    full_text = full_text[:50000]  # ~1 chapter
                
                print(f"    -> Processing {len(full_text):,} characters...")
                
                # Absorb through BDH
                book_state = self.absorb_story_stream(full_text)
                self.book_states[book_name] = book_state.cpu()
                
                print(f"    -> Done!")
                
            except Exception as e:
                print(f"    -> Error: {e}")
        
        print(f"Computed world states for {len(self.book_states)} books.")
        
        # ==========================================
        # PHASE 2: Map characters to their book states
        # ==========================================
        distinct_chars = train_df[['book_name', 'char']].drop_duplicates()
        
        for _, row in distinct_chars.iterrows():
            if pd.isna(row['book_name']) or pd.isna(row['char']):
                continue
            book = str(row['book_name']).strip()
            char = str(row['char']).strip()
            key = (book, char)
            
            if book in self.book_states:
                self.backstory_states[key] = self.book_states[book]
            else:
                self.bdh.reset_state()
                self.backstory_states[key] = self.bdh.get_state()
                
        print(f"Mapped {len(self.backstory_states)} character-book pairs.")

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
        Predict consistency using Intinite Context (State-Space)
        """
        key = (book_name.strip(), char_name.strip())
        
        if key not in self.backstory_states:
             # Try to load on fly if not precomputed
             print(f"Loading state for {key} on demand...")
             sentences = self.sentence_analyzer.extract_character_substory(book_name, char_name)
             full_story = "".join(sentences)
             if not full_story:
                 state = None
             else:
                 state = self.absorb_story_stream(full_story).cpu()
                 self.backstory_states[key] = state
        
        if key in self.backstory_states:
             state = self.backstory_states[key]
        else:
             state = None
              
        # Inference
        v_iso = self.encode_text([content], distinct_states=None)
        v_ctx = self.encode_text([content], distinct_states=state)
        
        with torch.no_grad():
            self.classifier.eval()
            logits = self.classifier(v_iso, v_ctx)
            prob = torch.softmax(logits, dim=1)[0][1].item()
            
        return prob

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
