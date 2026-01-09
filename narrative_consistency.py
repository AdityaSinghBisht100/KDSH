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
        "torch>=2.0.0",
        "numpy",
        "pandas",
        "tqdm",
        "torch-geometric>=2.3.0"
    )
    .add_local_dir(".", remote_path="/root")
)

# Define Volume
vol = modal.Volume.from_name("narrative-models", create_if_missing=True)

# Constants
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256

# --- Model Definitions ---

try:
    from torch_geometric.data import Data, Batch
    from temporal_causal_gnn import TemporalCausalGNN
    HAS_GNN = True
except ImportError:
    HAS_GNN = False
    print("Warning: Torch Geometric not found, GNN features disabled.")

class CoherenceClassifier(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        self.use_gnn = HAS_GNN
        
        if self.use_gnn:
            self.gnn = TemporalCausalGNN(d_model=input_dim, n_heads=4, dropout=0.1)
            
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, content_emb, context_emb):
        # GNN Enhancement
        if self.use_gnn:
            batch_size = content_emb.size(0)
            data_list = []
            edge_type = 4 # thematic_relates
            
            for i in range(batch_size):
                # 2 nodes: Content (0), Context (1)
                x = torch.stack([content_emb[i], context_emb[i]]) # [2, D]
                
                # Directed Edge: Context (1) -> Content (0)
                # This allows Content to attend to Context, but keeps Context pure (as the 'reference')
                # Prevents over-smoothing in a 2-node graph
                edge_index = torch.tensor([[1], [0]], dtype=torch.long, device=self.device)
                edge_attr = torch.tensor([edge_type], dtype=torch.long, device=self.device)
                
                # Create naive "pos"
                pos = torch.tensor([0, 0], dtype=torch.long, device=self.device)
                
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos))
            
            batch = Batch.from_data_list(data_list)
            
            # Run GNN
            out_nodes, _ = self.gnn(batch) # [2*B, D]
            out_nodes = out_nodes.view(batch_size, 2, -1)
            
            # Update embeddings with graph context
            content_emb = out_nodes[:, 0, :]
            context_emb = out_nodes[:, 1, :]
            
        features = torch.cat([
            content_emb, 
            context_emb, 
            torch.abs(content_emb - context_emb),
            content_emb * context_emb
        ], dim=1)
        return self.net(features)

class NarrativeDataset(Dataset):
    def __init__(self, dataframe, backstory_store):
        self.data = dataframe
        self.backstory_store = backstory_store
        self.samples = []
        
        # Pre-compute contexts to avoid overhead in training loop if possible
        # or just store text.
        for idx, row in dataframe.iterrows():
            key = (row['book_name'].strip(), row['char'].strip())
            content = row['content']
            label = 1 if row['label'] == 'consistent' else 0
            
            # Retrieve relevant context
            context_text = self.retrieve_context(key, content)
            
            self.samples.append({
                'content_text': content,
                'context_text': context_text,
                'label': label
            })
            
    def retrieve_context(self, key, content):
        if key not in self.backstory_store:
            return ""
        candidates = self.backstory_store[key]
        if not candidates:
            return ""
        
        query_words = set(re.findall(r'\w+', content.lower()))
        scored_candidates = []
        for sent in candidates:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            score = len(query_words.intersection(sent_words))
            scored_candidates.append((score, sent))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_context = [s[1] for s in scored_candidates[:3]]
        return " ".join(top_context)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# --- Main System Logic ---

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
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
        self.backstory_store = {}
        print("--> Initialization complete")

    def encode_text(self, text_list):
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
            
        with torch.no_grad():
            emb = self.bdh.embed(padded)
            sent_emb = emb.mean(dim=1)
        return sent_emb

    @modal.method()
    def train_and_evaluate(self):
        self._initialize_components()
        
        from data_utils import TextLoader, NarrativeDocumentBuilder
        
        # 1. Load Books
        print("\n=== Loading Books ===")
        books = {
            "The Count of Monte Cristo": "/root/files/The Count of Monte Cristo.txt",
            "In Search of the Castaways": "/root/files/In search of the castaways.txt"
        }
        
        # Pre-scan train.csv for characters
        df = pd.read_csv("/root/files/train.csv")
        chars_map = {}
        for book in df['book_name'].unique():
            chars_map[book] = df[df['book_name']==book]['char'].unique().tolist()
        
        builder = NarrativeDocumentBuilder()
        for book_name, file_path in books.items():
            print(f"Processing {book_name}...")
            if not os.path.exists(file_path):
                print(f"Error: File not found {file_path}")
                continue
                
            text = TextLoader.load_text(file_path)
            doc = builder.build(text)
            
            chars_to_track = chars_map.get(book_name, [])
            for char in chars_to_track:
                relevant_sentences = []
                aliases = [a.strip() for a in char.split('/')]
                for sent in doc.sentences:
                    if any(alias in sent.text for alias in aliases):
                        relevant_sentences.append(sent.text)
                
                key = (book_name.strip(), char.strip())
                self.backstory_store[key] = relevant_sentences
                print(f"  Indexed {len(relevant_sentences)} sentences for {char}")

        # 2. Train
        print("\n=== Training Classifier ===")
        dataset = NarrativeDataset(df, self.backstory_store)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=self._collate) # Batch size 8 for small memory/dataset
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.classifier.train()
        epochs = 30
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch in dataloader:
                content_txt = batch['content_text']
                context_txt = batch['context_text']
                labels = batch['label'].to(self.device)
                
                content_emb = self.encode_text(content_txt)
                context_emb = self.encode_text(context_txt)
                
                optimizer.zero_grad()
                outputs = self.classifier(content_emb, context_emb)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {correct/total:.2%}")

        # Save model
        torch.save(self.classifier.state_dict(), "/models/narrative_consistency.pt")
        torch.save(self.bdh.state_dict(), "/models/bdh_base.pt")
        vol.commit()
        print("\nSaved models to /models/narrative_consistency.pt and /models/bdh_base.pt")

        # 3. Test on a few examples
        print("\n=== Verification Results ===")
        self.classifier.eval()
        results = []
        
        # Pick 3 random examples
        test_samples = df.sample(3)
        for _, row in test_samples.iterrows():
            pred, conf, rationale = self.predict_single(row['book_name'], row['char'], row['content'])
            results.append({
                "Character": row['char'],
                "Content": row['content'][:100] + "...",
                "True Label": row['label'],
                "Predicted": pred,
                "Rationale": rationale
            })
            
        return results

    def predict_single(self, book, char, content):
        key = (book.strip(), char.strip())
        base_sentences = self.backstory_store.get(key, [])
        query_words = set(re.findall(r'\w+', content.lower()))
        
        scored = []
        if base_sentences:
            for s in base_sentences:
                s_words = set(re.findall(r'\w+', s.lower()))
                score = len(query_words.intersection(s_words))
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_context = [s[1] for s in scored[:3]]
            context_text = " ".join(top_context)
        else:
            top_context = []
            context_text = ""
            
        content_emb = self.encode_text([content])
        context_emb = self.encode_text([context_text])
        
        with torch.no_grad():
            outputs = self.classifier(content_emb, context_emb)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs).item()
            conf = probs[0][pred_idx].item()
            
        label = "consistent" if pred_idx == 1 else "contradict"
        
        if not top_context:
            rat = "No backstory found."
        else:
            rat = f"Based on: '{top_context[0][:150]}...'"
            
        return label, conf, rat

    def _collate(self, batch):
        return {
            'content_text': [item['content_text'] for item in batch],
            'context_text': [item['context_text'] for item in batch],
            'label': torch.tensor([item['label'] for item in batch])
        }

@app.local_entrypoint()
def main():
    print("Starting Consistency Classifier Training on Modal...")
    system = NarrativeConsistencySystem()
    results = system.train_and_evaluate.remote()
    
    import json
    with open("consistency_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for res in results:
        print(f"\nCharacter: {res['Character']}")
        print(f"Content:   {res['Content']}")
        print(f"Result:    {res['True Label']} vs {res['Predicted']}")
        print(f"Rationale: {res['Rationale']}")
    print("="*80)
