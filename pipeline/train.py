"""
Training Pipeline for BDH

Two-phase training:
1. Self-Supervised Pretraining: Train BDH on novel text (next-byte prediction)
   This teaches the model to "understand" and "remember" the story.
   
2. Consistency Fine-tuning: Fine-tune on labeled consistency pairs.
   This teaches the model to classify statements as consistent/contradictory.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from bdh import BDH_GPU, BDHConfig, ByteTokenizer


class NovelDataset(Dataset):
    """
    Dataset for self-supervised pretraining on novel text.
    
    Chunks the novel into fixed-size sequences for training.
    """
    def __init__(
        self,
        novel_paths: List[str],
        tokenizer: ByteTokenizer,
        seq_len: int = 2048,
        overlap: int = 256
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.overlap = overlap
        self.chunks = []
        
        # Load and chunk all novels
        for path in novel_paths:
            self._process_novel(path)
    
    def _process_novel(self, path: str):
        """Load a novel and create overlapping chunks."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Tokenize entire novel
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Create overlapping chunks
        stride = self.seq_len - self.overlap
        for i in range(0, len(tokens) - self.seq_len, stride):
            chunk = tokens[i:i + self.seq_len]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]


class ConsistencyDataset(Dataset):
    """
    Dataset for consistency classification.
    
    Each sample contains:
    - backstory: The accumulated character context (as bytes)
    - statement: The statement to check (as bytes)
    - label: 1 if consistent, 0 if contradiction
    """
    def __init__(
        self,
        df: pd.DataFrame,
        novel_dir: str,
        tokenizer: ByteTokenizer,
        max_backstory_len: int = 4096,
        max_statement_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_backstory_len = max_backstory_len
        self.max_statement_len = max_statement_len
        self.samples = []
        
        # Cache novels
        self.novel_cache = {}
        
        for _, row in df.iterrows():
            book_name = str(row.get('book_name', '')).strip()
            char = str(row.get('char', '')).strip()
            content = str(row.get('content', '')).strip()
            label_str = str(row.get('label', '')).strip().lower()
            
            if not content:
                continue
            
            # Get label
            label = 1 if label_str == 'consistent' else 0
            
            # Get backstory (simplified: use book context around character mentions)
            backstory = self._get_backstory(book_name, char, novel_dir)
            
            self.samples.append({
                'backstory': backstory,
                'statement': content,
                'label': label,
                'book': book_name,
                'char': char
            })
    
    def _get_backstory(self, book_name: str, char: str, novel_dir: str) -> str:
        """
        Extract backstory for a character from a novel.
        
        Simple approach: Find sentences mentioning the character.
        """
        # Find novel file
        novel_path = None
        for ext in ['.txt', '']:
            potential = os.path.join(novel_dir, book_name + ext)
            if os.path.exists(potential):
                novel_path = potential
                break
        
        if novel_path is None:
            return ""
        
        # Load novel (cached)
        if novel_path not in self.novel_cache:
            with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
                self.novel_cache[novel_path] = f.read()
        
        text = self.novel_cache[novel_path]
        
        # Extract sentences mentioning character
        char_lower = char.lower()
        sentences = []
        for sentence in text.split('.'):
            if char_lower in sentence.lower():
                sentences.append(sentence.strip() + '.')
        
        # Combine (limited length)
        backstory = ' '.join(sentences)
        return backstory[:self.max_backstory_len * 4]  # Rough char limit
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
        backstory_tokens = self.tokenizer.encode(sample['backstory'])[:self.max_backstory_len]
        statement_tokens = self.tokenizer.encode(sample['statement'])[:self.max_statement_len]
        
        return {
            'backstory': torch.tensor(backstory_tokens, dtype=torch.long),
            'statement': torch.tensor(statement_tokens, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


def collate_consistency(batch: List[Dict]) -> Dict:
    """Collate function for consistency dataset with padding."""
    max_back_len = max(len(b['backstory']) for b in batch)
    max_stmt_len = max(len(b['statement']) for b in batch)
    
    backstories = []
    statements = []
    labels = []
    
    for b in batch:
        # Pad backstory
        back = b['backstory']
        pad_len = max_back_len - len(back)
        if pad_len > 0:
            back = torch.cat([back, torch.zeros(pad_len, dtype=torch.long)])
        backstories.append(back)
        
        # Pad statement
        stmt = b['statement']
        pad_len = max_stmt_len - len(stmt)
        if pad_len > 0:
            stmt = torch.cat([stmt, torch.zeros(pad_len, dtype=torch.long)])
        statements.append(stmt)
        
        labels.append(b['label'])
    
    return {
        'backstory': torch.stack(backstories),
        'statement': torch.stack(statements),
        'label': torch.stack(labels)
    }


def pretrain_on_novels(
    model: BDH_GPU,
    novel_paths: List[str],
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> BDH_GPU:
    """
    Phase 1: Self-supervised pretraining on novel text.
    
    The model learns to predict the next byte given history.
    This forces it to "read" and "remember" the story in its state.
    """
    print("\n=== Phase 1: Self-Supervised Pretraining ===")
    print(f"Training on {len(novel_paths)} novels")
    
    model = model.to(device)
    tokenizer = ByteTokenizer()
    
    # Create dataset
    dataset = NovelDataset(novel_paths, tokenizer, seq_len=2048, overlap=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created {len(dataset)} training chunks")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, _ = model.get_loss(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
    
    # Save checkpoint
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved pretrained model to {save_path}")
    
    return model


def train_consistency_classifier(
    model: BDH_GPU,
    train_df: pd.DataFrame,
    novel_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> Tuple[BDH_GPU, nn.Module]:
    """
    Phase 2: Train a consistency classifier on top of the pretrained BDH.
    
    Strategy:
    1. Process backstory through BDH to get state σ
    2. Process statement through BDH (continuing from σ)
    3. Use perplexity difference as signal:
       - Low perplexity (expected) → Consistent
       - High perplexity (surprising) → Contradiction
    """
    print("\n=== Phase 2: Consistency Classifier Training ===")
    
    model = model.to(device)
    model.eval()  # Freeze BDH backbone during classifier training
    
    tokenizer = ByteTokenizer()
    
    # Create dataset
    dataset = ConsistencyDataset(train_df, novel_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_consistency)
    
    print(f"Training on {len(dataset)} samples")
    
    # Simple classifier: takes perplexity as input
    # Input: perplexity_of_statement (scalar)
    # Output: probability of consistent
    classifier = nn.Sequential(
        nn.Linear(2, 32),  # Input: [perplexity, mean_state_norm]
        nn.ReLU(),
        nn.Linear(32, 2)   # Output: [logit_contradict, logit_consistent]
    ).to(device)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            backstory = batch['backstory'].to(device)
            statement = batch['statement'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Process backstory to get state
                _, state = model.forward(backstory)
                
                # Get per-sample perplexity of statement given backstory context
                perplexity, _ = model.get_perplexity(statement, state, per_sample=True)  # [batch]
                
                # Get state representation
                state_rep = model.get_state_representation(state)
                if state_rep is not None:
                    state_norm = state_rep.norm(dim=-1)  # [batch]
                else:
                    state_norm = torch.zeros(backstory.shape[0], device=device)
            
            # Features: [perplexity, state_norm] - both are [batch]
            features = torch.stack([perplexity, state_norm], dim=-1)  # [batch, 2]
            
            # Classify
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.2%}'})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
    
    # Save
    if save_path:
        torch.save({
            'model': model.state_dict(),
            'classifier': classifier.state_dict()
        }, save_path)
        print(f"Saved to {save_path}")
    
    return model, classifier


def contrastive_finetune(
    model: BDH_GPU,
    train_df: pd.DataFrame,
    novel_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-5,
    margin: float = 1.0,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> BDH_GPU:
    """
    Phase 3: Contrastive fine-tuning on labeled pairs.
    
    Key insight: 
    - For CONSISTENT statements: perplexity with backstory should be LOWER
    - For CONTRADICTORY statements: perplexity with backstory should be HIGHER
    
    We use a margin-based loss to push these apart.
    """
    import glob
    
    print("\n=== Phase 3: Contrastive Fine-tuning ===")
    
    model = model.to(device)
    model.train()
    
    tokenizer = ByteTokenizer()
    
    # Prepare samples
    samples = []
    novel_cache = {}
    
    for _, row in train_df.iterrows():
        book_name = str(row.get('book_name', '')).strip()
        char = str(row.get('char', '')).strip()
        caption = str(row.get('caption', '')).strip()
        content = str(row.get('content', '')).strip()
        label_str = str(row.get('label', '')).strip().lower()
        
        if not content or not label_str:
            continue
        
        label = 1 if label_str == 'consistent' else 0
        
        # Get backstory (find the novel file)
        book_lower = book_name.lower().strip()
        txt_files = glob.glob(os.path.join(novel_dir, "*.txt"))
        novel_path = None
        
        for txt_file in txt_files:
            filename = os.path.basename(txt_file).lower()
            if book_lower in filename or filename.replace('.txt', '') in book_lower:
                novel_path = txt_file
                break
        
        if novel_path is None:
            continue
        
        # Load novel (cached)
        if novel_path not in novel_cache:
            with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
                novel_cache[novel_path] = f.read()
        
        text = novel_cache[novel_path]
        
        # Extract backstory paragraphs mentioning character
        char_lower = char.lower()
        relevant = []
        for para in text.split('\n\n'):
            if char_lower in para.lower():
                relevant.append(para.strip())
        
        backstory = '\n'.join(relevant[:30])[:8192]  # Limit size
        
        if not backstory:
            continue
        
        samples.append({
            'backstory': backstory,
            'statement': content,
            'label': label
        })
    
    print(f"Prepared {len(samples)} samples for contrastive training")
    
    # Optimizer (fine-tune with small LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle samples
        import random
        random.shuffle(samples)
        
        pbar = tqdm(range(0, len(samples), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i in pbar:
            batch = samples[i:i + batch_size]
            if len(batch) == 0:
                continue
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for sample in batch:
                backstory = sample['backstory']
                statement = sample['statement']
                label = sample['label']  # 1 = consistent, 0 = contradict
                
                # Tokenize
                back_tokens = tokenizer.encode(backstory)[:2048]
                stmt_tokens = tokenizer.encode(statement)[:256]
                
                back_tensor = torch.tensor([back_tokens], dtype=torch.long, device=device)
                stmt_tensor = torch.tensor([stmt_tokens], dtype=torch.long, device=device)
                
                # Get perplexity WITH backstory context
                _, state = model.forward(back_tensor)
                ppl_with, _ = model.get_perplexity(stmt_tensor, state, per_sample=True)
                
                # Get perplexity WITHOUT backstory (fresh state)
                ppl_without, _ = model.get_perplexity(stmt_tensor, None, per_sample=True)
                
                # Contrastive loss
                # diff = ppl_with - ppl_without
                # For consistent: we want ppl_with < ppl_without → diff < 0 → minimize diff
                # For contradict: we want ppl_with > ppl_without → diff > 0 → maximize diff
                diff = ppl_with - ppl_without
                
                if label == 1:  # Consistent
                    # Loss = max(0, diff + margin)  → push diff below -margin
                    loss = F.relu(diff + margin).mean()
                else:  # Contradict
                    # Loss = max(0, -diff + margin) → push diff above margin
                    loss = F.relu(-diff + margin).mean()
                
                batch_loss = batch_loss + loss
                
                # Track accuracy
                predicted = 1 if diff.item() < 0 else 0
                if predicted == label:
                    correct += 1
                total += 1
            
            # Average over batch
            batch_loss = batch_loss / len(batch)
            
            # Backward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'acc': f'{correct/max(1,total):.2%}'})
        
        avg_loss = total_loss / max(1, len(samples) // batch_size)
        accuracy = correct / max(1, total)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
    
    # Save
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved contrastive-finetuned model to {save_path}")
    
    return model
