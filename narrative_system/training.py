import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge
from .consistency import ContrastiveEnergyLoss, CounterfactualChecker

def train(system, epochs=20):
    print(f"Starting Training on {system.device}...")
    system._initialize_components()
    
    train_path = os.path.join(system.data_dir, "train.csv")
    test_path = os.path.join(system.data_dir, "test.csv")
    
    if not os.path.exists(train_path):
            print(f"Train file not found at {train_path}")
            return
            
    train_df = pd.read_csv(train_path)
    if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
    else:
            test_df = train_df.copy() # Mock
            
    combined = pd.concat([train_df, test_df])
    ingest_novel_knowledge(system, combined, test_mode=False)
    
    print("\n=== Phase 2: Supervised Training (train.csv) ===")
    print("Training BDH to minimize energy for consistent statements...")
    
    # Initialize implementation of Energy Loss
    # Margin 0.3 means we want Gap > 0.3
    loss_fn = ContrastiveEnergyLoss(margin=0.5).to(system.device)
    
    for epoch in range(epochs):
            loss = run_training_step(system, train_df, loss_fn)
            acc = evaluate_accuracy(system, test_df)
            print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f} Acc={acc:.2%}")
            
            # Save only BDH, as we have no classifier head
            torch.save(system.bdh.state_dict(), os.path.join(system.model_dir, "bdh_base.pt"))
            
    print("\n=== Training Complete ===")
    final_acc = evaluate_accuracy(system, test_df)
    print(f"Final Model Accuracy on Dataset: {final_acc:.2%}")

def run_training_step(system, train_df, loss_fn, batch_size=4):
    system.bdh.train()
    
    optimizer = torch.optim.Adam(
        system.bdh.parameters(),
        lr=5e-5 # Lower learning rate for stability
    )
    
    # Initialize consistency checker with BPE tokenizer
    checker = CounterfactualChecker(system.bdh, system.tokenizer, system.device)
    
    total_loss = 0
    num_samples = len(train_df)
    pbar = tqdm(total=num_samples, desc="Energy Training", unit="sample", leave=False)
    
    # Shuffle training data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    # Process ONE sample at a time to avoid OOM (gradient accumulation uses too much memory)
    for idx, row in train_df.iterrows():
        book = row['book_name'].strip()
        char = row['char'].strip()
        content = row['content']
        label = row['label'].strip().lower()
        is_contradict = (label == 'contradict')
        
        key = (book, char)
        if key in system.backstory_states:
            world_state = system.backstory_states[key]
        else:
            system.bdh.reset_state()
            world_state = system.bdh.get_state()
        
        optimizer.zero_grad()
        
        # 1. Compute Surprise(Statement) with gradient
        pos_surprise = checker.compute_surprise(content, world_state, training=True)
        
        # 2. Compute Surprise(Negation) with gradient
        negated = checker.negate(content)
        neg_surprise = checker.compute_surprise(negated, world_state, training=True)
        
        # 3. Compute Energy Loss
        loss = loss_fn(pos_surprise, neg_surprise, is_contradict)
        
        # Backward and step per sample to free memory immediately
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(system.bdh.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Free memory aggressively
        del pos_surprise, neg_surprise, loss
        if idx % 10 == 0:
            torch.cuda.empty_cache()
            
        pbar.update(1)
        pbar.set_postfix({'loss': f"{total_loss / (idx + 1):.4f}"})
        
    pbar.close()
    return total_loss / num_samples

def evaluate_accuracy(system, test_df):
    correct = 0
    total = 0
    system.bdh.eval()
    
    # Use inference logic
    with torch.no_grad():
        for _, row in test_df.iterrows():
            book = row['book_name']
            char = row['char']
            content = row['content']
            label = row['label'].strip().lower()
            
            # predict_single uses CounterfactualChecker internally
            score, _ = system.predict_single(book, char, content)
            
            pred = "consistent" if score > 0.5 else "contradict"
            if pred == label:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0
