
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge
from .consistency import ContrastiveEnergyLoss, CounterfactualChecker

def train(system, epochs=5):
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
    num_batches = (len(train_df) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Energy Training", unit="batch", leave=False)
    
    # Shuffle training data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    for start_idx in range(0, len(train_df), batch_size):
        batch = train_df.iloc[start_idx : start_idx + batch_size]
        
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=system.device)
        valid_samples = 0
        
        for _, row in batch.iterrows():
            book = row['book_name'].strip()
            char = row['char'].strip()
            content = row['content']
            label = row['label'].strip().lower()
            is_contradict = (label == 'contradict')
            
            key = (book, char)
            if key in system.backstory_states:
                # Use the learned world state
                world_state = system.backstory_states[key]
            else:
                # Fallback to empty state
                system.bdh.reset_state()
                world_state = system.bdh.get_state()
            
            # 1. Compute Surprise(Statement)
            # training=True enables gradients
            pos_surprise = checker.compute_surprise(content, world_state, training=True)
            
            # 2. Compute Surprise(Negation)
            negated = checker.negate(content)
            neg_surprise = checker.compute_surprise(negated, world_state, training=True)
            
            # 3. Compute Energy Loss
            # Pass float values wrapped in tensors managed by compute_surprise
            # Check consistency.py: it returns tensor if training=True
            loss = loss_fn(pos_surprise, neg_surprise, is_contradict)
            
            batch_loss += loss
            valid_samples += 1
            
        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            
            # Gradient clipping to prevent exploding gradients in recurrent state
            torch.nn.utils.clip_grad_norm_(system.bdh.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += batch_loss.item()
            
        pbar.update(1)
        pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})
        
    pbar.close()
    return total_loss / num_batches

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
