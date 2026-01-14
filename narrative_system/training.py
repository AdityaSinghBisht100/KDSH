
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge

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
    # Call the ingestion logic (which is now in system wrapper but delegates to ingestion.py)
    # Or call directly:
    ingest_novel_knowledge(system, combined, test_mode=False)
    
    print("\n=== Phase 2: Supervised Training (train.csv) ===")
    
    # Pre-training Validation (Immediate Feedback)
    print("Running initial validation on test set...")
    initial_acc = evaluate_accuracy(system, test_df)
    print(f"Initial Accuracy (Ingestion Only): {initial_acc:.2%}")
    
    print("Training model to detect consistency (Hybrid Mode)...")
    for epoch in range(epochs):
            loss = run_training_step(system, train_df)
            acc = evaluate_accuracy(system, test_df)
            print(f"Epoch {epoch+1}/{epochs}: Loss={loss:.4f} Acc={acc:.2%}")
            
            torch.save(system.classifier.state_dict(), os.path.join(system.model_dir, "narrative_consistency.pt"))
            if system.hybrid_classifier:
                 torch.save(system.hybrid_classifier.state_dict(), os.path.join(system.model_dir, "hybrid_consistency.pt"))
            torch.save(system.bdh.state_dict(), os.path.join(system.model_dir, "bdh_base.pt"))
            
    print("\n=== Training Complete ===")
    final_acc = evaluate_accuracy(system, test_df)
    print(f"Final Model Accuracy on Dataset: {final_acc:.2%}")
    if not os.path.exists(os.path.join(system.data_dir, "test.csv")):
            print("(Note: Evaluated on train.csv since test.csv was not found)")

def run_training_step(system, train_df, batch_size=4):
    system.classifier.train()
    system.hybrid_classifier.train()
    system.bdh.train()
    total_loss = 0
    
    optimizer = torch.optim.Adam(
        list(system.classifier.parameters()) + 
        list(system.hybrid_classifier.parameters()) + 
        list(system.bdh.parameters()),
        lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    num_batches = (len(train_df) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Training Hybrid Batches", unit="batch", leave=False)
    
    for start_idx in range(0, len(train_df), batch_size):
        batch = train_df.iloc[start_idx : start_idx + batch_size]
        if len(batch) < 1: 
            pbar.update(1)
            continue
        
        contents = batch['content'].tolist()
        negated_contents = [system.counterfactual_checker.negate(c) for c in contents]
        
        labels = torch.tensor(
            [1 if str(l).strip().lower() == 'consistent' else 0 for l in batch['label']],
            device=system.device
        )
        
        states = []
        for _, row in batch.iterrows():
            key = (str(row['book_name']).strip(), str(row['char']).strip())
            if key in system.backstory_states:
                states.append(system.backstory_states[key].to(system.device))
            else:
                system.bdh.reset_state()
                states.append(system.bdh.get_state())

        # Forward passes
        v_iso = system.encode_text(contents, distinct_states=None)
        v_ctx = system.encode_text(contents, distinct_states=states)
        v_neg_ctx = system.encode_text(negated_contents, distinct_states=states)
        
        # Surprise Energy Features (Differentiable proxies)
        surprise_s = torch.norm(v_ctx - v_iso, p=2, dim=1)
        # For negation surprise, we can use a similar proxy 
        # (v_neg_ctx vs v_iso_neg, but let's approximate or use v_neg_ctx vs v_iso)
        v_iso_neg = system.encode_text(negated_contents, distinct_states=None)
        surprise_neg = torch.norm(v_neg_ctx - v_iso_neg, p=2, dim=1)
        
        surprise_ratio = surprise_s / (surprise_neg + 1e-8)
        
        optimizer.zero_grad()
        
        # 1. Base Classifier Loss
        logits_base = system.classifier(v_iso, v_ctx)
        loss_base = criterion(logits_base, labels)
        
        # 2. Hybrid Classifier Loss
        # v_ctx and v_neg_ctx are used to see which one "fits" the backstory better
        logits_hybrid = system.hybrid_classifier(v_iso, v_ctx, v_neg_ctx, surprise_ratio)
        loss_hybrid = criterion(logits_hybrid, labels)
        
        # Combined Loss
        loss = loss_base + loss_hybrid
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    pbar.close()
    return total_loss / max(1, num_batches)

def evaluate_accuracy(system, test_df):
    correct = 0
    total = 0
    system.bdh.eval()
    system.classifier.eval()
    
    if hasattr(system, 'hybrid_classifier') and system.hybrid_classifier is not None:
            system.hybrid_classifier.eval()
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            book = row['book_name']
            char = row['char']
            content = row['content']
            label = row['label'].strip().lower()
            
            score, _ = system.predict_single(book, char, content)
            
            pred = "consistent" if score > 0.5 else "contradict"
            if pred == label:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0
