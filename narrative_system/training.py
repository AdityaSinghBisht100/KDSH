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
    
    # Initialize Optimizer ONCE to preserve momentum across epochs
    optimizer = torch.optim.Adam(
        system.bdh.parameters(),
        lr=5e-5 # Lower learning rate for stability
    )
    
    for epoch in range(epochs):
            train_loss = run_training_step(system, train_df, loss_fn, optimizer)
            metrics = evaluate_accuracy(system, test_df)
            
            print(f"\n--- Epoch {epoch+1}/{epochs} Summary ---")
            print(f"Loss:      {train_loss:.4f}")
            print(f"Accuracy:  {metrics['accuracy']:.2%}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print("Confusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"  [TN: {cm['tn']}, FP: {cm['fp']}]")
            print(f"  [FN: {cm['fn']}, TP: {cm['tp']}]")
            
            # Save only BDH, as we have no classifier head
            torch.save(system.bdh.state_dict(), os.path.join(system.model_dir, "bdh_base.pt"))
            
    print("\n=== Training Complete ===")
    final_metrics = evaluate_accuracy(system, test_df)
    print(f"Final Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"Final F1-Score: {final_metrics['f1']:.4f}")

def run_training_step(system, train_df, loss_fn, optimizer, batch_size=4):
    system.bdh.train()
    
    # Optimizer is now passed in
    
    # Initialize consistency checker with BPE tokenizer
    
    # Initialize consistency checker with BPE tokenizer
    checker = CounterfactualChecker(system.bdh, system.tokenizer, system.device)
    
    total_loss = 0
    num_samples = len(train_df)
    pbar = tqdm(total=num_samples, desc="Energy Training", unit="sample", leave=False)
    
    # Shuffle training data
    # Robustness: ensure content/char/book are strings and NaN-safe
    train_df = train_df.fillna('').sample(frac=1).reset_index(drop=True)
    
    # Process ONE sample at a time to avoid OOM (gradient accumulation uses too much memory)
    for idx, row in train_df.iterrows():
        book = str(row['book_name']).strip()
        char = str(row['char']).strip()
        content = str(row['content']).strip()
        label = str(row['label']).strip().lower()
        
        if not content: continue # Skip empty content rows
        is_contradict = (label == 'contradict')
        
        key = (book, char)
        if key in system.backstory_states:
            world_state = system.backstory_states[key]
        else:
            system.bdh.reset_state()
            world_state = system.bdh.get_state()
        
        optimizer.zero_grad()
        
        # 1. Compute Energy with context
        energy_context = checker.compute_energy(content, world_state)
        
        # 2. Compute Energy WITHOUT context (Reset State)
        energy_base = checker.compute_energy(content, None)
        
        # 3. Compute Differential Energy Loss
        # We want: 
        # - Consistent: context_energy < base_energy (Negative Delta)
        # - Contradict: context_energy > base_energy (Positive Delta)
        
        if is_contradict:
            # PUSH: make context energy MUCH LARGER than base energy
            # Loss = clamp(margin + base - context, min=0)
            loss = torch.clamp(0.1 + energy_base - energy_context, min=0)
        else:
            # PULL: make context energy MUCH SMALLER than base energy
            # Loss = clamp(margin + context - base, min=0)
            loss = torch.clamp(0.1 + energy_context - energy_base, min=0)
        
        # Backward and step per sample to free memory immediately
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(system.bdh.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Free memory aggressively
        del energy_context, energy_base, loss
        if idx % 5 == 0:
            torch.cuda.empty_cache()
            
        pbar.update(1)
        pbar.set_postfix({'loss': f"{total_loss / (idx + 1):.4f}"})
        
    pbar.close()
    return total_loss / num_samples

def evaluate_accuracy(system, test_df):
    results = []
    labels = []
    
    system.bdh.eval()
    
    # Robustness: fill NaNs
    test_df = test_df.fillna('')
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            book = str(row['book_name']).strip()
            char = str(row['char']).strip()
            content = str(row['content']).strip()
            label = str(row['label']).strip().lower()
            
            if not content: continue # Skip malformed rows
            
            score, _ = system.predict_single(book, char, content)
            
            pred = "consistent" if score > 0.5 else "contradict"
            results.append(pred)
            labels.append(label)
            
    # Binary Classification Metrics (Positive = contradict)
    tp = tn = fp = fn = 0
    for p, l in zip(results, labels):
        if p == "contradict" and l == "contradict": tp += 1
        elif p == "consistent" and l == "consistent": tn += 1
        elif p == "contradict" and l == "consistent": fp += 1
        elif p == "consistent" and l == "contradict": fn += 1

    total = len(labels)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }
