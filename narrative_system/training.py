import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge

def train(system, epochs=20):
    from .system import set_seed
    set_seed(42)
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
    print("Training BDH with Multi-Task (Causal + Semantic) Objective...")
    
    # Initialize Optimizer ONCE to preserve momentum across epochs
    optimizer = torch.optim.Adam(
        system.bdh.parameters(),
        lr=5e-5 # Lower learning rate for stability
    )
    
    for epoch in range(epochs):
            train_loss = run_training_step(system, train_df, optimizer)
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
            
            # Save unified architecture weights
            save_path = os.path.join(system.model_dir, "bdh_transformer.pt")
            torch.save(system.bdh.state_dict(), save_path)
            print(f"  💾 Model checkpoint saved to {save_path}")
            
    print("\n=== Training Complete ===")
    final_metrics = evaluate_accuracy(system, test_df)
    print(f"Final Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"Final F1-Score: {final_metrics['f1']:.4f}")

def run_training_step(system, train_df, optimizer):
    system.bdh.train()
    
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
        
        if not content: continue 
        is_contradict = (label == 'contradict')
        
        key = (book, char)
        backstory_state = system.backstory_states.get(key)
        
        optimizer.zero_grad()
        
        # 1. Get Base Semantic Embedding [1, 384]
        vec_np = system.encoder.encode([content], convert_to_numpy=True)
        vec = torch.from_numpy(vec_np).to(system.device).unsqueeze(1) # [1, 1, 384]
        
        # 2. Refined Forward Pass with Memory
        if backstory_state:
             system.bdh.set_state(backstory_state)
        else:
             system.bdh.reset_state()
             
        # Label: 1.0 for consistent, 0.0 for contradict
        y_label = torch.tensor([[1.0 if not is_contradict else 0.0]], device=system.device)
        
        # forward returns (soul_vec, logit, loss)
        _, _, loss = system.bdh(vec, targets=y_label, use_state=True)
        
        # 3. Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.bdh.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Cleanup
        del logits, loss, loss_causal, loss_semantic
        if idx % 10 == 0:
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
