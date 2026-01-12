
import os
import torch
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge

def predict_single(system, book_name, char_name, content):
    book_name = book_name.strip()
    char_name = char_name.strip()
    key = (book_name, char_name)
    
    if system.bdh is None: system._initialize_components()
    
    if key not in system.backstory_states:
            if book_name in system.world_states:
                ws = system.world_states[book_name]
                state = ws.get_query_state(char_name)
                system.backstory_states[key] = state.to(system.device)
            else:
                book_path = os.path.join(system.data_dir, f"{book_name}.txt")
                if not os.path.exists(book_path):
                    return 0.5, "Unknown book - Neutral confidence"
                
                with open(book_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                system.bdh.reset_state()
                system.absorb_story_stream(text[:10000])
                state = system.bdh.get_state()
                system.backstory_states[key] = state

    state = system.backstory_states[key].to(system.device)
    
    v_iso = system.encode_text([content], distinct_states=None)
    v_ctx = system.encode_text([content], distinct_states=state)
    
    with torch.no_grad():
            logits = system.classifier(v_iso, v_ctx)
            probs = torch.softmax(logits, dim=1)
            score = probs[0, 1].item()
            
            # Rationale Generation
            # Rationale Generation
            if score > 0.5:
                # CONSISTENT
                rationale = "High confidence consistency: Statement aligns strongly with world state."
            else:
                # CONTRADICTION - Attempt Retrieval
                rationale = "Strong contradiction: Statement directly opposes established world state."
                
                # Retrieval Logic
                try:
                    if book_name in system.world_states:
                        ws = system.world_states[book_name]
                        if hasattr(ws, 'entity_memories') and char_name in ws.entity_memories:
                             memories = ws.entity_memories[char_name]
                             if memories:
                                 # Compute cosine similarity
                                 # memories is list of (vec, text)
                                 mem_vecs = torch.stack([m[0] for m in memories]).to(system.device) # [N, D]
                                 
                                 # Re-encode statement cleanly (detached)
                                 system.bdh.reset_state()
                                 stmt_tokens = torch.tensor([[ord(c) % 256 for c in content]], dtype=torch.long, device=system.device)
                                 e_seq = system.bdh(stmt_tokens, use_state=False, return_embeddings=True)
                                 stmt_vec = e_seq.mean(dim=1) # [1, D]
                                 
                                 sims = torch.nn.functional.cosine_similarity(stmt_vec, mem_vecs)
                                 best_idx = torch.argmax(sims).item()
                                 best_score = sims[best_idx].item()
                                 
                                 evidence_text = memories[best_idx][1]
                                 
                                 # If similarity is decent, use it
                                 if best_score > 0.4:
                                     rationale = f"Contradiction based on evidence: '{evidence_text}'"
                except Exception as e:
                    print(f"Retrieval failed: {e}")

            return score, rationale

def generate_predictions(system, input_file="test.csv", output_file="predictions.csv"):
    """Run batch inference on a CSV and save binary predictions with rationale."""
    print(f"\n=== Generating Predictions for {input_file} ===")
    system._initialize_components()
    
    train_path = os.path.join(system.data_dir, "train.csv")
    dummy_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame({'book_name': [], 'char': []})
    
    # Ensure knowledge is loaded
    ingest_novel_knowledge(system, dummy_df, test_mode=False)
    
    input_path = os.path.join(system.data_dir, input_file)
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples.")
    
    results = []
    system.bdh.eval()
    system.classifier.eval()
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting", unit="sample"):
            book = row.get('book_name', '')
            char = row.get('char', '')
            content = row.get('content', '')
            
            # Note: calling predict_single from this module, passing system
            score, rationale = predict_single(system, book, char, content)
            pred_binary = 1 if score > 0.5 else 0
            
            results.append({
                'id': row.get('id', _),
                'prediction': pred_binary,
                'rationale': rationale
            })
    
    # OUTPUT PATH FIX: Use CWD if not specified otherwise
    # If explicit output_file is provided, use it directly (relative to CWD)
    output_path = output_file
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    print(result_df.head())

def interactive_session(system):
    """Run an interactive CLI session to query the model."""
    print("\n=== KDSH Interactive Query Mode ===")
    print("Type 'exit' or 'quit' to stop.\n")
    system._initialize_components()
    
    while True:
        try:
            print("-" * 40)
            book = input("Book Name (e.g. 'The Count of Monte Cristo'): ").strip()
            if book.lower() in ['exit', 'quit']: break
            
            char = input("Character Name (e.g. 'Edmond Dantes'): ").strip()
            if char.lower() in ['exit', 'quit']: break
            
            stmt = input("Statement to check: ").strip()
            if stmt.lower() in ['exit', 'quit']: break
            
            if not book or not char or not stmt:
                print("Error: All fields are required.")
                continue
            
            print(f"\nAnalyzing consistency for {char} in '{book}'...")
            score, rationale = predict_single(system, book, char, stmt)
            
            pred = "Consistent" if score > 0.5 else "Contradict"
            confidence = score if score > 0.5 else 1 - score
            
            print(f"\nResult: {pred.upper()}")
            print(f"Confidence (Score): {confidence:.2%} ({score:.4f})")
            print(f"Rationale: {rationale}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    print("\nGoodbye!")
