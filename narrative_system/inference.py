import os
import torch
import pandas as pd
from tqdm import tqdm
from .ingestion import ingest_novel_knowledge
from .consistency import CounterfactualChecker

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
                    return 0.5, "Unknown context"
                
                with open(book_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                system.bdh.reset_state()
                system.absorb_story_stream(text[:10000])
                state = system.bdh.get_state()
                system.backstory_states[key] = state

    state = system.backstory_states[key] # List of states
    
    # Check linkage using research-standard PMI
    label, pmi = system.checker.check_linkage(content, state)
    
    score = 0.9 if label == "consistent" else 0.1
    rationale = f"PMI Score: {pmi:.4f}. "
    
    if score > 0.5:
        rationale += f"Backstory strongly supports this statement."
    else:
        rationale += f"Backstory significantly reduces the likelihood of this statement."
        
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
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting", unit="sample"):
            book = str(row.get('book_name', '')).strip() if pd.notna(row.get('book_name')) else ''
            char = str(row.get('char', '')).strip() if pd.notna(row.get('char')) else ''
            content = str(row.get('content', '')).strip() if pd.notna(row.get('content')) else ''
            
            if book == 'nan': book = ''
            if char == 'nan': char = ''
            if content == 'nan': content = ''
            
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
