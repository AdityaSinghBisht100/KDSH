"""
Inference Pipeline for BDH

Uses the trained BDH model to predict narrative consistency.

Two strategies:
1. Perplexity-based: Compare perplexity of statement with/without backstory
2. Classifier-based: Use the trained consistency classifier
"""
import os
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from bdh import BDH_GPU, BDHConfig, ByteTokenizer


def predict_with_perplexity(
    model: BDH_GPU,
    backstory: str,
    statement: str,
    tokenizer: ByteTokenizer,
    device: str = "cuda",
    threshold: float = 100.0
) -> Tuple[str, float, str]:
    """
    Predict consistency using perplexity.
    
    Idea: 
    - If the statement is consistent with the backstory, the model
      (having "read" the backstory) will find it unsurprising (low perplexity).
    - If it contradicts, the model will be surprised (high perplexity).
    
    Args:
        model: Pretrained BDH model
        backstory: Character backstory text
        statement: Statement to check
        tokenizer: Byte tokenizer
        device: Device to use
        threshold: Perplexity threshold for consistent/contradict
    
    Returns:
        (prediction, perplexity, rationale)
    """
    model.eval()
    
    with torch.no_grad():
        # Tokenize
        back_tokens = tokenizer.encode(backstory)[:4096]
        stmt_tokens = tokenizer.encode(statement)[:512]
        
        back_tensor = torch.tensor([back_tokens], dtype=torch.long, device=device)
        stmt_tensor = torch.tensor([stmt_tokens], dtype=torch.long, device=device)
        
        # Process backstory to get state
        _, state = model.forward(back_tensor)
        
        # Get perplexity of statement given backstory
        perplexity, _ = model.get_perplexity(stmt_tensor, state)
        ppl_value = perplexity.item()
        
        # Classify based on perplexity
        if ppl_value < threshold:
            prediction = "consistent"
        else:
            prediction = "contradict"
        
        rationale = f"Perplexity: {ppl_value:.2f}. {'Low (expected)' if ppl_value < threshold else 'High (surprising)'}."
    
    return prediction, ppl_value, rationale


def predict_with_classifier(
    model: BDH_GPU,
    classifier: nn.Module,
    backstory: str,
    statement: str,
    tokenizer: ByteTokenizer,
    device: str = "cuda"
) -> Tuple[str, float, str]:
    """
    Predict consistency using the trained classifier.
    
    Args:
        model: Pretrained BDH model
        classifier: Trained consistency classifier
        backstory: Character backstory text
        statement: Statement to check
        tokenizer: Byte tokenizer
        device: Device to use
    
    Returns:
        (prediction, confidence, rationale)
    """
    model.eval()
    classifier.eval()
    
    with torch.no_grad():
        # Tokenize
        back_tokens = tokenizer.encode(backstory)[:4096]
        stmt_tokens = tokenizer.encode(statement)[:512]
        
        back_tensor = torch.tensor([back_tokens], dtype=torch.long, device=device)
        stmt_tensor = torch.tensor([stmt_tokens], dtype=torch.long, device=device)
        
        # Process backstory to get state
        _, state = model.forward(back_tensor)
        
        # Get perplexity of statement
        perplexity, _ = model.get_perplexity(stmt_tensor, state)
        
        # Get state features
        state_rep = model.get_state_representation(state)
        if state_rep is not None:
            state_norm = state_rep.norm(dim=-1, keepdim=True).mean(dim=-1, keepdim=True)
        else:
            state_norm = torch.zeros(1, 1, device=device)
        
        # Features
        features = torch.cat([perplexity.view(1, 1), state_norm], dim=-1)
        
        # Classify
        logits = classifier(features)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()
        
        prediction = "consistent" if pred_idx == 1 else "contradict"
        rationale = f"Perplexity: {perplexity.item():.2f}. Confidence: {confidence:.2%}."
    
    return prediction, confidence, rationale


def get_backstory_from_novel(
    book_name: str,
    character: str,
    novel_dir: str,
    max_length: int = 4096,
    caption: str = ""
) -> str:
    """
    Extract backstory for a character from a novel.
    
    Uses BOTH character name AND caption keywords for better context retrieval.
    """
    import glob
    
    # Find novel file using fuzzy matching
    novel_path = None
    book_lower = book_name.lower().strip()
    
    # Get all txt files in the directory
    txt_files = glob.glob(os.path.join(novel_dir, "*.txt"))
    
    for txt_file in txt_files:
        filename = os.path.basename(txt_file).lower()
        # Check if book name is contained in filename (fuzzy match)
        if book_lower in filename or filename.replace('.txt', '') in book_lower:
            novel_path = txt_file
            break
    
    # Also try exact match
    if novel_path is None:
        for ext in ['.txt', '']:
            potential = os.path.join(novel_dir, book_name + ext)
            if os.path.exists(potential):
                novel_path = potential
                break
    
    if novel_path is None:
        return ""
    
    # Load novel
    with open(novel_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Extract keywords from caption for additional context
    caption_keywords = []
    if caption:
        # Split caption into words and filter out common words
        stop_words = {'the', 'a', 'an', 'of', 'in', 'and', 'or', 'to', 'for', 'with', 'on', 'at', 'by'}
        caption_keywords = [w.lower() for w in caption.split() if len(w) > 2 and w.lower() not in stop_words]
    
    char_lower = character.lower()
    relevant = []
    
    # Score and collect paragraphs
    for paragraph in text.split('\n\n'):
        para_lower = paragraph.lower()
        
        # Must contain character name
        if char_lower not in para_lower:
            continue
        
        # Score by keyword matches
        score = 1  # Base score for containing character
        for keyword in caption_keywords:
            if keyword in para_lower:
                score += 1
        
        relevant.append((score, paragraph.strip()))
    
    # Sort by relevance score (descending)
    relevant.sort(key=lambda x: x[0], reverse=True)
    
    # Take top paragraphs
    backstory_parts = [p[1] for p in relevant[:50]]  # Top 50 most relevant
    backstory = '\n'.join(backstory_parts)
    
    # Truncate to max_length characters
    if len(backstory) > max_length * 4:
        backstory = backstory[:max_length * 4]
    
    return backstory


def generate_predictions(
    model: BDH_GPU,
    classifier: Optional[nn.Module],
    test_df: pd.DataFrame,
    novel_dir: str,
    output_path: str,
    device: str = "cuda",
    use_classifier: bool = True,
    perplexity_threshold: float = 100.0
) -> pd.DataFrame:
    """
    Generate predictions for test data.
    
    Args:
        model: Pretrained BDH model
        classifier: Trained classifier (optional)
        test_df: Test DataFrame
        novel_dir: Directory containing novels
        output_path: Where to save predictions
        device: Device to use
        use_classifier: Whether to use classifier or raw perplexity
        perplexity_threshold: Threshold for perplexity-based classification
    
    Returns:
        DataFrame with predictions
    """
    print("\n=== Generating Predictions ===")
    
    model = model.to(device)
    model.eval()
    
    if classifier is not None:
        classifier = classifier.to(device)
        classifier.eval()
    
    tokenizer = ByteTokenizer()
    
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        row_id = row.get('id', 0)
        book_name = str(row.get('book_name', '')).strip()
        character = str(row.get('char', '')).strip()
        caption = str(row.get('caption', '')).strip()  # Get caption for context
        content = str(row.get('content', '')).strip()
        
        if not content:
            results.append({
                'id': row_id,
                'prediction': 1,
                'rationale': 'Empty content'
            })
            continue
        
        # Get backstory using character AND caption
        backstory = get_backstory_from_novel(book_name, character, novel_dir, caption=caption)
        
        if not backstory:
            # No backstory found, default to consistent
            results.append({
                'id': row_id,
                'prediction': 1,
                'rationale': 'No backstory found'
            })
            continue
        
        # Predict
        if use_classifier and classifier is not None:
            pred, conf, rationale = predict_with_classifier(
                model, classifier, backstory, content, tokenizer, device
            )
        else:
            pred, conf, rationale = predict_with_perplexity(
                model, backstory, content, tokenizer, device, perplexity_threshold
            )
        
        results.append({
            'id': row_id,
            'prediction': 1 if pred == 'consistent' else 0,
            'rationale': rationale
        })
    
    # Save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")
    
    return result_df
