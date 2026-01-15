"""
Novel Ingestion Pipeline

Reads novels and builds character memory states using semantic embeddings.
"""
import os
import torch
from typing import Set, Dict
from tqdm import tqdm
import pandas as pd

def ingest_novels(
    embedder,
    memory,
    data_dir: str,
    train_df: pd.DataFrame,
    chunk_size: int = 500,
    device: str = "cuda"
):
    """
    Ingest novels into character memory.
    
    Args:
        embedder: SemanticEmbedder instance
        memory: CharacterMemory instance
        data_dir: Path to data directory
        train_df: DataFrame with book_name and char columns
        chunk_size: Characters per chunk
        device: Device to use
    """
    print("\n=== Phase 1: Ingesting Novels into Memory ===")
    
    # Get unique books and their entities
    book_entities: Dict[str, Set[str]] = {}
    for _, row in train_df.iterrows():
        book = str(row.get('book_name', '')).strip()
        char = str(row.get('char', '')).strip()
        if book and char:
            if book not in book_entities:
                book_entities[book] = set()
            book_entities[book].add(char)
    
    # Book path mapping
    book_paths = {
        "The Count of Monte Cristo": os.path.join(data_dir, "The Count of Monte Cristo.txt"),
        "In Search of the Castaways": os.path.join(data_dir, "In search of the castaways.txt")
    }
    
    print(f"Found {len(book_entities)} books with {sum(len(e) for e in book_entities.values())} characters")
    
    for book_name, entities in book_entities.items():
        book_path = book_paths.get(book_name) or os.path.join(data_dir, f"{book_name}.txt")
        
        if not os.path.exists(book_path):
            print(f"⚠️ Skipping '{book_name}' - file not found")
            continue
        
        print(f"\n📖 Ingesting '{book_name}'...")
        
        # Register entities
        memory.register_entities(book_name, entities)
        
        # Read novel
        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"   {len(text):,} characters, {len(entities)} characters to track")
        
        # Process in chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk_text in tqdm(chunks, desc=f"   Processing", leave=False):
            # Embed the chunk and clone to avoid inference mode issues
            chunk_embedding = embedder.encode(chunk_text).clone().detach()
            
            # Update global state
            memory.update_global(book_name, chunk_embedding.squeeze())
            
            # Update mentioned entities
            chunk_lower = chunk_text.lower()
            for entity in entities:
                if entity.lower() in chunk_lower:
                    memory.update_entity(book_name, entity, chunk_embedding.squeeze())
        
        print(f"   ✅ Done! Memory built for {book_name}")
    
    print(f"\n✅ Ingestion complete! {len(memory.world_states)} books in memory")
    return memory
