import os
import torch
import pandas as pd
from tqdm import tqdm
from .world_state import WorldState, EntityWriteGate

def ingest_novel_knowledge(system, train_df, test_mode=True):
    """
    PHASE 1: FEEEDING KNOWLEDGE.
    Reads full novels to populate the Entity-Aware World State (Memory).
    NO TRAINING occurs here (weights are not updated).
    """
    print("\n=== Phase 1: Feeding Knowledge (Ingesting Novels) ===")
    
    cache_path = os.path.join(system.model_dir, "world_state_cache.pt")
    if os.path.exists(cache_path) and not test_mode:
        print(f"  ✨ Found cached World State at {cache_path}")
        print(f"  ✨ Loading knowledge from disk (skipping raw text ingestion)...")
        try:
            cached_data = torch.load(cache_path, weights_only=False)
            # Validation: Check first state shape
            test_key = list(cached_data['backstory_states'].keys())[0]
            test_state = cached_data['backstory_states'][test_key][0]
            if test_state.shape[-1] != system.config.n_embd // system.config.n_head:
                 raise ValueError("Cache shape mismatch (Different Architecture)")
                 
            system.world_states = cached_data['world_states']
            system.backstory_states = cached_data['backstory_states']
            print(f"  ✅ Loaded state for {len(system.world_states)} books. Jumping to Phase 2.")
            return
        except Exception as e:
            print(f"  ⚠️ Cache Incompatible: {e}. Re-ingesting...")
    
    system.bdh.eval()
    
    # Get unique books
    unique_books = train_df['book_name'].dropna().unique()
    print(f"Found {len(unique_books)} unique books to ingest.")
    
    if test_mode:
        print("  >> TEST MODE: Using first 50,000 characters only <<")
    
    # Book paths mapping
    book_paths = {
        "The Count of Monte Cristo": os.path.join(system.data_dir, "The Count of Monte Cristo.txt"),
        "In Search of the Castaways": os.path.join(system.data_dir, "In search of the castaways.txt")
    }
    
    # Process each book
    for book_name in unique_books:
        book_name = str(book_name).strip()
        # Try specific map first, then generic in data_dir
        book_path = book_paths.get(book_name)
        
        if not book_path or not os.path.exists(book_path):
                # Try direct filename match
                candidate = os.path.join(system.data_dir, f"{book_name}.txt")
                if os.path.exists(candidate):
                    book_path = candidate
        
        if not book_path or not os.path.exists(book_path):
            print(f"  -> Warning: Novel file not found for '{book_name}' in {system.data_dir}")
            continue
            
        try:
            print(f"  📖 Feeding knowledge from '{book_name}'...")
            with open(book_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            if test_mode:
                full_text = full_text[:50000]
            
            print(f"    -> Ingesting {len(full_text):,} chars into Memory...")
            
            book_entities = set(
                train_df[train_df['book_name'].str.strip() == book_name]['char']
                .dropna().str.strip().unique()
            )
            
            world_state = _entity_aware_ingest(system, full_text, book_entities)
            system.world_states[book_name] = world_state
            print(f"    -> Done! Knowledge State updated for {len(book_entities)} characters.")
            
        except Exception as e:
            print(f"    -> Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Knowledge Ingestion Complete for {len(system.world_states)} books.")
    
    distinct_chars = train_df[['book_name', 'char']].drop_duplicates()
    
    for _, row in distinct_chars.iterrows():
        if pd.isna(row['book_name']) or pd.isna(row['char']):
            continue
        book = str(row['book_name']).strip()
        char = str(row['char']).strip()
        key = (book, char)
        
        if book in system.world_states:
            system.backstory_states[key] = system.world_states[book].get_query_state(char, alpha=0.3)
        else:
            system.bdh.reset_state()
            system.backstory_states[key] = system.bdh.get_state()
            
    print(f"Mapped {len(system.backstory_states)} character-book pairs with merged states.")
    
    # SAVE CACHE
    if not test_mode:
        print("  💾 Saving World State cache to disk...")
        torch.save({
            'world_states': system.world_states,
            'backstory_states': system.backstory_states
        }, cache_path)
        print("  ✅ Cache saved.")

def _entity_aware_ingest(system, text: str, entities: set) -> WorldState:
    system.bdh.reset_state()
    
    # Helper to capture Current state as list of layers
    def get_full_state(model):
        return [block.attn.state.clone().cpu() for block in model.transformer.h]

    global_state = get_full_state(system.bdh)
    entity_states = {e: [s.clone() for s in global_state] for e in entities}
    entity_memories = {e: [] for e in entities} 
    
    # 1. Semantic Chunking (Sentence Level)
    import re
    chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 10] # Filter very short noise
    
    batch_size = 16
    total_chunks = len(chunks)
    
    with torch.no_grad():
        with tqdm(total=total_chunks, desc="  Semantic Ingestion", unit="sent", leave=False) as pbar:
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                
                # 2. Convert Batch to Semantic Vectors [B, 384]
                vecs_np = system.encoder.encode(batch, convert_to_numpy=True)
                # vecs: [B, 1, 384]
                vecs = torch.from_numpy(vecs_np).to(system.device).unsqueeze(1)
                
                # 3. Process Chunk-by-Chunk for State Updates
                for idx, chunk_text in enumerate(batch):
                    v = vecs[idx:idx+1] # [1, 1, 384]
                    
                    # Update Global State
                    system.bdh.reset_state()
                    for layer_idx, block in enumerate(system.bdh.transformer.h):
                        block.attn.state.copy_(global_state[layer_idx].to(system.device))
                    
                    system.bdh(v, use_state=True)
                    global_state = get_full_state(system.bdh)
                    
                    # Update Mentioned Entities
                    chunk_lower = chunk_text.lower()
                    mentioned_entities = [e for e in entities if e.lower() in chunk_lower]
                    
                    for entity in mentioned_entities:
                        system.bdh.reset_state()
                        for layer_idx, block in enumerate(system.bdh.transformer.h):
                            block.attn.state.copy_(entity_states[entity][layer_idx].to(system.device))
                        
                        system.bdh(v, use_state=True)
                        entity_states[entity] = get_full_state(system.bdh)
                        
                pbar.update(len(batch))
    
    world_state = WorldState(
        global_state=global_state,
        entity_states=entity_states,
        known_entities=entities
    )
    world_state.entity_memories = entity_memories
    return world_state
