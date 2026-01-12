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
            cached_data = torch.load(cache_path)
            system.world_states = cached_data['world_states']
            system.backstory_states = cached_data['backstory_states']
            print(f"  ✅ Loaded state for {len(system.world_states)} books. Jumping to Phase 2.")
            return
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {e}. Re-ingesting...")
    
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
    # Optimization: Larger chunk size for faster ingestion on GPU
    chunk_size = 2048 
    
    system.bdh.reset_state()
    initial_state = system.bdh.get_state() # Keep on Device
    
    global_state = initial_state.clone()
    entity_states = {e: initial_state.clone() for e in entities}
    
    entity_timestamps = {e: 0.0 for e in entities}
    global_timestamp = 0.0
    
    entity_update_counts = {e: 0 for e in entities}
    gate_values = {e: [] for e in entities} 
    entity_memories = {e: [] for e in entities} # New: Sentence Memory 
    
    state_dim = initial_state.numel()
    write_gate = EntityWriteGate(state_dim=64, proj_dim=32).to(system.device)
    write_gate.eval() 
    
    current_time = 0.0
    
    # Calculate total chunks for progress bar
    total_chunks = (len(text) + chunk_size - 1) // chunk_size
    
    with torch.no_grad():
        with tqdm(total=total_chunks, desc="  Processing Chunks (GPU)", unit="chunk", leave=False) as pbar:
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                tokens = torch.tensor([[ord(c) % 256 for c in chunk]], dtype=torch.long, device=system.device)
                if tokens.size(1) == 0:
                    pbar.update(1)
                    continue
                
                current_time += 1.0
                
                system.bdh.reset_state()
                system.bdh.set_state(global_state.detach().clone()) # On Device
                system.bdh(tokens, use_state=True)
                global_state = system.bdh.get_state() # Keep on Device
                global_timestamp = current_time
                
                chunk_lower = chunk.lower()
                mentioned_entities = [e for e in entities if e.lower() in chunk_lower]
                chunk_emb = global_state
                
                # New: Save sentence-level evidence
                if mentioned_entities:
                    # Simple sentence splitting
                    sentences = [s.strip() for s in chunk.replace('?', '.').replace('!', '.').split('.') if len(s.split()) > 4]
                    for sent in sentences:
                        sent_lower = sent.lower()
                        for entity in mentioned_entities:
                            if entity.lower() in sent_lower:
                                # Encode sentence as evidence (detached from graph)
                                with torch.no_grad():
                                    sent_tokens = torch.tensor([[ord(c) % 256 for c in sent]], dtype=torch.long, device=system.device)
                                    if sent_tokens.size(1) > 0:
                                        system.bdh.reset_state()
                                        e_seq = system.bdh(sent_tokens, use_state=False, return_embeddings=True)
                                        sent_vec = e_seq.mean(dim=1) # Keep on Device [1, D]
                                        
                                        if len(entity_memories[entity]) < 200:
                                            entity_memories[entity].append((sent_vec, sent))

                
                for entity in mentioned_entities:
                    old_state = entity_states[entity].detach().clone()
                    
                    system.bdh.reset_state()
                    system.bdh.set_state(old_state) # On Device
                    system.bdh(tokens, use_state=True)
                    new_state = system.bdh.get_state() # On Device
                    
                    gate = write_gate(
                        old_state,
                        chunk_emb, 
                        global_state
                    ).item()
                    
                    update = new_state - old_state
                    entity_states[entity] = old_state + gate * update
                    entity_timestamps[entity] = current_time
                    
                    entity_update_counts[entity] += 1
                    gate_values[entity].append(gate)
                
                pbar.update(1)
    
    # Move to CPU only at the very end for storage efficiency
    world_state = WorldState(
        global_state=global_state.detach().cpu(),
        entity_states={k: v.detach().cpu() for k, v in entity_states.items()},
        known_entities=entities
    )
    # Move memories to CPU for storage
    cpu_memories = {}
    for e, mems in entity_memories.items():
        cpu_memories[e] = [(v.detach().cpu(), t) for v, t in mems]
        
    world_state.entity_memories = cpu_memories
    world_state.entity_timestamps = entity_timestamps
    world_state.global_timestamp = global_timestamp
    
    return world_state
