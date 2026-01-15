"""
Character Memory and World State

Manages:
- Global book state (overall narrative context)
- Per-entity states (character-specific memories)
- Merging entity states for queries
"""
import torch
from typing import Dict, Set, List, Optional
from .state_space import BDHStateSpace

class WorldState:
    """
    Holds all memory for a book.
    """
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.global_state: Optional[torch.Tensor] = None  # Overall book context
        self.entity_states: Dict[str, torch.Tensor] = {}  # Per-character states
        self.known_entities: Set[str] = set()
    
    def add_entity(self, entity_name: str, initial_state: Optional[torch.Tensor] = None):
        """Register a character to track."""
        self.known_entities.add(entity_name)
        if initial_state is not None:
            self.entity_states[entity_name] = initial_state
        else:
            self.entity_states[entity_name] = torch.zeros(self.embedding_dim)
    
    def update_entity(self, entity_name: str, state: torch.Tensor):
        """Update a character's state."""
        self.entity_states[entity_name] = state
    
    def get_entity_state(self, entity_name: str) -> Optional[torch.Tensor]:
        """Get a character's memory state."""
        return self.entity_states.get(entity_name)
    
    def get_merged_state(self, entity_name: str, global_weight: float = 0.3) -> torch.Tensor:
        """
        Get character state merged with global context.
        
        merged = (1 - α) * entity_state + α * global_state
        """
        entity_state = self.entity_states.get(entity_name)
        
        if entity_state is None:
            return self.global_state if self.global_state is not None else torch.zeros(self.embedding_dim)
        
        if self.global_state is None:
            return entity_state
        
        return (1 - global_weight) * entity_state + global_weight * self.global_state


class CharacterMemory:
    """
    Manages memory for all books and characters.
    """
    def __init__(self, embedding_dim: int = 384, n_heads: int = 4, device: str = "cuda"):
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.device = device
        
        # Per-book world states
        self.world_states: Dict[str, WorldState] = {}
        
        # BDH state machines (reused for processing) - MOVE TO DEVICE
        self.global_bdh = BDHStateSpace(embedding_dim, n_heads).to(device)
        self.entity_bdh = BDHStateSpace(embedding_dim, n_heads).to(device)
    
    def get_or_create_world(self, book_name: str) -> WorldState:
        """Get or create a world state for a book."""
        if book_name not in self.world_states:
            self.world_states[book_name] = WorldState(self.embedding_dim)
        return self.world_states[book_name]
    
    def register_entities(self, book_name: str, entities: Set[str]):
        """Register entities to track for a book."""
        world = self.get_or_create_world(book_name)
        for entity in entities:
            world.add_entity(entity)
    
    def update_global(self, book_name: str, embedding: torch.Tensor):
        """Update global state with new chunk embedding."""
        world = self.get_or_create_world(book_name)
        
        # Shape: [n_heads, head_dim] for BDH state buffer
        head_dim = self.embedding_dim // self.n_heads
        if world.global_state is not None:
            initial_state = world.global_state.view(self.n_heads, head_dim).to(self.device)
        else:
            initial_state = torch.zeros(self.n_heads, head_dim, device=self.device)
        
        self.global_bdh.set_state(initial_state)
        with torch.no_grad():
            new_state = self.global_bdh(embedding.to(self.device))
        world.global_state = new_state.flatten().detach().cpu()  # Store flat on CPU
    
    def update_entity(self, book_name: str, entity_name: str, embedding: torch.Tensor):
        """Update a specific entity's state."""
        world = self.get_or_create_world(book_name)
        
        current = world.get_entity_state(entity_name)
        if current is not None:
            self.entity_bdh.set_state(current.view(self.n_heads, -1).to(self.device))
        else:
            self.entity_bdh.reset_state()
        
        with torch.no_grad():
            new_state = self.entity_bdh(embedding.to(self.device))
        world.update_entity(entity_name, new_state.flatten().detach().cpu())  # Store on CPU
    
    def get_backstory_embedding(self, book_name: str, entity_name: str) -> Optional[torch.Tensor]:
        """Get the backstory embedding for a character."""
        if book_name not in self.world_states:
            return None
        return self.world_states[book_name].get_merged_state(entity_name)
    
    def save(self, path: str):
        """Save all memory to disk."""
        torch.save({
            'world_states': {
                name: {
                    'global': ws.global_state,
                    'entities': ws.entity_states,
                    'known': ws.known_entities
                }
                for name, ws in self.world_states.items()
            }
        }, path)
    
    def load(self, path: str):
        """Load memory from disk."""
        data = torch.load(path, weights_only=False)
        for name, ws_data in data['world_states'].items():
            ws = self.get_or_create_world(name)
            ws.global_state = ws_data['global']
            ws.entity_states = ws_data['entities']
            ws.known_entities = ws_data['known']
