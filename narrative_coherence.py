# Copyright 2025 Pathway Technology, Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from data_utils import NarrativeDocument, ConstraintExtractor
from temporal_causal_gnn import NarrativeGraph, TemporalCausalGNN


@dataclass
class BackstoryConstraints:
    """Extracted constraints from backstory."""
    character_constraints: Dict[str, Dict]
    temporal_constraints: List[Tuple[int, int, str]]
    causal_patterns: List[Tuple[int, int]]
    entity_embeddings: Dict[str, torch.Tensor]
    sentence_embeddings: torch.Tensor
    

class ThematicCoherenceModule(nn.Module):
    """Encodes and understands backstory themes, constraints, and narrative elements."""
    
    def __init__(self, bdh_model, d_model: int, device: torch.device):
        super().__init__()
        self.bdh_model = bdh_model
        self.d_model = d_model
        self.device = device
        
        # Attention pooling for aggregating sentence embeddings
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Query vector for attention pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Thematic vector projection
        self.thematic_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def encode_text_chunks(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text using BDH model.
        Args:
            text_tokens: [batch_size, seq_len] byte tokens
        Returns:
            [batch_size, seq_len, d_model] embeddings
        """
        # Get BDH embeddings (use intermediate representations)
        with torch.no_grad():
            # Use BDH's embedding layer
            embeddings = self.bdh_model.embed(text_tokens)  # [B, T, d_model]
        
        return embeddings
    
    def aggregate_sentence_embeddings(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple sentence embeddings into global thematic vector.
        Args:
            sentence_embeddings: [num_sentences, d_model]
        Returns:
            [d_model] global thematic vector
        """
        # Add batch dimension
        sent_emb = sentence_embeddings.unsqueeze(0)  # [1, num_sentences, d_model]
        
        # Use attention pooling with global query
        batch_size = sent_emb.size(0)
        query = self.global_query.expand(batch_size, -1, -1)  # [1, 1, d_model]
        
        thematic_vec, _ = self.attention_pooling(
            query=query,
            key=sent_emb,
            value=sent_emb
        )  # [1, 1, d_model]
        
        thematic_vec = thematic_vec.squeeze(0).squeeze(0)  # [d_model]
        
        # Project to thematic space
        thematic_vec = self.thematic_projection(thematic_vec)
        
        return thematic_vec
    
    def encode_entities(self, entity_sentence_embeddings: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Encode each entity based on sentences it appears in.
        Args:
            entity_sentence_embeddings: {entity_name: [list of sentence embeddings]}
        Returns:
            {entity_name: entity_embedding}
        """
        entity_embeddings = {}
        
        for entity_name, sent_embs in entity_sentence_embeddings.items():
            if len(sent_embs) == 0:
                continue
            
            # Average embeddings of sentences mentioning this entity
            # Ensure all tensors are on the correct device
            sent_embs_device = [emb.to(self.device) for emb in sent_embs]
            stacked = torch.stack(sent_embs_device)  # [num_mentions, d_model]
            entity_emb = torch.mean(stacked, dim=0)  # [d_model]
            
            # Encode with entity encoder (ensure encoder is on device)
            entity_emb = self.entity_encoder(entity_emb.to(self.device))
            
            entity_embeddings[entity_name] = entity_emb
        
        return entity_embeddings
    
    def forward(self, backstory_document: NarrativeDocument, max_chunk_size: int = 512) -> BackstoryConstraints:
        """
        Process backstory and extract constraints.
        Args:
            backstory_document: Parsed backstory document
            max_chunk_size: Maximum tokens per chunk
        Returns:
            BackstoryConstraints object
        """
        # Encode all sentences
        sentence_embeddings = []
        
        for sentence in backstory_document.sentences:
            # Tokenize sentence
            tokens = torch.tensor(sentence.tokens[:max_chunk_size], dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Encode with BDH
            emb = self.encode_text_chunks(tokens)  # [1, seq_len, d_model]
            
            # Mean pool over sequence length
            sent_emb = torch.mean(emb, dim=1).squeeze(0)  # [d_model]
            sentence_embeddings.append(sent_emb)
        
        sentence_embeddings = torch.stack(sentence_embeddings)  # [num_sentences, d_model]
        
        # Extract constraints
        character_constraints = ConstraintExtractor.extract_character_constraints(backstory_document)
        temporal_constraints = ConstraintExtractor.extract_temporal_constraints(backstory_document)
        causal_patterns = ConstraintExtractor.extract_causal_patterns(backstory_document)
        
        # Encode entities
        entity_sentence_embeddings = {}
        for entity_name, entity in backstory_document.entities.items():
            entity_sent_embs = [sentence_embeddings[idx] for idx in entity.mentions]
            entity_sentence_embeddings[entity_name] = entity_sent_embs
        
        entity_embeddings = self.encode_entities(entity_sentence_embeddings)
        
        return BackstoryConstraints(
            character_constraints=character_constraints,
            temporal_constraints=temporal_constraints,
            causal_patterns=causal_patterns,
            entity_embeddings=entity_embeddings,
            sentence_embeddings=sentence_embeddings
        )


class ConstraintTracker:
    """Tracks and manages constraints extracted from backstory."""
    
    def __init__(self, backstory_constraints: BackstoryConstraints, device: torch.device):
        self.constraints = backstory_constraints
        self.device = device
        
        # Index entities for fast lookup
        self.entity_index = {
            name: idx for idx, name in enumerate(backstory_constraints.entity_embeddings.keys())
        }
    
    def get_entity_embedding(self, entity_name: str) -> Optional[torch.Tensor]:
        """Get embedding for a specific entity."""
        return self.constraints.entity_embeddings.get(entity_name)
    
    def get_relevant_constraints(self, current_entities: List[str]) -> Dict:
        """Get constraints relevant to entities in current story."""
        relevant = {
            'characters': {},
            'temporal': [],
            'causal': []
        }
        
        for entity in current_entities:
            if entity in self.constraints.character_constraints:
                relevant['characters'][entity] = self.constraints.character_constraints[entity]
        
        return relevant
    
    def check_character_consistency(self, entity_name: str, current_context: str) -> float:
        """
        Check if character usage is consistent with backstory.
        Returns consistency score [0, 1].
        """
        if entity_name not in self.constraints.character_constraints:
            return 0.5  # Unknown entity, neutral score
        
        # Simple heuristic: if entity mentioned in backstory, consistent
        return 1.0


class KnowledgeGraphBuilder:
    """Builds graph representation of backstory elements."""
    
    def __init__(self, d_model: int, device: torch.device):
        self.d_model = d_model
        self.device = device
    
    def build_from_backstory(
        self,
        backstory_document: NarrativeDocument,
        backstory_constraints: BackstoryConstraints
    ) -> NarrativeGraph:
        """Build knowledge graph from backstory."""
        graph = NarrativeGraph(self.d_model, self.device)
        
        # Add sentence nodes
        sentence_to_node = {}
        for i, sentence in enumerate(backstory_document.sentences):
            node_idx = graph.add_node(
                feature=backstory_constraints.sentence_embeddings[i],
                node_type='event',
                position=i,
                metadata={'text': sentence.text, 'entities': sentence.entities}
            )
            sentence_to_node[i] = node_idx
        
        # Add temporal edges
        for sent_idx1, sent_idx2, relation in backstory_constraints.temporal_constraints:
            if sent_idx1 in sentence_to_node and sent_idx2 in sentence_to_node:
                graph.add_edge(
                    sentence_to_node[sent_idx1],
                    sentence_to_node[sent_idx2],
                    f'temporal_{relation}',
                    metadata={'relation': relation}
                )
        
        # Add causal edges
        for sent_idx1, sent_idx2 in backstory_constraints.causal_patterns:
            if sent_idx1 in sentence_to_node and sent_idx2 in sentence_to_node:
                graph.add_edge(
                    sentence_to_node[sent_idx1],
                    sentence_to_node[sent_idx2],
                    'causal_causes',
                    metadata={'type': 'causal'}
                )
        
        # Add character nodes and edges
        for entity_name, entity_emb in backstory_constraints.entity_embeddings.items():
            char_node_idx = graph.add_node(
                feature=entity_emb,
                node_type='character',
                position=-1,  # Characters don't have temporal position
                metadata={'name': entity_name}
            )
            
            # Connect character to sentences they're mentioned in
            if entity_name in backstory_constraints.character_constraints:
                mentioned_in = backstory_constraints.character_constraints[entity_name]['mentioned_in']
                for sent_idx in mentioned_in:
                    if sent_idx in sentence_to_node:
                        graph.add_edge(
                            char_node_idx,
                            sentence_to_node[sent_idx],
                            'character_involves',
                            metadata={'character': entity_name}
                        )
        
        return graph


class NarrativeCoherenceSystem(nn.Module):
    """Main orchestrator integrating all modules for narrative coherence analysis."""
    
    def __init__(self, bdh_model, d_model: int = 256, device: torch.device = torch.device('cpu')):
        super().__init__()
        
        self.bdh_model = bdh_model
        self.d_model = d_model
        self.device = device
        
        # Thematic coherence module - move to device
        self.thematic_module = ThematicCoherenceModule(bdh_model, d_model, device).to(device)
        
        # Temporal-causal GNN - move to device
        try:
            self.temporal_causal_gnn = TemporalCausalGNN(d_model, n_heads=4, dropout=0.1).to(device)
            self.use_gnn = True
        except ImportError:
            self.temporal_causal_gnn = None
            self.use_gnn = False
        
        # Knowledge graph builder
        self.graph_builder = KnowledgeGraphBuilder(d_model, device)
        
        # Constraint tracker (will be set during backstory processing)
        self.constraint_tracker: Optional[ConstraintTracker] = None
        self.backstory_graph: Optional[NarrativeGraph] = None
    
    def process_backstory(self, backstory_document: NarrativeDocument):
        """Process and encode backstory, build knowledge graph."""
        # Extract constraints using thematic module
        backstory_constraints = self.thematic_module(backstory_document)
        
        # Initialize constraint tracker
        self.constraint_tracker = ConstraintTracker(backstory_constraints, self.device)
        
        # Build knowledge graph
        self.backstory_graph = self.graph_builder.build_from_backstory(
            backstory_document,
            backstory_constraints
        )
        
        # Process graph with GNN to get refined embeddings (if GNN available)
        if self.use_gnn and self.temporal_causal_gnn is not None:
            try:
                graph_data = self.backstory_graph.to_torch_geometric()
                updated_features, _ = self.temporal_causal_gnn(graph_data)
                
                # Update graph with refined features
                for i in range(len(self.backstory_graph.node_features)):
                    self.backstory_graph.node_features[i] = updated_features[i]
            except Exception as e:
                print(f"Warning: GNN processing skipped: {e}")
    
    def get_backstory_summary(self) -> Dict:
        """Get summary of processed backstory."""
        if self.constraint_tracker is None:
            return {}
        
        return {
            'num_characters': len(self.constraint_tracker.constraints.character_constraints),
            'num_temporal_constraints': len(self.constraint_tracker.constraints.temporal_constraints),
            'num_causal_patterns': len(self.constraint_tracker.constraints.causal_patterns),
            'num_sentences': len(self.constraint_tracker.constraints.sentence_embeddings),
        }
