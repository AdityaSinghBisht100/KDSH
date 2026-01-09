# Copyright 2025 Pathway Technology, Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: PyTorch Geometric not installed. Install with: pip install torch-geometric")


class TemporalEncoding(nn.Module):
    """Learnable temporal position encoding."""
    
    def __init__(self, d_model: int, max_positions: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        
        # Learnable temporal embeddings
        self.temporal_embedding = nn.Embedding(max_positions, d_model)
        
        # Sinusoidal encoding as initialization
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.temporal_embedding.weight.data.copy_(pe)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [N] tensor of temporal positions
        Returns:
            [N, d_model] temporal encodings
        """
        positions = torch.clamp(positions, 0, self.max_positions - 1)
        return self.temporal_embedding(positions)


class EdgeTypeEmbedding(nn.Module):
    """Embeddings for different edge types in narrative graph."""
    
    EDGE_TYPES = {
        'temporal_before': 0,
        'temporal_after': 1,
        'causal_causes': 2,
        'causal_enables': 3,
        'thematic_relates': 4,
        'character_involves': 5,
    }
    
    def __init__(self, d_model: int):
        super().__init__()
        self.num_types = len(self.EDGE_TYPES)
        self.edge_embedding = nn.Embedding(self.num_types, d_model)
    
    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_types: [E] tensor of edge type indices
        Returns:
            [E, d_model] edge type embeddings
        """
        return self.edge_embedding(edge_types)


class TemporalGNN(nn.Module):
    """Graph Attention Network for temporal relationship modeling."""
    
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric required for TemporalGNN")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(d_model)
        
        # Edge type embedding
        self.edge_type_embedding = EdgeTypeEmbedding(d_model)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    edge_dim=d_model,  # Edge features
                    add_self_loops=True,
                    concat=True,
                )
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        temporal_positions: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, d_model] node feature embeddings
            edge_index: [2, E] edge connectivity
            edge_types: [E] edge type indices
            temporal_positions: [N] temporal position of each node
            batch: [N] batch assignment for each node
        Returns:
            [N, d_model] updated node features
        """
        # Add temporal encoding if provided
        x = node_features
        if temporal_positions is not None:
            temporal_enc = self.temporal_encoding(temporal_positions)
            x = x + temporal_enc
        
        # Get edge features
        edge_attr = self.edge_type_embedding(edge_types)
        
        # Apply GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_prev = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = norm(x + x_prev)  # Residual connection
            x = F.relu(x)
        
        return x


class CausalGNN(nn.Module):
    """Message passing network for causal dependency tracking."""
    
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric required for CausalGNN")
        
        self.d_model = d_model
        
        # Edge type embedding
        self.edge_type_embedding = EdgeTypeEmbedding(d_model)
        
        # Causal attention layers (directional)
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    edge_dim=d_model,
                    add_self_loops=False,  # No self-loops for causal
                    concat=True,
                )
            )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Causal strength predictor
        self.causal_strength = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [N, d_model]
            edge_index: [2, E] directed edges (source -> target)
            edge_types: [E] edge type indices
        Returns:
            node_features: [N, d_model] updated features
            causal_scores: [E] causal strength for each edge
        """
        x = node_features
        edge_attr = self.edge_type_embedding(edge_types)
        
        # Apply causal GNN layers
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            x_prev = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = norm(x + x_prev)
            x = F.relu(x)
        
        # Compute causal strength for each edge
        source_features = x[edge_index[0]]  # [E, d_model]
        target_features = x[edge_index[1]]  # [E, d_model]
        edge_features = torch.cat([source_features, target_features], dim=-1)  # [E, 2*d_model]
        causal_scores = self.causal_strength(edge_features).squeeze(-1)  # [E]
        
        return x, causal_scores


class NarrativeGraph:
    """Dynamic graph structure for story elements."""
    
    def __init__(self, d_model: int, device: torch.device):
        self.d_model = d_model
        self.device = device
        
        # Node storage
        self.node_features: List[torch.Tensor] = []
        self.node_types: List[str] = []  # 'event', 'character', 'location', 'theme'
        self.node_positions: List[int] = []  # Temporal position
        self.node_metadata: List[Dict] = []  # Additional info
        
        # Edge storage
        self.edge_index: List[Tuple[int, int]] = []
        self.edge_types: List[int] = []
        self.edge_metadata: List[Dict] = []
    
    def add_node(
        self,
        feature: torch.Tensor,
        node_type: str,
        position: int,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a node to the graph. Returns node index."""
        node_idx = len(self.node_features)
        self.node_features.append(feature.to(self.device))
        self.node_types.append(node_type)
        self.node_positions.append(position)
        self.node_metadata.append(metadata or {})
        return node_idx
    
    def add_edge(
        self,
        source_idx: int,
        target_idx: int,
        edge_type: str,
        metadata: Optional[Dict] = None
    ):
        """Add an edge to the graph."""
        edge_type_idx = EdgeTypeEmbedding.EDGE_TYPES.get(edge_type, 0)
        self.edge_index.append((source_idx, target_idx))
        self.edge_types.append(edge_type_idx)
        self.edge_metadata.append(metadata or {})
    
    def to_torch_geometric(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric required")
        
        if len(self.node_features) == 0:
            # Empty graph
            return Data(
                x=torch.zeros((1, self.d_model), device=self.device),
                edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                edge_attr=torch.zeros((0,), dtype=torch.long, device=self.device),
            )
        
        # Stack node features
        x = torch.stack(self.node_features)  # [N, d_model]
        
        # Create edge index tensor
        if len(self.edge_index) > 0:
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=self.device).t()  # [2, E]
            edge_attr = torch.tensor(self.edge_types, dtype=torch.long, device=self.device)  # [E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0,), dtype=torch.long, device=self.device)
        
        # Temporal positions
        pos = torch.tensor(self.node_positions, dtype=torch.long, device=self.device)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
        )
    
    def get_subgraph(self, node_indices: List[int]) -> 'NarrativeGraph':
        """Extract subgraph containing specified nodes."""
        subgraph = NarrativeGraph(self.d_model, self.device)
        
        # Map old indices to new indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Add nodes
        for old_idx in node_indices:
            subgraph.add_node(
                self.node_features[old_idx],
                self.node_types[old_idx],
                self.node_positions[old_idx],
                self.node_metadata[old_idx]
            )
        
        # Add edges that connect nodes in subgraph
        for i, (src, tgt) in enumerate(self.edge_index):
            if src in index_map and tgt in index_map:
                new_src = index_map[src]
                new_tgt = index_map[tgt]
                edge_type_name = [k for k, v in EdgeTypeEmbedding.EDGE_TYPES.items() if v == self.edge_types[i]][0]
                subgraph.add_edge(new_src, new_tgt, edge_type_name, self.edge_metadata[i])
        
        return subgraph


class TemporalCausalGNN(nn.Module):
    """Combined temporal and causal GNN for narrative understanding."""
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Temporal reasoning module
        self.temporal_gnn = TemporalGNN(d_model, n_heads=n_heads, n_layers=3, dropout=dropout)
        
        # Causal reasoning module
        self.causal_gnn = CausalGNN(d_model, n_heads=n_heads, n_layers=2, dropout=dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_data: PyTorch Geometric Data object
        Returns:
            node_features: [N, d_model] updated node features
            causal_scores: [E] causal strength for edges
        """
        # Temporal reasoning
        temporal_features = self.temporal_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_types=graph_data.edge_attr,
            temporal_positions=graph_data.pos if hasattr(graph_data, 'pos') else None,
        )
        
        # Causal reasoning
        causal_features, causal_scores = self.causal_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_types=graph_data.edge_attr,
        )
        
        # Fuse temporal and causal features
        fused_features = self.fusion(
            torch.cat([temporal_features, causal_features], dim=-1)
        )
        
        return fused_features, causal_scores
