import torch
import networkx as nx
from typing import List, Dict, Any, Optional

class BDHStateManager:
    def __init__(self, config):
        self.config = config
        self.character_states = {} # Map char_name -> BDH state (v_as)
        self.constraint_networks = {} # Map char_name -> nx.Graph
        self.event_graphs = {} # Map char_name -> nx.DiGraph
        self.coherence_scores = {} # Map char_name -> float

    def process_narrative(self, novel_name: str, char_name: str, context_embeddings: List[torch.Tensor], constraints: List[Dict[str, Any]]):
        """
        Updates character states and networks based on the processed narrative.
        """
        if char_name not in self.character_states:
            self.character_states[char_name] = []
            self.constraint_networks[char_name] = nx.Graph()
            self.event_graphs[char_name] = nx.DiGraph()
            
        # Store embeddings (as history for now, or could be a running hidden state)
        self.character_states[char_name].extend(context_embeddings)
        
        # Build constraint network
        net = self.constraint_networks[char_name]
        for i, c in enumerate(constraints):
            node_id = f"{char_name}_c_{len(net)}"
            net.add_node(node_id, **c)
            # Simple heuristic: connect to previous constraint
            if i > 0:
                prev_id = f"{char_name}_c_{len(net)-2}"
                net.add_edge(prev_id, node_id, relationship="sequential")

        # Update event graph for causal/temporal chains
        event_graph = self.event_graphs[char_name]
        for c in constraints:
            if c["type"] in ["temporal", "causal", "event_sequence"]:
                event_id = f"event_{len(event_graph)}"
                event_graph.add_node(event_id, **c)
                # Link to last event
                last_nodes = [n for n, d in event_graph.in_degree() if d == 0 and n != event_id]
                if last_nodes:
                    event_graph.add_edge(last_nodes[-1], event_id)

    def get_character_development(self, char_name: str) -> float:
        # Placeholder for development score based on constraint accumulation
        if char_name not in self.constraint_networks:
            return 0.0
        return len(self.constraint_networks[char_name].nodes) / 10.0

    def get_constraint_network(self, char_name: str) -> nx.Graph:
        return self.constraint_networks.get(char_name, nx.Graph())

    def calculate_coherence(self, char_name: str) -> float:
        # Placeholder for coherence calculation
        # e.g., density of the constraint network or logic checks
        net = self.get_constraint_network(char_name)
        if len(net) < 2:
            return 1.0
        return nx.density(net) # Example metric