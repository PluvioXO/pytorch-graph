"""
Position calculation utilities for 3D neural network visualization.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import networkx as nx
from .layer_info import LayerInfo


class PositionCalculator:
    """
    Calculates 3D positions for neural network layers in visualization.
    """
    
    def __init__(self, layout_style: str = 'hierarchical', spacing: float = 2.0):
        """
        Initialize position calculator.
        
        Args:
            layout_style: Layout algorithm ('hierarchical', 'circular', 'spring', 'custom')
            spacing: Spacing between layers
        """
        self.layout_style = layout_style
        self.spacing = spacing
        self.supported_layouts = ['hierarchical', 'circular', 'spring', 'custom']
        
        if layout_style not in self.supported_layouts:
            raise ValueError(f"Layout style '{layout_style}' not supported. "
                           f"Choose from {self.supported_layouts}")
    
    def calculate_positions(self, layers: List[LayerInfo], 
                          connections: Optional[Dict[str, List[str]]] = None) -> List[LayerInfo]:
        """
        Calculate 3D positions for all layers.
        
        Args:
            layers: List of LayerInfo objects
            connections: Optional dictionary of layer connections
            
        Returns:
            List of LayerInfo objects with updated positions
        """
        if not layers:
            return layers
        
        # Create a copy to avoid modifying original
        positioned_layers = [layer for layer in layers]
        
        if self.layout_style == 'hierarchical':
            positioned_layers = self._hierarchical_layout(positioned_layers, connections)
        elif self.layout_style == 'circular':
            positioned_layers = self._circular_layout(positioned_layers)
        elif self.layout_style == 'spring':
            positioned_layers = self._spring_layout(positioned_layers, connections)
        elif self.layout_style == 'custom':
            positioned_layers = self._custom_layout(positioned_layers, connections)
        
        return positioned_layers
    
    def _hierarchical_layout(self, layers: List[LayerInfo], 
                           connections: Optional[Dict[str, List[str]]] = None) -> List[LayerInfo]:
        """
        Arrange layers in a hierarchical (left-to-right) layout.
        """
        # Create a graph to determine layer order
        if connections:
            graph = self._create_graph(layers, connections)
            layer_levels = self._get_layer_levels(graph)
        else:
            # Simple sequential layout
            layer_levels = {layer.name: i for i, layer in enumerate(layers)}
        
        # Group layers by level
        levels = {}
        for layer in layers:
            level = layer_levels.get(layer.name, 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(layer)
        
        # Position layers
        positioned_layers = []
        for level, level_layers in levels.items():
            x = level * self.spacing
            
            # Arrange layers in this level vertically
            total_height = len(level_layers) * self.spacing
            start_y = -total_height / 2
            
            for i, layer in enumerate(level_layers):
                y = start_y + i * self.spacing
                z = 0.0
                
                # Add some randomness for better visualization
                if len(level_layers) > 1:
                    z = np.random.uniform(-0.5, 0.5)
                
                layer.position = (x, y, z)
                positioned_layers.append(layer)
        
        return positioned_layers
    
    def _circular_layout(self, layers: List[LayerInfo]) -> List[LayerInfo]:
        """
        Arrange layers in a circular layout.
        """
        n_layers = len(layers)
        radius = max(n_layers * 0.5, 3.0)
        
        positioned_layers = []
        for i, layer in enumerate(layers):
            angle = 2 * np.pi * i / n_layers
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.0
            
            layer.position = (x, y, z)
            positioned_layers.append(layer)
        
        return positioned_layers
    
    def _spring_layout(self, layers: List[LayerInfo], 
                      connections: Optional[Dict[str, List[str]]] = None) -> List[LayerInfo]:
        """
        Use spring layout algorithm for positioning.
        """
        if not connections:
            # Fall back to hierarchical layout
            return self._hierarchical_layout(layers)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        for layer in layers:
            G.add_node(layer.name)
        
        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)
        
        # Use NetworkX spring layout
        pos_2d = nx.spring_layout(G, k=self.spacing, iterations=50)
        
        # Convert to 3D positions
        positioned_layers = []
        for layer in layers:
            if layer.name in pos_2d:
                x, y = pos_2d[layer.name]
                x *= 10  # Scale up for better visualization
                y *= 10
                z = np.random.uniform(-1, 1)  # Add some Z variation
            else:
                x = y = z = 0.0
            
            layer.position = (x, y, z)
            positioned_layers.append(layer)
        
        return positioned_layers
    
    def _custom_layout(self, layers: List[LayerInfo], 
                      connections: Optional[Dict[str, List[str]]] = None) -> List[LayerInfo]:
        """
        Custom layout that groups similar layer types.
        """
        # Group layers by type
        layer_groups = {}
        for layer in layers:
            layer_type = layer.layer_type
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append(layer)
        
        # Define type order and positioning
        type_order = [
            'Input', 'Embedding',
            'Conv1d', 'Conv2d', 'Conv3d',
            'BatchNorm', 'ReLU', 'Dropout',
            'MaxPool', 'AvgPool', 'AdaptivePool',
            'Flatten', 'Reshape',
            'Linear', 'Dense',
            'LSTM', 'GRU', 'RNN',
            'Sigmoid', 'Tanh', 'Softmax'
        ]
        
        positioned_layers = []
        x_offset = 0
        
        for layer_type in type_order:
            if layer_type not in layer_groups:
                continue
            
            type_layers = layer_groups[layer_type]
            
            # Arrange layers of this type
            for i, layer in enumerate(type_layers):
                y = (i - len(type_layers) / 2) * self.spacing
                z = 0.0
                
                layer.position = (x_offset, y, z)
                positioned_layers.append(layer)
            
            x_offset += self.spacing * 1.5
        
        # Handle any remaining layers not in type_order
        remaining_layers = [layer for layer in layers 
                          if layer.layer_type not in type_order]
        
        for i, layer in enumerate(remaining_layers):
            y = (i - len(remaining_layers) / 2) * self.spacing
            z = 0.0
            layer.position = (x_offset, y, z)
            positioned_layers.append(layer)
        
        return positioned_layers
    
    def _create_graph(self, layers: List[LayerInfo], 
                     connections: Dict[str, List[str]]) -> nx.DiGraph:
        """
        Create a NetworkX graph from layer connections.
        """
        G = nx.DiGraph()
        
        # Add nodes
        for layer in layers:
            G.add_node(layer.name)
        
        # Add edges
        for source, targets in connections.items():
            for target in targets:
                if source in [l.name for l in layers] and target in [l.name for l in layers]:
                    G.add_edge(source, target)
        
        return G
    
    def _get_layer_levels(self, graph: nx.DiGraph) -> Dict[str, int]:
        """
        Determine the hierarchical level of each layer.
        """
        # Find nodes with no incoming edges (input layers)
        input_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        
        if not input_nodes:
            # If no clear input nodes, use the first node
            input_nodes = [list(graph.nodes())[0]] if graph.nodes() else []
        
        levels = {}
        visited = set()
        
        # BFS to assign levels
        current_level = [input_nodes] if input_nodes else []
        level_num = 0
        
        while current_level:
            next_level = []
            for node_list in current_level:
                for node in node_list if isinstance(node_list, list) else [node_list]:
                    if node not in visited:
                        levels[node] = level_num
                        visited.add(node)
                        
                        # Add successors to next level
                        successors = list(graph.successors(node))
                        if successors:
                            next_level.extend(successors)
            
            current_level = next_level
            level_num += 1
        
        # Handle any unvisited nodes
        for node in graph.nodes():
            if node not in levels:
                levels[node] = level_num
        
        return levels
    
    def optimize_positions(self, layers: List[LayerInfo], 
                          connections: Optional[Dict[str, List[str]]] = None) -> List[LayerInfo]:
        """
        Optimize positions to minimize edge crossings and improve readability.
        """
        if not connections:
            return layers
        
        # Simple optimization: minimize total edge length
        positioned_layers = layers.copy()
        
        # Iterative improvement
        for iteration in range(10):
            improved = False
            
            for i, layer in enumerate(positioned_layers):
                # Calculate ideal position based on connected layers
                connected_positions = []
                
                # Find connected layers
                for source, targets in connections.items():
                    if source == layer.name:
                        for target in targets:
                            target_layer = next((l for l in positioned_layers if l.name == target), None)
                            if target_layer:
                                connected_positions.append(target_layer.position)
                    
                    if layer.name in targets:
                        source_layer = next((l for l in positioned_layers if l.name == source), None)
                        if source_layer:
                            connected_positions.append(source_layer.position)
                
                if connected_positions:
                    # Calculate centroid
                    centroid = np.mean(connected_positions, axis=0)
                    
                    # Move slightly towards centroid
                    current_pos = np.array(layer.position)
                    new_pos = current_pos + 0.1 * (centroid - current_pos)
                    
                    # Update position
                    layer.position = tuple(new_pos)
                    improved = True
            
            if not improved:
                break
        
        return positioned_layers 