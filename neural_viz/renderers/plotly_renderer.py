"""
Plotly-based 3D renderer for neural network visualization.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ..utils.layer_info import LayerInfo


class PlotlyRenderer:
    """
    Renders neural network architectures using Plotly for interactive 3D visualization.
    """
    
    def __init__(self, theme: str = 'plotly_dark', width: int = 1200, height: int = 800):
        """
        Initialize the Plotly renderer.
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', etc.)
            width: Figure width in pixels
            height: Figure height in pixels
        """
        self.theme = theme
        self.width = width
        self.height = height
        self.fig = None
        
    def render(self, layers: List[LayerInfo], 
               connections: Optional[Dict[str, List[str]]] = None,
               title: str = "Neural Network Architecture",
               show_connections: bool = True,
               show_labels: bool = True,
               interactive: bool = True,
               **kwargs) -> go.Figure:
        """
        Render the neural network in 3D.
        
        Args:
            layers: List of LayerInfo objects with positions
            connections: Dictionary of layer connections
            title: Plot title
            show_connections: Whether to show connections between layers
            show_labels: Whether to show layer labels
            interactive: Whether to enable interactive features
            **kwargs: Additional rendering options
            
        Returns:
            Plotly Figure object
        """
        # Create figure
        self.fig = go.Figure()
        
        # Render layers as 3D shapes
        self._render_layers(layers, show_labels)
        
        # Render connections between layers
        if show_connections and connections:
            self._render_connections(layers, connections)
        
        # Configure layout
        self._configure_layout(title, interactive)
        
        # Apply theme
        self.fig.update_layout(template=self.theme)
        
        return self.fig
    
    def _render_layers(self, layers: List[LayerInfo], show_labels: bool = True):
        """Render layers as 3D boxes/spheres."""
        for layer in layers:
            # Get layer properties
            x, y, z = layer.position
            width, height, depth = layer.size
            color = layer.color
            
            # Create 3D box for the layer
            self._add_3d_box(
                x, y, z, 
                width, height, depth,
                color=color,
                name=layer.get_display_name(),
                hover_info=self._get_layer_hover_info(layer)
            )
            
            # Add label if requested
            if show_labels:
                self._add_text_label(
                    x, y, z + depth/2 + 0.2,
                    layer.name,
                    color='white' if 'dark' in self.theme else 'black'
                )
    
    def _add_3d_box(self, x: float, y: float, z: float,
                    width: float, height: float, depth: float,
                    color: str, name: str, hover_info: str):
        """Add a 3D box to represent a layer."""
        # Define the 8 vertices of a box
        vertices = np.array([
            [x - width/2, y - height/2, z - depth/2],  # 0
            [x + width/2, y - height/2, z - depth/2],  # 1
            [x + width/2, y + height/2, z - depth/2],  # 2
            [x - width/2, y + height/2, z - depth/2],  # 3
            [x - width/2, y - height/2, z + depth/2],  # 4
            [x + width/2, y - height/2, z + depth/2],  # 5
            [x + width/2, y + height/2, z + depth/2],  # 6
            [x - width/2, y + height/2, z + depth/2],  # 7
        ])
        
        # Define the 12 triangular faces of the box (2 triangles per face)
        faces = np.array([
            # Bottom face (z = z - depth/2)
            [0, 1, 2], [0, 2, 3],
            # Top face (z = z + depth/2)
            [4, 7, 6], [4, 6, 5],
            # Front face (y = y - height/2)
            [0, 4, 5], [0, 5, 1],
            # Back face (y = y + height/2)
            [2, 6, 7], [2, 7, 3],
            # Left face (x = x - width/2)
            [0, 3, 7], [0, 7, 4],
            # Right face (x = x + width/2)
            [1, 5, 6], [1, 6, 2],
        ])
        
        # Create mesh3d trace
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=0.8,
            name=name,
            hovertemplate=hover_info + "<extra></extra>",
            showlegend=False
        )
        
        self.fig.add_trace(mesh)
    
    def _add_text_label(self, x: float, y: float, z: float, 
                       text: str, color: str = 'black'):
        """Add a text label at the specified position."""
        label = go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='text',
            text=[text],
            textfont=dict(size=10, color=color),
            showlegend=False,
            hoverinfo='skip'
        )
        
        self.fig.add_trace(label)
    
    def _render_connections(self, layers: List[LayerInfo], 
                          connections: Dict[str, List[str]]):
        """Render connections between layers as lines."""
        # Create a mapping from layer names to positions
        layer_positions = {layer.name: layer.position for layer in layers}
        
        # Collect all connection lines
        line_x, line_y, line_z = [], [], []
        
        for source_name, target_names in connections.items():
            if source_name not in layer_positions:
                continue
                
            source_pos = layer_positions[source_name]
            
            for target_name in target_names:
                if target_name not in layer_positions:
                    continue
                    
                target_pos = layer_positions[target_name]
                
                # Add line segments
                line_x.extend([source_pos[0], target_pos[0], None])
                line_y.extend([source_pos[1], target_pos[1], None])
                line_z.extend([source_pos[2], target_pos[2], None])
        
        # Add connections as a single trace
        if line_x:
            connections_trace = go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                line=dict(
                    color='rgba(128, 128, 128, 0.6)',
                    width=3
                ),
                name='Connections',
                showlegend=False,
                hoverinfo='skip'
            )
            
            self.fig.add_trace(connections_trace)
    
    def _get_layer_hover_info(self, layer: LayerInfo) -> str:
        """Generate hover information for a layer."""
        info = f"<b>{layer.name}</b><br>"
        info += f"Type: {layer.layer_type}<br>"
        info += f"Input Shape: {layer.input_shape}<br>"
        info += f"Output Shape: {layer.output_shape}<br>"
        info += f"Parameters: {layer.parameters:,}<br>"
        info += f"Trainable: {layer.trainable_params:,}<br>"
        
        # Add metadata information
        if layer.metadata:
            info += "<br><b>Details:</b><br>"
            for key, value in layer.metadata.items():
                if key not in ['has_weight', 'has_bias']:
                    info += f"{key.replace('_', ' ').title()}: {value}<br>"
        
        return info
    
    def _configure_layout(self, title: str, interactive: bool = True):
        """Configure the 3D plot layout."""
        self.fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(
                    title="Layer Progression",
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    showbackground=False
                ),
                yaxis=dict(
                    title="Layer Groups",
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    showbackground=False
                ),
                zaxis=dict(
                    title="Depth",
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    showbackground=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube' if interactive else 'auto'
            ),
            width=self.width,
            height=self.height,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        # Configure interaction
        if not interactive:
            self.fig.update_layout(
                scene_dragmode=False,
                scene_camera_projection_type='orthographic'
            )
    
    def add_parameter_visualization(self, layers: List[LayerInfo]):
        """Add parameter count visualization as bar chart."""
        # Create subplot with secondary y-axis
        fig_with_bars = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            specs=[[{"type": "scene"}], [{"type": "xy"}]],
            subplot_titles=("3D Architecture", "Parameter Count by Layer")
        )
        
        # Copy 3D traces to subplot
        for trace in self.fig.data:
            fig_with_bars.add_trace(trace, row=1, col=1)
        
        # Add parameter bar chart
        layer_names = [layer.name for layer in layers]
        param_counts = [layer.parameters for layer in layers]
        
        bar_trace = go.Bar(
            x=layer_names,
            y=param_counts,
            marker_color=[layer.color for layer in layers],
            name="Parameters",
            showlegend=False
        )
        
        fig_with_bars.add_trace(bar_trace, row=2, col=1)
        
        # Update layout
        fig_with_bars.update_layout(self.fig.layout)
        fig_with_bars.update_xaxes(tickangle=45, row=2, col=1)
        fig_with_bars.update_yaxes(title_text="Parameter Count", row=2, col=1)
        
        self.fig = fig_with_bars
    
    def export_html(self, filename: str, include_plotlyjs: str = 'cdn'):
        """Export the visualization as HTML."""
        if self.fig is None:
            raise ValueError("No figure to export. Call render() first.")
        
        self.fig.write_html(filename, include_plotlyjs=include_plotlyjs)
    
    def export_image(self, filename: str, format: str = 'png', 
                    scale: float = 2.0, width: Optional[int] = None, 
                    height: Optional[int] = None):
        """Export the visualization as static image."""
        if self.fig is None:
            raise ValueError("No figure to export. Call render() first.")
        
        self.fig.write_image(
            filename, 
            format=format, 
            scale=scale,
            width=width or self.width,
            height=height or self.height
        )
    
    def show(self, **kwargs):
        """Display the visualization."""
        if self.fig is None:
            raise ValueError("No figure to show. Call render() first.")
        
        self.fig.show(**kwargs)
    
    def get_figure(self) -> go.Figure:
        """Get the current figure object."""
        return self.fig 