"""
Diagram renderer for creating PNG architecture diagrams from PyTorch models.
"""

from typing import List, Dict, Any, Optional, Tuple
import warnings
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. PNG diagram generation will be disabled.")

from ..utils.layer_info import LayerInfo


class DiagramRenderer:
    """Renders neural network architecture diagrams as PNG files."""
    
    def __init__(self, width: int = 16, height: int = 10, dpi: int = 300, style: str = "standard"):
        """
        Initialize diagram renderer.
        
        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch for output quality
            style: Rendering style ('standard' or 'research_paper')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for PNG diagrams. Install with: pip install matplotlib")
        
        self.width = width
        self.height = height
        self.dpi = dpi
        self.style = style
        
        # Configure style-specific settings
        if self.style == "research_paper":
            self._setup_research_paper_style()
        else:
            self._setup_standard_style()
    
    def _setup_standard_style(self):
        """Setup standard diagram style."""
        self.color_map = {
            'Conv1d': '#3498db', 'Conv2d': '#2980b9', 'Conv3d': '#1f4e79',
            'ConvTranspose2d': '#5dade2',
            'Linear': '#e74c3c', 'LazyLinear': '#c0392b', 'Bilinear': '#f1948a',
            'ReLU': '#27ae60', 'LeakyReLU': '#2ecc71', 'Sigmoid': '#58d68d',
            'Tanh': '#82e5aa', 'GELU': '#a9dfbf', 'SiLU': '#d5f4e6',
            'BatchNorm1d': '#f39c12', 'BatchNorm2d': '#e67e22', 'BatchNorm3d': '#d68910',
            'LayerNorm': '#f8c471', 'GroupNorm': '#f7dc6f',
            'MaxPool1d': '#8e44ad', 'MaxPool2d': '#9b59b6', 'MaxPool3d': '#a569bd',
            'AvgPool2d': '#bb8fce', 'AdaptiveAvgPool2d': '#d2b4de',
            'LSTM': '#16a085', 'GRU': '#48c9b0', 'RNN': '#76d7c4',
            'Dropout': '#95a5a6', 'Dropout2d': '#bdc3c7',
            'Embedding': '#a0522d',
            'Flatten': '#34495e', 'Reshape': '#34495e',
            'default': '#7f8c8d'
        }
        self.font_family = 'sans-serif'
        self.plt_style = 'default'
    
    def _setup_research_paper_style(self):
        """Setup research paper quality style."""
        # Muted, printer-friendly colors for research papers
        self.color_map = {
            'Conv1d': '#e3f2fd', 'Conv2d': '#e3f2fd', 'Conv3d': '#e3f2fd',
            'ConvTranspose2d': '#e3f2fd',
            'Linear': '#ffebee', 'LazyLinear': '#ffebee', 'Bilinear': '#ffebee',
            'ReLU': '#e8f5e8', 'LeakyReLU': '#e8f5e8', 'Sigmoid': '#e8f5e8',
            'Tanh': '#e8f5e8', 'GELU': '#e8f5e8', 'SiLU': '#e8f5e8',
            'BatchNorm1d': '#fff3e0', 'BatchNorm2d': '#fff3e0', 'BatchNorm3d': '#fff3e0',
            'LayerNorm': '#fff3e0', 'GroupNorm': '#fff3e0',
            'MaxPool1d': '#f3e5f5', 'MaxPool2d': '#f3e5f5', 'MaxPool3d': '#f3e5f5',
            'AvgPool2d': '#f3e5f5', 'AdaptiveAvgPool2d': '#f3e5f5',
            'LSTM': '#e0f2f1', 'GRU': '#e0f2f1', 'RNN': '#e0f2f1',
            'Dropout': '#f5f5f5', 'Dropout2d': '#f5f5f5',
            'Embedding': '#efebe9',
            'Flatten': '#f8f9fa', 'Reshape': '#f8f9fa',
            'default': '#f5f5f5'
        }
        self.font_family = 'serif'
        self.plt_style = 'classic'
    
    def render_architecture_diagram(self, layers: List[LayerInfo], 
                                  connections: Dict[str, List[str]],
                                  title: str = "Neural Network Architecture",
                                  output_path: str = "architecture.png") -> str:
        """
        Render architecture diagram and save as PNG.
        
        Args:
            layers: List of layer information
            connections: Layer connections
            title: Diagram title
            output_path: Output PNG file path
            
        Returns:
            Path to the saved PNG file
        """
        if not layers:
            raise ValueError("No layers provided for diagram generation")
        
        # Apply matplotlib style
        plt.style.use(self.plt_style)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(layers) + 2)
        ax.axis('off')
        
        # Add title with style-appropriate formatting
        title_size = 18 if self.style == "research_paper" else 20
        ax.text(5, len(layers) + 1.5, title, fontsize=title_size, fontweight='bold', 
                ha='center', va='center', fontfamily=self.font_family)
        
        # Calculate layer positions and sizes
        layer_positions = {}
        box_width = 3.5
        box_height = 0.8
        x_center = 5
        
        # Draw layers from top to bottom
        for i, layer in enumerate(layers):
            y_pos = len(layers) - i
            layer_positions[layer.name] = (x_center, y_pos)
            
            # Get color for layer type
            color = self.color_map.get(layer.layer_type, self.color_map['default'])
            
            # Create layer box with style-appropriate formatting
            if self.style == "research_paper":
                # Clean rectangles for research papers
                from matplotlib.patches import Rectangle
                box = Rectangle(
                    (x_center - box_width/2, y_pos - box_height/2),
                    box_width, box_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.0,
                    alpha=0.9
                )
                text_color = 'black'
            else:
                # Rounded boxes for standard style
                box = FancyBboxPatch(
                    (x_center - box_width/2, y_pos - box_height/2),
                    box_width, box_height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                text_color = 'white'
            
            ax.add_patch(box)
            
            # Add layer name and type with appropriate text color
            ax.text(x_center, y_pos + 0.1, layer.name, 
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   color=text_color, fontfamily=self.font_family)
            ax.text(x_center, y_pos - 0.1, f"({layer.layer_type})", 
                   fontsize=10, ha='center', va='center',
                   color=text_color, style='italic', fontfamily=self.font_family)
            
            # Add parameter count and shape info to the right
            param_text = f"{layer.parameters:,} params" if layer.parameters > 0 else "0 params"
            shape_text = f"Input: {layer.input_shape}\nOutput: {layer.output_shape}"
            
            info_color = '#333333' if self.style == "research_paper" else 'black'
            
            ax.text(x_center + box_width/2 + 0.2, y_pos + 0.1, param_text,
                   fontsize=9, ha='left', va='center', fontweight='bold',
                   fontfamily=self.font_family, color=info_color)
            ax.text(x_center + box_width/2 + 0.2, y_pos - 0.2, shape_text,
                   fontsize=8, ha='left', va='center', color='gray',
                   fontfamily=self.font_family)
        
        # Draw connections
        for source_name, target_names in connections.items():
            if source_name in layer_positions:
                source_pos = layer_positions[source_name]
                
                for target_name in target_names:
                    if target_name in layer_positions:
                        target_pos = layer_positions[target_name]
                        
                        # Draw arrow from source to target
                        arrow = ConnectionPatch(
                            (source_pos[0], source_pos[1] - box_height/2),
                            (target_pos[0], target_pos[1] + box_height/2),
                            "data", "data",
                            arrowstyle="->",
                            shrinkA=5, shrinkB=5,
                            mutation_scale=20,
                            fc="black", ec="black",
                            linewidth=2
                        )
                        ax.add_patch(arrow)
        
        # Add legend and summary with appropriate styling
        self._add_legend(ax, layers)
        self._add_summary(ax, layers)
        
        # Add figure caption for research papers
        if self.style == "research_paper":
            ax.text(5, 0.3, f"Figure: {title}", 
                   fontsize=10, ha='center', va='center', fontfamily=self.font_family, 
                   style='italic', color='#333333')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def _add_legend(self, ax, layers: List[LayerInfo]):
        """Add legend showing layer type colors."""
        unique_types = list(set(layer.layer_type for layer in layers))
        
        if self.style == "research_paper":
            # Text-based legend for research papers
            legend_x = 0.5
            legend_y_start = len(layers) - 1
            
            ax.text(legend_x, legend_y_start + 1, "Layer Types:", 
                   fontsize=11, fontweight='bold', ha='left', va='center', 
                   fontfamily=self.font_family)
            
            for i, layer_type in enumerate(sorted(unique_types)):
                y_pos = legend_y_start - i * 0.35
                color = self.color_map.get(layer_type, self.color_map['default'])
                
                # Small colored box
                legend_box = patches.Rectangle((legend_x, y_pos - 0.08), 0.25, 0.16, 
                                             facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(legend_box)
                
                # Type label
                ax.text(legend_x + 0.35, y_pos, layer_type, 
                       fontsize=9, ha='left', va='center', fontfamily=self.font_family)
        else:
            # Standard matplotlib legend
            legend_elements = []
            for layer_type in sorted(unique_types):
                color = self.color_map.get(layer_type, self.color_map['default'])
                legend_elements.append(patches.Patch(color=color, label=layer_type))
            
            ax.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(-0.1, 0.5), fontsize=10)
    
    def _add_summary(self, ax, layers: List[LayerInfo]):
        """Add summary statistics to the diagram."""
        total_params = sum(layer.parameters for layer in layers)
        trainable_params = sum(layer.trainable_params for layer in layers)
        total_layers = len(layers)
        
        if self.style == "research_paper":
            # Compact summary for research papers
            summary_text = f"""Model Summary:
Architecture: Neural Network
Total Layers: {total_layers}
Parameters: {self._format_param_count(total_params)}
Input Resolution: Variable
Output Classes: Variable

Key Features:
• End-to-end trainable
• Batch processing
• GPU compatible"""
        else:
            # Detailed summary for standard style
            summary_text = f"""Model Summary:
Total Layers: {total_layers}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable: {total_params - trainable_params:,}"""
        
        # Position summary on the right side with appropriate styling
        ax.text(8.5, len(layers)/2, summary_text, 
               fontsize=9 if self.style == "research_paper" else 11, 
               ha='left', va='center', fontfamily=self.font_family,
               bbox=dict(boxstyle="round,pad=0.4" if self.style == "research_paper" else "round,pad=0.5", 
                        facecolor='white' if self.style == "research_paper" else 'lightgray', 
                        edgecolor='black' if self.style == "research_paper" else None,
                        linewidth=1 if self.style == "research_paper" else 0,
                        alpha=0.95 if self.style == "research_paper" else 0.8))
    
    def _format_param_count(self, count):
        """Format parameter count for research papers."""
        if count >= 1000000:
            return f"{count/1000000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        else:
            return str(count)
    
    def render_model_diagram(self, model, input_shape: Tuple[int, ...], 
                           title: str = None, output_path: str = "model_architecture.png") -> str:
        """
        Render diagram directly from PyTorch model.
        
        Args:
            model: PyTorch model (torch.nn.Module)
            input_shape: Input tensor shape
            title: Diagram title (auto-generated if None)
            output_path: Output PNG file path
            
        Returns:
            Path to the saved PNG file
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for model diagram generation")
        
        # Import the parser to extract layer info
        from ..core.parser import PyTorchModelParser
        
        # Parse the model
        parser = PyTorchModelParser()
        layers, connections = parser.parse_model(model, input_shape)
        
        # Generate title if not provided
        if title is None:
            model_name = type(model).__name__
            param_count = sum(p.numel() for p in model.parameters())
            title = f"{model_name} Architecture ({param_count:,} parameters)"
        
        # Render the diagram
        return self.render_architecture_diagram(layers, connections, title, output_path)


class SimpleDiagramRenderer:
    """Simplified diagram renderer that works without matplotlib."""
    
    def __init__(self):
        """Initialize simple renderer."""
        pass
    
    def render_text_diagram(self, layers: List[LayerInfo], 
                           connections: Dict[str, List[str]],
                           title: str = "Neural Network Architecture",
                           output_path: str = "architecture.txt") -> str:
        """
        Create a text-based architecture diagram.
        
        Args:
            layers: List of layer information
            connections: Layer connections
            title: Diagram title
            output_path: Output text file path
            
        Returns:
            Path to the saved text file
        """
        output = []
        output.append("=" * 80)
        output.append(f"  {title}")
        output.append("=" * 80)
        output.append("")
        
        # Add layer information
        total_params = sum(layer.parameters for layer in layers)
        output.append(f"Total Layers: {len(layers)}")
        output.append(f"Total Parameters: {total_params:,}")
        output.append("")
        output.append("Layer Details:")
        output.append("-" * 80)
        
        for i, layer in enumerate(layers, 1):
            # Layer type indicator
            indicator = "🔵" if "Conv" in layer.layer_type else "🔴" if layer.layer_type == "Linear" else "🟢"
            
            output.append(f"{i:2d}. {indicator} {layer.name:<15} ({layer.layer_type:<12}) {layer.parameters:>10,} params")
            output.append(f"    Input:  {layer.input_shape}")
            output.append(f"    Output: {layer.output_shape}")
            output.append("")
        
        # Add connections
        output.append("Connections:")
        output.append("-" * 40)
        for source, targets in connections.items():
            for target in targets:
                output.append(f"{source} → {target}")
        
        output.append("")
        output.append("=" * 80)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(output))
        
        return output_path
    
    def render_model_diagram(self, model, input_shape: Tuple[int, ...], 
                           title: str = None, output_path: str = "model_architecture.txt") -> str:
        """
        Render text diagram directly from PyTorch model.
        
        Args:
            model: PyTorch model (torch.nn.Module)
            input_shape: Input tensor shape
            title: Diagram title (auto-generated if None)
            output_path: Output text file path
            
        Returns:
            Path to the saved text file
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for model diagram generation")
        
        # Import the parser to extract layer info
        from ..core.parser import PyTorchModelParser
        
        # Parse the model
        parser = PyTorchModelParser()
        layers, connections = parser.parse_model(model, input_shape)
        
        # Generate title if not provided
        if title is None:
            model_name = type(model).__name__
            param_count = sum(p.numel() for p in model.parameters())
            title = f"{model_name} Architecture ({param_count:,} parameters)"
        
        # Render the diagram
        return self.render_text_diagram(layers, connections, title, output_path) 