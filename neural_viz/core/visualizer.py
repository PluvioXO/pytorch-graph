"""
Main neural network visualizer that orchestrates parsing, positioning, and rendering.
"""

from typing import List, Dict, Optional, Any, Union, Tuple
import warnings
from .parser import ModelParser
from ..utils.layer_info import LayerInfo
from ..utils.position_calculator import PositionCalculator
from ..renderers.plotly_renderer import PlotlyRenderer


class NeuralNetworkVisualizer:
    """
    Main class for visualizing neural network architectures in 3D.
    
    This class orchestrates the entire visualization pipeline:
    1. Parse the model to extract layer information
    2. Calculate 3D positions for layers
    3. Render the visualization
    """
    
    def __init__(self, renderer: str = 'plotly', layout_style: str = 'hierarchical',
                 spacing: float = 2.0, theme: str = 'plotly_dark', 
                 width: int = 1200, height: int = 800):
        """
        Initialize the neural network visualizer.
        
        Args:
            renderer: Rendering backend ('plotly' or 'matplotlib')
            layout_style: Layout algorithm ('hierarchical', 'circular', 'spring', 'custom')
            spacing: Spacing between layers
            theme: Color theme for visualization
            width: Figure width in pixels
            height: Figure height in pixels
        """
        self.renderer_type = renderer
        self.layout_style = layout_style
        self.spacing = spacing
        self.theme = theme
        self.width = width
        self.height = height
        
        # Initialize components
        self.parser = ModelParser()
        self.position_calculator = PositionCalculator(layout_style, spacing)
        
        # Initialize renderer
        if renderer == 'plotly':
            self.renderer = PlotlyRenderer(theme, width, height)
        else:
            raise ValueError(f"Unsupported renderer: {renderer}")
    
    def visualize(self, model, framework: Optional[str] = None, 
                 input_shape: Optional[Tuple[int, ...]] = None,
                 title: Optional[str] = None,
                 show_connections: bool = True,
                 show_labels: bool = True,
                 show_parameters: bool = False,
                 optimize_layout: bool = True,
                 export_path: Optional[str] = None,
                 **kwargs) -> Any:
        """
        Visualize a neural network model.
        
        Args:
            model: Neural network model (PyTorch or Keras)
            framework: Framework hint ('pytorch' or 'keras')
            input_shape: Input tensor shape (required for PyTorch)
            title: Plot title
            show_connections: Whether to show connections between layers
            show_labels: Whether to show layer labels
            show_parameters: Whether to show parameter count visualization
            optimize_layout: Whether to optimize layer positions
            export_path: Path to export the visualization (optional)
            **kwargs: Additional rendering options
            
        Returns:
            Rendered visualization object
        """
        # Validate model
        validation_warnings = self.parser.validate_model(model, framework)
        if validation_warnings:
            for warning in validation_warnings:
                warnings.warn(warning)
        
        # Parse model
        layers, connections = self.parser.parse_model(model, framework, input_shape)
        
        if not layers:
            raise ValueError("No layers found in the model")
        
        # Calculate positions
        positioned_layers = self.position_calculator.calculate_positions(layers, connections)
        
        # Optimize layout if requested
        if optimize_layout:
            positioned_layers = self.position_calculator.optimize_positions(
                positioned_layers, connections
            )
        
        # Generate title if not provided
        if title is None:
            framework_name = framework or self.parser._detect_framework(model)
            total_params = sum(layer.parameters for layer in positioned_layers)
            title = f"{framework_name.title()} Model - {len(positioned_layers)} Layers, {total_params:,} Parameters"
        
        # Render visualization
        fig = self.renderer.render(
            positioned_layers,
            connections,
            title=title,
            show_connections=show_connections,
            show_labels=show_labels,
            **kwargs
        )
        
        # Add parameter visualization if requested
        if show_parameters:
            self.renderer.add_parameter_visualization(positioned_layers)
        
        # Export if path provided
        if export_path:
            if export_path.endswith('.html'):
                self.renderer.export_html(export_path)
            elif export_path.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                format_type = export_path.split('.')[-1]
                self.renderer.export_image(export_path, format=format_type)
            else:
                warnings.warn(f"Unknown export format for {export_path}")
        
        return fig
    
    def visualize_pytorch(self, model, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """
        Convenience method for visualizing PyTorch models.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            **kwargs: Additional visualization options
            
        Returns:
            Rendered visualization object
        """
        return self.visualize(model, framework='pytorch', input_shape=input_shape, **kwargs)
    
    def visualize_keras(self, model, **kwargs) -> Any:
        """
        Convenience method for visualizing Keras models.
        
        Args:
            model: Keras model
            **kwargs: Additional visualization options
            
        Returns:
            Rendered visualization object
        """
        return self.visualize(model, framework='keras', **kwargs)
    
    def get_model_summary(self, model, framework: Optional[str] = None,
                         input_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        Get a summary of the model structure.
        
        Args:
            model: Neural network model
            framework: Framework hint
            input_shape: Input tensor shape (for PyTorch)
            
        Returns:
            Dictionary containing model summary
        """
        return self.parser.get_model_summary(model, framework, input_shape)
    
    def compare_models(self, models: List[Any], names: Optional[List[str]] = None,
                      frameworks: Optional[List[str]] = None,
                      input_shapes: Optional[List[Tuple[int, ...]]] = None,
                      **kwargs) -> Any:
        """
        Compare multiple models in a single visualization.
        
        Args:
            models: List of neural network models
            names: Optional list of model names
            frameworks: Optional list of framework hints
            input_shapes: Optional list of input shapes (for PyTorch models)
            **kwargs: Additional visualization options
            
        Returns:
            Rendered comparison visualization
        """
        if not models:
            raise ValueError("No models provided for comparison")
        
        if names is None:
            names = [f"Model {i+1}" for i in range(len(models))]
        
        if frameworks is None:
            frameworks = [None] * len(models)
        
        if input_shapes is None:
            input_shapes = [None] * len(models)
        
        # Parse all models
        all_layers = []
        all_connections = {}
        x_offset = 0
        
        for i, (model, name, framework, input_shape) in enumerate(
            zip(models, names, frameworks, input_shapes)
        ):
            # Parse model
            layers, connections = self.parser.parse_model(model, framework, input_shape)
            
            # Offset layers horizontally
            for layer in layers:
                layer.name = f"{name}_{layer.name}"
                layer.position = (layer.position[0] + x_offset, layer.position[1], layer.position[2])
            
            # Update connections with new names
            model_connections = {}
            for source, targets in connections.items():
                new_source = f"{name}_{source}"
                new_targets = [f"{name}_{target}" for target in targets]
                model_connections[new_source] = new_targets
            
            all_layers.extend(layers)
            all_connections.update(model_connections)
            
            # Calculate offset for next model
            max_x = max(layer.position[0] for layer in layers) if layers else 0
            x_offset = max_x + self.spacing * 3
        
        # Position all layers
        positioned_layers = self.position_calculator.calculate_positions(all_layers, all_connections)
        
        # Render comparison
        title = f"Model Comparison: {', '.join(names)}"
        fig = self.renderer.render(
            positioned_layers,
            all_connections,
            title=title,
            **kwargs
        )
        
        return fig
    
    def create_animation(self, model, framework: Optional[str] = None,
                        input_shape: Optional[Tuple[int, ...]] = None,
                        steps: int = 10, **kwargs) -> Any:
        """
        Create an animated visualization showing the model building process.
        
        Args:
            model: Neural network model
            framework: Framework hint
            input_shape: Input tensor shape (for PyTorch)
            steps: Number of animation steps
            **kwargs: Additional visualization options
            
        Returns:
            Animated visualization object
        """
        # Parse model
        layers, connections = self.parser.parse_model(model, framework, input_shape)
        
        if not layers:
            raise ValueError("No layers found in the model")
        
        # Calculate final positions
        positioned_layers = self.position_calculator.calculate_positions(layers, connections)
        
        # Create animation frames
        frames = []
        layers_per_step = max(1, len(positioned_layers) // steps)
        
        for step in range(steps):
            # Determine which layers to show in this frame
            end_idx = min((step + 1) * layers_per_step, len(positioned_layers))
            frame_layers = positioned_layers[:end_idx]
            
            # Create frame connections
            frame_connections = {}
            for source, targets in connections.items():
                if any(layer.name == source for layer in frame_layers):
                    valid_targets = [t for t in targets 
                                   if any(layer.name == t for layer in frame_layers)]
                    if valid_targets:
                        frame_connections[source] = valid_targets
            
            # Render frame
            frame_fig = self.renderer.render(
                frame_layers,
                frame_connections,
                title=f"Building Model - Step {step + 1}/{steps}",
                **kwargs
            )
            
            frames.append(frame_fig)
        
        return frames
    
    def export_summary_report(self, model, framework: Optional[str] = None,
                             input_shape: Optional[Tuple[int, ...]] = None,
                             output_path: str = "model_report.html"):
        """
        Export a comprehensive HTML report of the model.
        
        Args:
            model: Neural network model
            framework: Framework hint
            input_shape: Input tensor shape (for PyTorch)
            output_path: Path for the output HTML file
        """
        # Get model summary
        summary = self.get_model_summary(model, framework, input_shape)
        
        # Create visualization
        fig = self.visualize(
            model, framework, input_shape,
            show_parameters=True,
            title=f"Model Architecture Report"
        )
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Model Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metric {{ margin: 10px 0; }}
                .visualization {{ width: 100%; height: 800px; }}
            </style>
        </head>
        <body>
            <h1>Neural Network Model Report</h1>
            
            <div class="summary">
                <h2>Model Summary</h2>
                <div class="metric"><strong>Total Layers:</strong> {summary['total_layers']}</div>
                <div class="metric"><strong>Total Parameters:</strong> {summary['total_parameters']:,}</div>
                <div class="metric"><strong>Trainable Parameters:</strong> {summary['trainable_parameters']:,}</div>
                <div class="metric"><strong>Non-trainable Parameters:</strong> {summary['non_trainable_parameters']:,}</div>
                <div class="metric"><strong>Input Shape:</strong> {summary['input_shape']}</div>
                <div class="metric"><strong>Output Shape:</strong> {summary['output_shape']}</div>
                
                <h3>Layer Types</h3>
                <ul>
        """
        
        for layer_type, count in summary['layer_types'].items():
            html_content += f"<li>{layer_type}: {count}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="visualization">
        """
        
        # Add the plotly figure
        html_content += fig.to_html(include_plotlyjs='cdn', div_id="visualization")
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Model report exported to {output_path}")
    
    def set_theme(self, theme: str):
        """Set the visualization theme."""
        self.theme = theme
        if hasattr(self.renderer, 'theme'):
            self.renderer.theme = theme
    
    def set_layout_style(self, layout_style: str):
        """Set the layout style for positioning layers."""
        self.layout_style = layout_style
        self.position_calculator = PositionCalculator(layout_style, self.spacing)
    
    def set_spacing(self, spacing: float):
        """Set the spacing between layers."""
        self.spacing = spacing
        self.position_calculator.spacing = spacing 