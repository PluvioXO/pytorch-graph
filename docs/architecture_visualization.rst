Architecture Visualization
==========================

PyTorch Graph provides comprehensive architecture visualization capabilities for PyTorch models.

Overview
--------

The architecture visualization module allows you to generate professional-quality diagrams of your PyTorch models in multiple styles:

* **Flowchart Style**: Enhanced vertical flowchart with detailed information
* **Research Paper Style**: Clean, academic formatting for publications
* **Standard Style**: Traditional neural network visualization

Features
--------

* **Professional Quality**: High-resolution output up to 300 DPI
* **Multiple Styles**: Choose the best style for your use case
* **Comprehensive Information**: Parameter counts, memory usage, tensor shapes
* **Customizable Output**: Adjustable dimensions, DPI, and styling
* **Publication Ready**: Perfect for research papers and presentations

Basic Usage
-----------

Generate a simple architecture diagram:

.. code-block:: python

   import torch.nn as nn
   from pytorch_graph import generate_architecture_diagram

   model = nn.Sequential(
       nn.Linear(784, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="model_architecture.png",
       title="My Neural Network"
   )

Diagram Styles
--------------

Flowchart Style (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The flowchart style provides enhanced information display:

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="flowchart.png",
       style="flowchart"
   )

**Features:**
* Lightning bolt icons for activation functions
* Memory usage per layer (e.g., "~1.2MB")
* Data flow indicators on arrows
* Summary panel with total parameters and memory
* Color-coded model complexity

Research Paper Style
~~~~~~~~~~~~~~~~~~~~

Perfect for academic publications:

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="research.png",
       style="research_paper"
   )

**Features:**
* Clean, minimal design
* Academic formatting and typography
* Publication-ready quality
* Professional color scheme

Standard Style
~~~~~~~~~~~~~~

Traditional neural network visualization:

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="standard.png",
       style="standard"
   )

**Features:**
* Classic neural network layout
* Balanced information density
* Traditional styling

Advanced Configuration
----------------------

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="custom.png",
       title="Custom Model Architecture",
       style="flowchart",
       dpi=300,
       show_legend=True
   )

High-Quality Output
~~~~~~~~~~~~~~~~~~~

For publication-quality images:

.. code-block:: python

   generate_architecture_diagram(
       model=model,
       input_shape=(1, 784),
       output_path="publication_quality.png",
       style="research_paper",
       dpi=300
   )

Examples
--------

CNN Architecture
~~~~~~~~~~~~~~~~

.. code-block:: python

   cnn_model = nn.Sequential(
       nn.Conv2d(3, 32, 3, padding=1),
       nn.BatchNorm2d(32),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Conv2d(32, 64, 3, padding=1),
       nn.BatchNorm2d(64),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((1, 1)),
       nn.Flatten(),
       nn.Linear(64, 10)
   )

   generate_architecture_diagram(
       model=cnn_model,
       input_shape=(1, 3, 32, 32),
       output_path="cnn_architecture.png",
       title="CNN Architecture"
   )

ResNet-like Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ResidualBlock(nn.Module):
       def __init__(self, in_channels, out_channels):
           super().__init__()
           self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
           self.bn1 = nn.BatchNorm2d(out_channels)
           self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
           self.bn2 = nn.BatchNorm2d(out_channels)
           self.relu = nn.ReLU()
           
       def forward(self, x):
           residual = x
           out = self.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           out += residual
           return self.relu(out)

   class ResNetModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
           self.bn1 = nn.BatchNorm2d(64)
           self.relu = nn.ReLU()
           self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
           
           self.res_block1 = ResidualBlock(64, 64)
           self.res_block2 = ResidualBlock(64, 64)
           
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           self.fc = nn.Linear(64, 1000)
           
       def forward(self, x):
           x = self.relu(self.bn1(self.conv1(x)))
           x = self.maxpool(x)
           x = self.res_block1(x)
           x = self.res_block2(x)
           x = self.avgpool(x)
           x = torch.flatten(x, 1)
           x = self.fc(x)
           return x

   resnet_model = ResNetModel()
   
   generate_architecture_diagram(
       model=resnet_model,
       input_shape=(1, 3, 224, 224),
       output_path="resnet_architecture.png",
       title="ResNet-like Architecture"
   )

Best Practices
--------------

* **Choose the right style** for your use case
* **Use high DPI** (300) for publication-quality output
* **Provide meaningful titles** for your diagrams
* **Test with smaller models** first to verify the output
* **Use appropriate input shapes** that match your model's expected input

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'matplotlib'**
   Install matplotlib: ``pip install matplotlib``

**File not found errors**
   Ensure the output directory exists

**Large model visualization issues**
   Consider using smaller input shapes for testing

**Style not found errors**
   Use one of the supported styles: "flowchart", "research_paper", "standard"

See Also
--------

* :doc:`computational_graph_tracking` - For computational graph visualization
* :doc:`model_analysis` - For model analysis functions
* :doc:`api/architecture` - For complete API reference
