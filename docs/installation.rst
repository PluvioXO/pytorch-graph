Installation
============

PyTorch Graph can be installed using pip. The package is available on PyPI and supports Python 3.8+.

Basic Installation
------------------

Install the core package:

.. code-block:: bash

   pip install pytorch-graph

This will install PyTorch Graph with all essential dependencies.

Enhanced Installation
---------------------

For additional features and better performance:

.. code-block:: bash

   pip install pytorch-graph[full]

This includes:
* Enhanced visualization features
* Additional export formats
* Performance optimizations
* Extended color schemes

Development Installation
------------------------

For development and contributing:

.. code-block:: bash

   pip install pytorch-graph[dev]

This includes:
* Development dependencies
* Testing frameworks
* Code quality tools
* Documentation tools

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/your-username/pytorch-graph.git
   cd pytorch-graph
   pip install -e .[dev]

Requirements
------------

Core Requirements
~~~~~~~~~~~~~~~~~

* **Python**: ≥ 3.8
* **PyTorch**: ≥ 1.8.0
* **matplotlib**: ≥ 3.3.0
* **numpy**: ≥ 1.19.0

Optional Requirements
~~~~~~~~~~~~~~~~~~~~~

* **plotly**: For interactive visualizations
* **torchinfo**: For enhanced model summaries
* **networkx**: For advanced graph analysis
* **pillow**: For image processing

Verification
------------

Verify your installation:

.. code-block:: python

   import torch
   from pytorch_graph import generate_architecture_diagram
   
   # Create a simple model
   model = torch.nn.Sequential(
       torch.nn.Linear(10, 5),
       torch.nn.ReLU(),
       torch.nn.Linear(5, 1)
   )
   
   # Generate a test diagram
   generate_architecture_diagram(
       model=model,
       input_shape=(1, 10),
       output_path="test_diagram.png"
   )
   
   print("✅ PyTorch Graph installed successfully!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Install PyTorch first: ``pip install torch``

**ImportError: No module named 'matplotlib'**
   Install matplotlib: ``pip install matplotlib``

**Permission denied errors**
   Use ``pip install --user pytorch-graph`` for user installation

**Version conflicts**
   Use a virtual environment:

   .. code-block:: bash

      python -m venv pytorch-graph-env
      source pytorch-graph-env/bin/activate  # On Windows: pytorch-graph-env\Scripts\activate
      pip install pytorch-graph

Performance Tips
~~~~~~~~~~~~~~~~

* Use ``pytorch-graph[full]`` for better performance
* Ensure you have sufficient memory for large models
* Use GPU acceleration when available
* Consider using smaller input tensors for initial testing

Support
-------

If you encounter issues:

1. Check the `troubleshooting section <troubleshooting.html>`_
2. Search `GitHub Issues <https://github.com/your-username/pytorch-graph/issues>`_
3. Create a new issue with detailed information
4. Join our `Discussions <https://github.com/your-username/pytorch-graph/discussions>`_
