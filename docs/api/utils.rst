Utilities API
=============

This module provides utility functions and classes for PyTorch Graph.

Classes
-------

.. autoclass:: pytorch_graph.GraphNode
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.GraphEdge
   :members:
   :undoc-members:

.. autoclass:: pytorch_graph.OperationType
   :members:
   :undoc-members:

Examples
--------

Using GraphNode
~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import GraphNode, OperationType

   # Create a graph node
   node = GraphNode(
       id="node_1",
       name="Linear Layer",
       operation_type=OperationType.FORWARD,
       module_name="linear",
       input_shapes=[(1, 784)],
       output_shapes=[(1, 128)]
   )

   print(f"Node: {node.name}")
   print(f"Type: {node.operation_type}")
   print(f"Input shapes: {node.input_shapes}")

Using GraphEdge
~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import GraphEdge

   # Create a graph edge
   edge = GraphEdge(
       source_id="node_1",
       target_id="node_2",
       edge_type="data_flow",
       tensor_shape=(1, 128)
   )

   print(f"Edge: {edge.source_id} -> {edge.target_id}")
   print(f"Type: {edge.edge_type}")

Using OperationType
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_graph import OperationType

   # Check operation types
   print(OperationType.FORWARD.value)  # "forward"
   print(OperationType.BACKWARD.value)  # "backward"
   print(OperationType.TENSOR_OP.value)  # "tensor_op"

See Also
--------

* :doc:`computational_graph_tracking` - For computational graph analysis
* :doc:`model_analysis` - For model analysis functions
