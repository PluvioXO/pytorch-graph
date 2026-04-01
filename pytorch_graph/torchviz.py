"""
torchviz-style autograd graph helpers built on top of pytorch-graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .utils.computational_graph import ComputationalGraphTracker


DOT_PALETTE = {
    "linear": "#D55C4B",
    "convolution": "#4A90E2",
    "activation": "#3FAE68",
    "normalization": "#E7A73F",
    "pooling": "#8C62D7",
    "reduction": "#708090",
    "tensor": "#516779",
    "backward": "#7E5BD6",
    "gradient": "#7B8794",
    "parameter": "#B84234",
    "io": "#2AA198",
    "other": "#8E9AA8",
}


def _slugify(value: str) -> str:
    """Build a filesystem-friendly stem from a human-readable title."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() or "autograd_graph"


def _normalize_output_path(
    filename: Optional[str],
    directory: Optional[str],
    suffix: str,
    fallback_title: str,
) -> Path:
    """Resolve a render target similar to graphviz render/save semantics."""
    base = Path(directory) if directory else Path.cwd()
    if filename is None:
        target = base / f"{_slugify(fallback_title)}{suffix}"
    else:
        target = Path(filename)
        if not target.is_absolute():
            target = base / target
        if target.suffix == "":
            target = target.with_suffix(suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _escape_dot(value: str) -> str:
    """Escape labels for DOT output."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_shape(shape: Any) -> str:
    """Format tensor shapes consistently for labels."""
    if shape in (None, (), []):
        return "n/a"
    if isinstance(shape, (list, tuple)):
        return "x".join(str(dim) for dim in shape)
    return str(shape)


def _shorten(value: Any, max_chars: int) -> str:
    """Limit text length so node labels stay readable."""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _merge_named_tensors(
    model: Optional[nn.Module],
    params: Optional[Mapping[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Merge model parameters with explicit user-supplied names."""
    named_tensors: Dict[str, torch.Tensor] = {}

    if model is not None:
        for name, parameter in model.named_parameters():
            named_tensors[name] = parameter

    for name, value in (params or {}).items():
        if torch.is_tensor(value):
            named_tensors[name] = value

    return named_tensors


def _apply_output_names(tracker: ComputationalGraphTracker, output_names: Optional[Sequence[str]]) -> None:
    """Rename synthetic output nodes when the caller provides stable names."""
    if output_names is None:
        return

    graph_output_nodes = [
        node
        for node in tracker.nodes.values()
        if (getattr(node, "metadata", None) or {}).get("is_output")
    ]
    if len(graph_output_nodes) != len(output_names):
        raise ValueError(
            f"Expected {len(graph_output_nodes)} output names, received {len(output_names)}."
        )

    for node, output_name in zip(sorted(graph_output_nodes, key=lambda item: item.id), output_names):
        node.name = output_name


@dataclass
class AutogradGraph:
    """Rich graph artifact with torchviz-like ergonomics and better exports."""

    tracker: ComputationalGraphTracker
    title: str
    show_shapes: bool = True
    show_parameters: bool = True
    show_metadata: bool = False
    max_attr_chars: int = 80

    @property
    def graph_data(self) -> Dict[str, Any]:
        """Return the serialized graph payload."""
        return self.tracker.get_graph_data()

    @property
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics for the captured graph."""
        return self.graph_data["summary"]

    @property
    def source(self) -> str:
        """Graphviz DOT source for compatibility with torchviz workflows."""
        return self.to_dot()

    def to_dict(self) -> Dict[str, Any]:
        """Return the graph as a plain dictionary."""
        return self.graph_data

    def to_dot(
        self,
        *,
        graph_name: str = "AutogradGraph",
        rankdir: str = "LR",
        show_shapes: Optional[bool] = None,
        show_parameters: Optional[bool] = None,
        show_metadata: Optional[bool] = None,
        max_attr_chars: Optional[int] = None,
    ) -> str:
        """Serialize the captured graph to DOT text."""
        graph_data = self.graph_data
        show_shapes = self.show_shapes if show_shapes is None else show_shapes
        show_parameters = self.show_parameters if show_parameters is None else show_parameters
        show_metadata = self.show_metadata if show_metadata is None else show_metadata
        max_attr_chars = self.max_attr_chars if max_attr_chars is None else max_attr_chars

        lines = [
            f'digraph "{_escape_dot(graph_name)}" {{',
            f"  rankdir={rankdir};",
            '  graph [bgcolor="white", fontname="Helvetica", labelloc="t", labeljust="l"];',
            '  node [shape="box", style="rounded,filled", color="#D0D5DD", fontname="Helvetica", fontsize="10", margin="0.15,0.08"];',
            '  edge [color="#98A2B3", arrowsize="0.7", penwidth="1.1", fontname="Helvetica", fontsize="9"];',
            f'  label="{_escape_dot(self.title)}";',
        ]

        for node in graph_data["nodes"]:
            metadata = node.get("metadata") or {}
            family = metadata.get("family", "other")
            fill = DOT_PALETTE.get(family, DOT_PALETTE["other"])

            label_lines = [node.get("name", "Unknown")]
            module_name = node.get("module_name")
            if module_name:
                label_lines.append(f"module: {module_name}")

            if show_shapes:
                input_shapes = node.get("input_shapes") or []
                output_shapes = node.get("output_shapes") or []
                if input_shapes:
                    label_lines.append(
                        "in: " + ", ".join(_format_shape(shape) for shape in input_shapes)
                    )
                if output_shapes:
                    label_lines.append(
                        "out: " + ", ".join(_format_shape(shape) for shape in output_shapes)
                    )

            if show_parameters:
                parameter_count = (node.get("parameters") or {}).get("count")
                if parameter_count:
                    label_lines.append(f"params: {parameter_count:,}")

            if show_metadata:
                for key, value in metadata.items():
                    if key in {"family", "depth", "is_output", "is_parameter"}:
                        continue
                    label_lines.append(f"{key}: {_shorten(value, max_attr_chars)}")

            label = _escape_dot("\n".join(label_lines))
            lines.append(
                f'  "{_escape_dot(node["id"])}" [label="{label}", fillcolor="{fill}", fontcolor="white"];'
            )

        for edge in graph_data["edges"]:
            edge_label_parts = []
            if show_shapes and edge.get("tensor_shape"):
                edge_label_parts.append(_format_shape(edge["tensor_shape"]))
            if edge.get("edge_type") not in {"autograd_dependency", None}:
                edge_label_parts.append(edge["edge_type"].replace("_", " "))

            edge_label = ""
            if edge_label_parts:
                edge_label = f' [label="{_escape_dot(" | ".join(edge_label_parts))}"]'

            lines.append(
                f'  "{_escape_dot(edge["source_id"])}" -> "{_escape_dot(edge["target_id"])}"{edge_label};'
            )

        lines.append("}")
        return "\n".join(lines)

    def save(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """Save DOT source to disk, mirroring graphviz's save() behavior."""
        return self.save_dot(filename=filename, directory=directory)

    def save_dot(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """Write the DOT source to a file."""
        target = _normalize_output_path(filename, directory, ".dot", self.title)
        target.write_text(self.to_dot(), encoding="utf-8")
        return str(target)

    def save_json(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """Write the serialized graph payload to JSON."""
        target = _normalize_output_path(filename, directory, ".json", self.title)
        target.write_text(json.dumps(self.graph_data, indent=2, default=str), encoding="utf-8")
        return str(target)

    def save_png(
        self,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        *,
        width: int = 1200,
        height: int = 800,
        dpi: int = 300,
        show_legend: bool = True,
        node_size: int = 20,
        font_size: int = 10,
        submission_type: Optional[str] = None,
    ) -> str:
        """Render the graph to a publication-quality PNG."""
        target = _normalize_output_path(filename, directory, ".png", self.title)
        return self.tracker.save_graph_png(
            filepath=str(target),
            width=width,
            height=height,
            dpi=dpi,
            show_legend=show_legend,
            node_size=node_size,
            font_size=font_size,
            submission_type=submission_type,
            title=self.title,
        )

    def render(
        self,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        *,
        format: str = "png",
        cleanup: bool = False,
        **kwargs: Any,
    ) -> str:
        """Render to PNG, DOT, or JSON using graphviz-like semantics."""
        del cleanup

        normalized_format = format.lower()
        if normalized_format == "png":
            return self.save_png(filename=filename, directory=directory, **kwargs)
        if normalized_format in {"dot", "gv"}:
            return self.save_dot(filename=filename, directory=directory)
        if normalized_format == "json":
            return self.save_json(filename=filename, directory=directory)
        raise ValueError(f"Unsupported render format: {format}")

    def __str__(self) -> str:
        return self.source


def make_dot(
    var: Any,
    params: Optional[Mapping[str, Any]] = None,
    *,
    model: Optional[nn.Module] = None,
    inputs: Optional[Any] = None,
    title: Optional[str] = None,
    output_names: Optional[Sequence[str]] = None,
    show_shapes: bool = True,
    show_parameters: bool = True,
    show_metadata: bool = False,
    show_attrs: bool = False,
    show_saved: bool = False,
    max_attr_chars: int = 80,
    track_memory: bool = False,
    track_timing: bool = False,
    track_tensor_ops: bool = False,
) -> AutogradGraph:
    """
    Build a torchviz-style autograd graph with richer exports and no Graphviz dependency.

    When both ``model`` and ``inputs`` are provided, the graph is recaptured from a fresh
    forward pass so module names, tensor shapes, and parameter counts are attached to the
    autograd nodes. Otherwise, the graph is traced directly from ``var``.
    """
    metadata_enabled = show_metadata or show_attrs or show_saved
    named_tensors = _merge_named_tensors(model, params)

    tracker_model = model if model is not None else nn.Identity()
    tracker = ComputationalGraphTracker(
        tracker_model,
        track_memory=track_memory if model is not None and inputs is not None else False,
        track_timing=track_timing if model is not None and inputs is not None else False,
        track_tensor_ops=track_tensor_ops if model is not None and inputs is not None else False,
    )

    if model is not None and inputs is not None:
        tracker.capture_execution(
            inputs,
            parameter_names={id(tensor): name for name, tensor in named_tensors.items()},
        )
        _apply_output_names(tracker, output_names)
        resolved_title = title or f"{type(model).__name__} Autograd Graph"
    else:
        if var is None:
            raise ValueError("make_dot() requires `var` when `model` and `inputs` are not provided.")
        tracker.capture_output(
            var,
            params=named_tensors,
            output_names=list(output_names) if output_names is not None else None,
        )
        resolved_title = title or "PyTorch Autograd Graph"

    return AutogradGraph(
        tracker=tracker,
        title=resolved_title,
        show_shapes=show_shapes,
        show_parameters=show_parameters,
        show_metadata=metadata_enabled,
        max_attr_chars=max_attr_chars,
    )


make_autograd_dot = make_dot

