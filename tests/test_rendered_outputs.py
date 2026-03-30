from pathlib import Path

from PIL import Image, ImageStat
import pytest
import torch
import torch.nn as nn

from pytorch_graph import (
    analyze_model,
    generate_architecture_diagram,
    track_computational_graph,
)
from pytorch_graph.utils.submission_styles import normalize_submission_type


def build_sample_model() -> nn.Module:
    """Create a compact CNN that exercises the main rendering paths."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def assert_png_has_visible_content(path: Path, min_width: int = 800, min_height: int = 500) -> None:
    """Verify that a generated PNG exists, opens, and contains visible drawing content."""
    assert path.exists(), f"Expected output file to exist: {path}"
    assert path.stat().st_size > 10_000, f"Expected a non-trivial PNG file: {path}"

    with Image.open(path) as image:
        width, height = image.size
        assert width >= min_width, f"Expected width >= {min_width}, got {width}"
        assert height >= min_height, f"Expected height >= {min_height}, got {height}"

        grayscale = image.convert("L")
        minimum, maximum = grayscale.getextrema()
        stats = ImageStat.Stat(grayscale)

        assert minimum < 245, "Image appears too close to blank white output"
        assert maximum - minimum > 15, "Image has too little tonal variation"
        assert stats.stddev[0] > 5, "Image appears visually flat"


def test_architecture_diagrams_render_distinct_publication_outputs(tmp_path: Path) -> None:
    model = build_sample_model()

    research_path = tmp_path / "cnn_research.png"
    flowchart_path = tmp_path / "cnn_flowchart.png"

    returned_research = generate_architecture_diagram(
        model=model,
        input_shape=(3, 32, 32),
        output_path=str(research_path),
        title="CNN Architecture (Research Paper Style)",
        style="research_paper",
    )
    returned_flowchart = generate_architecture_diagram(
        model=model,
        input_shape=(3, 32, 32),
        output_path=str(flowchart_path),
        title="CNN Architecture (Flowchart Style)",
        style="flowchart",
    )

    assert Path(returned_research) == research_path
    assert Path(returned_flowchart) == flowchart_path

    assert_png_has_visible_content(research_path, min_width=1200, min_height=1200)
    assert_png_has_visible_content(flowchart_path, min_width=900, min_height=1200)
    assert research_path.read_bytes() != flowchart_path.read_bytes()


@pytest.mark.parametrize("submission_type", ["arxiv", "iop", "icml", "neurips"])
def test_research_diagram_supports_submission_profiles(tmp_path: Path, submission_type: str) -> None:
    model = build_sample_model()
    output_path = tmp_path / f"cnn_{submission_type}.png"

    returned_path = generate_architecture_diagram(
        model=model,
        input_shape=(3, 32, 32),
        output_path=str(output_path),
        title=f"CNN Architecture ({submission_type})",
        style="research_paper",
        submission_type=submission_type,
    )

    assert Path(returned_path) == output_path
    assert_png_has_visible_content(output_path, min_width=1200, min_height=1200)


def test_submission_alias_normalizes_arixiv_to_arxiv() -> None:
    assert normalize_submission_type("arixiv") == "arxiv"


def test_submission_profiles_produce_distinct_architecture_artifacts(tmp_path: Path) -> None:
    model = build_sample_model()
    arxiv_path = tmp_path / "cnn_arxiv.png"
    neurips_path = tmp_path / "cnn_neurips.png"

    generate_architecture_diagram(
        model=model,
        input_shape=(3, 32, 32),
        output_path=str(arxiv_path),
        title="CNN Architecture (arXiv)",
        style="research_paper",
        submission_type="arxiv",
    )
    generate_architecture_diagram(
        model=model,
        input_shape=(3, 32, 32),
        output_path=str(neurips_path),
        title="CNN Architecture (NeurIPS)",
        style="research_paper",
        submission_type="neurips",
    )

    assert_png_has_visible_content(arxiv_path, min_width=1200, min_height=1200)
    assert_png_has_visible_content(neurips_path, min_width=1200, min_height=1200)
    assert arxiv_path.read_bytes() != neurips_path.read_bytes()


def test_computational_graph_png_renders_visible_dependency_graph(tmp_path: Path) -> None:
    model = build_sample_model()
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
    tracker = track_computational_graph(model, input_tensor)

    output_path = tmp_path / "cnn_computational_graph.png"
    returned_path = tracker.save_graph_png(
        filepath=str(output_path),
        width=1200,
        height=900,
        dpi=150,
        show_legend=True,
        node_size=18,
        font_size=9,
        submission_type="neurips",
    )

    assert Path(returned_path) == output_path
    assert_png_has_visible_content(output_path, min_width=1000, min_height=700)

    summary = tracker.get_graph_summary()
    assert summary["graph_source"] == "autograd"
    assert summary["total_nodes"] >= 6
    assert summary["total_edges"] >= 5

    graph_data = tracker.get_graph_data()
    assert any((node.get("metadata") or {}).get("family") for node in graph_data["nodes"])
    assert any((node.get("metadata") or {}).get("depth") is not None for node in graph_data["nodes"])


def test_model_analysis_returns_expected_sections_for_demo_like_model() -> None:
    model = build_sample_model()

    analysis = analyze_model(model, input_shape=(3, 32, 32), detailed=True)

    assert "basic_info" in analysis
    assert "parameters" in analysis
    assert "memory" in analysis
    assert "layers" in analysis
    assert "complexity" in analysis
    assert "architecture" in analysis

    assert analysis["basic_info"]["total_parameters"] > 0
    assert analysis["basic_info"]["trainable_parameters"] > 0
    assert analysis["basic_info"]["total_layers"] >= 5
    assert analysis["memory"]["total_memory_mb"] > 0
