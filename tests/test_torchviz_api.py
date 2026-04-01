from pathlib import Path

from PIL import Image, ImageStat
import torch
import torch.nn as nn

from pytorch_graph import make_dot


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


def build_conv_model() -> nn.Module:
    """Create a compact model that exercises parameter and activation nodes."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5),
    )


def test_make_dot_from_output_exports_dot_json_and_png(tmp_path: Path) -> None:
    model = build_conv_model()
    input_tensor = torch.randn(1, 3, 16, 16, requires_grad=True)
    output = model(input_tensor)

    graph = make_dot(
        output,
        params=dict(model.named_parameters()),
        title="Torchviz Replacement",
        show_metadata=True,
    )

    assert graph.summary["graph_source"] == "autograd_output"
    assert "digraph" in graph.source
    assert "0.weight" in graph.source

    dot_path = Path(graph.render("replacement_graph", directory=str(tmp_path), format="dot"))
    json_path = Path(graph.render("replacement_graph", directory=str(tmp_path), format="json"))
    png_path = Path(
        graph.render(
            "replacement_graph",
            directory=str(tmp_path),
            format="png",
            width=1100,
            height=800,
            dpi=150,
            submission_type="neurips",
        )
    )

    assert dot_path.suffix == ".dot"
    assert json_path.suffix == ".json"
    assert png_path.suffix == ".png"
    assert '"graph_source": "autograd_output"' in json_path.read_text(encoding="utf-8")
    assert_png_has_visible_content(png_path, min_width=900, min_height=650)


def test_make_dot_with_model_inputs_supports_multi_input_and_named_outputs(tmp_path: Path) -> None:
    class MultiInputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(6, 4)

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self.proj(left + right)

    model = MultiInputModel()
    left = torch.randn(2, 6, requires_grad=True)
    right = torch.randn(2, 6, requires_grad=True)
    output = model(left, right)

    graph = make_dot(
        output,
        model=model,
        inputs=(left, right),
        output_names=["logits"],
        show_metadata=True,
    )

    graph_dict = graph.to_dict()
    output_nodes = [
        node
        for node in graph_dict["nodes"]
        if (node.get("metadata") or {}).get("is_output")
    ]

    assert graph.summary["graph_source"] == "autograd"
    assert any(node["name"] == "logits" for node in output_nodes)
    assert "module: proj" in graph.source

    png_path = Path(
        graph.save_png(
            filename="multi_input_graph",
            directory=str(tmp_path),
            width=1000,
            height=700,
            dpi=150,
            submission_type="arxiv",
        )
    )

    assert_png_has_visible_content(png_path, min_width=850, min_height=550)

