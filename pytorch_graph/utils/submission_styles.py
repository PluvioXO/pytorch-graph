"""
Shared styling profiles for publication-targeted rendering.
"""

from typing import Dict, Optional


_BASE_FAMILY_PALETTE = {
    "Convolution": "#355C7D",
    "Dense": "#C06C84",
    "Activation": "#6C8E5A",
    "Normalization": "#8C7B5A",
    "Pooling": "#7A6F9B",
    "Recurrent": "#4D8076",
    "Regularization": "#8A8F98",
    "Tensor Shape": "#5F6B7A",
    "Other": "#7B7F86",
}


SUBMISSION_STYLE_PROFILES: Dict[str, Dict[str, object]] = {
    "default": {
        "label": "Generic",
        "font_family": "serif",
        "caption_prefix": "Figure",
        "title_color": "#111827",
        "subtitle_color": "#4B5563",
        "rule_color": "#D1D5DB",
        "row_rule_color": "#F3F4F6",
        "box_fill": "#FCFCFD",
        "box_border": "#D0D5DD",
        "shadow_color": "#E5E7EB",
        "arrow_color": "#98A2B3",
        "legend_frame": "#D0D5DD",
        "note_color": "#667085",
        "family_palette": _BASE_FAMILY_PALETTE,
    },
    "arxiv": {
        "label": "arXiv",
        "font_family": "serif",
        "caption_prefix": "Figure",
        "title_color": "#101828",
        "subtitle_color": "#475467",
        "rule_color": "#CBD5E1",
        "row_rule_color": "#EEF2F6",
        "box_fill": "#FCFCFD",
        "box_border": "#D0D5DD",
        "shadow_color": "#E4E7EC",
        "arrow_color": "#98A2B3",
        "legend_frame": "#D0D5DD",
        "note_color": "#667085",
        "family_palette": {
            "Convolution": "#355C7D",
            "Dense": "#8E5872",
            "Activation": "#6C8E5A",
            "Normalization": "#7D6C52",
            "Pooling": "#6B73A6",
            "Recurrent": "#4D8076",
            "Regularization": "#8A8F98",
            "Tensor Shape": "#5F6B7A",
            "Other": "#7B7F86",
        },
    },
    "iop": {
        "label": "IOP",
        "font_family": "serif",
        "caption_prefix": "Fig.",
        "title_color": "#0F2747",
        "subtitle_color": "#36516E",
        "rule_color": "#B7C5D3",
        "row_rule_color": "#E8EEF5",
        "box_fill": "#FBFCFE",
        "box_border": "#BFCAD7",
        "shadow_color": "#DCE6F0",
        "arrow_color": "#6E8FA8",
        "legend_frame": "#BFCAD7",
        "note_color": "#52667A",
        "family_palette": {
            "Convolution": "#204A87",
            "Dense": "#A34747",
            "Activation": "#3F7D58",
            "Normalization": "#8E6C3F",
            "Pooling": "#6C63A8",
            "Recurrent": "#287271",
            "Regularization": "#7E8B99",
            "Tensor Shape": "#506176",
            "Other": "#6C7580",
        },
    },
    "icml": {
        "label": "ICML",
        "font_family": "sans-serif",
        "caption_prefix": "Figure",
        "title_color": "#0F172A",
        "subtitle_color": "#334155",
        "rule_color": "#BFCCD9",
        "row_rule_color": "#EDF4FA",
        "box_fill": "#FCFDFE",
        "box_border": "#C8D5E2",
        "shadow_color": "#E2E8F0",
        "arrow_color": "#6C8FB3",
        "legend_frame": "#C8D5E2",
        "note_color": "#475569",
        "family_palette": {
            "Convolution": "#2563EB",
            "Dense": "#EA580C",
            "Activation": "#16A34A",
            "Normalization": "#A16207",
            "Pooling": "#7C3AED",
            "Recurrent": "#0F766E",
            "Regularization": "#64748B",
            "Tensor Shape": "#475569",
            "Other": "#6B7280",
        },
    },
    "neurips": {
        "label": "NeurIPS",
        "font_family": "sans-serif",
        "caption_prefix": "Figure",
        "title_color": "#241B2F",
        "subtitle_color": "#55445F",
        "rule_color": "#D4C7D6",
        "row_rule_color": "#F6EFF5",
        "box_fill": "#FDFCFE",
        "box_border": "#D7CDD8",
        "shadow_color": "#EADFEA",
        "arrow_color": "#A08EAA",
        "legend_frame": "#D7CDD8",
        "note_color": "#6B5A73",
        "family_palette": {
            "Convolution": "#7C3AED",
            "Dense": "#C2416A",
            "Activation": "#0F9D8A",
            "Normalization": "#B7791F",
            "Pooling": "#8B5CF6",
            "Recurrent": "#0E7490",
            "Regularization": "#7C8595",
            "Tensor Shape": "#5B6475",
            "Other": "#7A7486",
        },
    },
}


SUBMISSION_TYPE_ALIASES = {
    "arixiv": "arxiv",
    "nips": "neurips",
}


def normalize_submission_type(submission_type: Optional[str]) -> str:
    """Normalize user-provided submission types and aliases."""
    if not submission_type:
        return "default"

    normalized = submission_type.strip().lower().replace("-", "_")
    normalized = SUBMISSION_TYPE_ALIASES.get(normalized, normalized)

    if normalized not in SUBMISSION_STYLE_PROFILES:
        supported = ", ".join(sorted(name for name in SUBMISSION_STYLE_PROFILES if name != "default"))
        raise ValueError(
            f"Unsupported submission_type: {submission_type}. "
            f"Use one of: {supported}."
        )

    return normalized


def get_submission_style(submission_type: Optional[str]) -> Dict[str, object]:
    """Return the normalized style profile for a submission target."""
    normalized = normalize_submission_type(submission_type)
    profile = dict(SUBMISSION_STYLE_PROFILES["default"])
    profile.update(SUBMISSION_STYLE_PROFILES[normalized])
    profile["key"] = normalized
    return profile


def tint_hex_color(hex_color: str, blend: float = 0.86) -> str:
    """Blend a hex color toward white for soft publication fills."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {hex_color}")

    blend = max(0.0, min(1.0, blend))
    channels = []
    for index in (0, 2, 4):
        component = int(color[index:index + 2], 16)
        blended = round(component * (1.0 - blend) + 255 * blend)
        channels.append(f"{blended:02X}")

    return f"#{''.join(channels)}"
