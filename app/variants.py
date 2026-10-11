"""Utilities for repeatable concept variants and comparison exports."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

VARIANT_DIRECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Reference-faithful refinement",
        "Prioritize fidelity to the supplied sketch or written brief. Keep the silhouette and feature placement conservative; refine only materials, edge quality, and proportions where needed.",
    ),
    (
        "Alternative form exploration",
        "Explore a clearly different but plausible enclosure silhouette and control layout. Preserve every explicit must-have feature, required interface, and stated dimensional constraint.",
    ),
    (
        "Material and interaction study",
        "Explore a distinct material palette and thoughtful interaction details while keeping the core form recognizable and the requested functions intact. Avoid decorative features that conflict with the constraints.",
    ),
)


def variant_direction(index: int) -> tuple[str, str]:
    """Return a human-readable, repeatable direction for variant 1, 2, or 3."""
    if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= len(VARIANT_DIRECTIONS):
        raise ValueError("Variant index must be 1, 2, or 3.")
    return VARIANT_DIRECTIONS[index - 1]


def add_variant_direction(prompt: str, index: int) -> tuple[str, str]:
    """Append a controlled variation instruction without discarding constraints."""
    title, direction = variant_direction(index)
    augmented = (
        f"{prompt.strip()}\n\n"
        f"Concept variation {index}: {title}. {direction} "
        "The variation is exploratory, not a verified engineering solution. Keep all explicit must-have features and exclusions."
    )
    return augmented, title


def _safe_slug(text: str, max_length: int = 52) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return result[:max_length] or "concept"


def create_comparison_sheet(
    image_paths: Sequence[str | Path],
    labels: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Build a labeled contact sheet with Pillow; does not call any AI service."""
    if not image_paths:
        raise ValueError("At least one concept image is required.")
    if len(image_paths) != len(labels):
        raise ValueError("Each image must have a matching label.")
    if len(image_paths) > 3:
        raise ValueError("Comparison sheets support up to three variants.")

    columns = min(3, len(image_paths))
    cell_w, image_h, title_h, gap, margin = 440, 330, 58, 18, 24
    rows = (len(image_paths) + columns - 1) // columns
    canvas_w = margin * 2 + columns * cell_w + (columns - 1) * gap
    canvas_h = margin * 2 + rows * (title_h + image_h) + (rows - 1) * gap + 46
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#111827")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("segoeui.ttf", 19)
        font_sub = ImageFont.truetype("segoeui.ttf", 13)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    for idx, (path_value, label) in enumerate(zip(image_paths, labels)):
        path = Path(path_value)
        if not path.is_file():
            raise ValueError(f"Concept image does not exist: {path}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_w - 20, image_h - 20), Image.Resampling.LANCZOS)
            preview = Image.new("RGB", (cell_w - 20, image_h - 20), "#F3F4F6")
            preview.paste(image, ((preview.width - image.width) // 2, (preview.height - image.height) // 2))

        row, col = divmod(idx, columns)
        x = margin + col * (cell_w + gap)
        y = margin + row * (title_h + image_h + gap)
        draw.rounded_rectangle((x, y, x + cell_w, y + title_h + image_h), radius=14, fill="#1F2937", outline="#374151", width=2)
        draw.text((x + 14, y + 10), f"{idx + 1:02d}  {label[:38]}", fill="#F9FAFB", font=font_title)
        draw.text((x + 14, y + 35), "AI CONCEPT · VISUAL REVIEW", fill="#9CA3AF", font=font_sub)
        canvas.paste(preview, (x + 10, y + title_h + 10))

    footer_y = canvas_h - 34
    draw.text((margin, footer_y), "VisionForge AI  |  Visual comparison only — not validated engineering", fill="#D1D5DB", font=font_sub)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG", optimize=True)
    return out


def create_variant_bundle(
    product_name: str,
    image_paths: Sequence[str | Path],
    labels: Sequence[str],
    comparison_sheet: str | Path,
    output_path: str | Path,
) -> Path:
    """Package all variant artifacts in one ZIP while excluding nested ZIP archives."""
    if not image_paths:
        raise ValueError("At least one concept image is required for an export bundle.")
    if len(image_paths) != len(labels):
        raise ValueError("Each image must have a matching label.")
    sheet = Path(comparison_sheet)
    if not sheet.is_file():
        raise ValueError("Comparison sheet was not found.")

    manifest_lines = [
        f"# VisionForge AI — {product_name}",
        "",
        f"This bundle contains {len(image_paths)} generated concept variant(s).",
        "",
        "## Included variants",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for idx, (path_value, label) in enumerate(zip(image_paths, labels), start=1):
            image_path = Path(path_value)
            if not image_path.is_file():
                raise ValueError(f"Concept image does not exist: {image_path}")
            image_arc = f"variants/{idx:02d}_{_safe_slug(label)}.png"
            bundle.write(image_path, arcname=image_arc)
            manifest_lines.append(f"{idx}. **{label}** — `{image_arc}`")

            run_dir = image_path.parent
            for artifact_name in ("brief.md", "metadata.json", "sketch_processed.png", "sketch_edges.png"):
                artifact = run_dir / artifact_name
                if artifact.is_file():
                    bundle.write(artifact, arcname=f"variants/{idx:02d}_{_safe_slug(label)}/{artifact_name}")

        bundle.write(sheet, arcname="comparison_sheet.png")
        manifest_lines.extend([
            "",
            "## Limitations",
            "These are AI-generated visual concepts. They are not validated CAD, schematics, verified dimensions, a bill of materials, a safety assessment, or manufacturing instructions. Review all technical requirements independently.",
        ])
        bundle.writestr("README.md", "\n".join(manifest_lines) + "\n")
    return out
