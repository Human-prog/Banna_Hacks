"""Image preparation and lightweight diagnostics for VisionForge AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

MAX_SIDE = 1024
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 35_000_000
PROFILES = ("Original", "Clean sketch", "High-contrast linework")


@dataclass
class PreparedSketch:
    """Prepared images plus measurable facts about the uploaded sketch."""

    processed: Image.Image
    edges: Image.Image
    report: dict[str, int | float | str]


def prepare_sketch(path: str | Path | None, profile: str = "Clean sketch") -> PreparedSketch:
    """Orient, validate, downscale and enhance an uploaded sketch using Pillow only."""
    if not path:
        raise ValueError("Upload a sketch before running image processing.")

    source = Path(path)
    if not source.is_file():
        raise ValueError("The uploaded image could not be found. Please upload it again.")

    byte_size = source.stat().st_size
    if byte_size <= 0:
        raise ValueError("The uploaded file is empty.")
    if byte_size > MAX_UPLOAD_BYTES:
        raise ValueError("The sketch is larger than 20 MB. Resize or compress it and try again.")
    if profile not in PROFILES:
        profile = "Clean sketch"

    try:
        with Image.open(source) as opened:
            source_format = opened.format or source.suffix.lstrip(".").upper() or "Unknown"
            oriented = ImageOps.exif_transpose(opened)
            original_width, original_height = oriented.size
            if original_width < 64 or original_height < 64:
                raise ValueError("The image is too small. Upload a sketch at least 64 × 64 pixels.")
            if original_width * original_height > MAX_PIXELS:
                raise ValueError("The image dimensions are too large to process safely (over 35 megapixels).")
            rgb = oriented.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("This file is not a supported image. Try PNG, JPG, JPEG, or WEBP.") from exc
    except OSError as exc:
        raise ValueError("The image could not be read. Try saving it as PNG or JPG and uploading again.") from exc

    rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    if profile == "Original":
        processed = rgb.copy()
    else:
        gray = ImageOps.grayscale(rgb)
        gray = ImageOps.autocontrast(gray, cutoff=(1, 1))
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        if profile == "High-contrast linework":
            gray = ImageEnhance.Contrast(gray).enhance(1.9)
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
        else:
            gray = ImageEnhance.Contrast(gray).enhance(1.35)
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
        processed = gray.convert("RGB")

    gray_processed = ImageOps.grayscale(processed)
    edges = ImageOps.autocontrast(gray_processed.filter(ImageFilter.FIND_EDGES))
    stats = ImageStat.Stat(gray_processed)
    histogram = edges.histogram()
    total_pixels = max(1, edges.width * edges.height)
    edge_pixels = sum(histogram[48:])

    report: dict[str, int | float | str] = {
        "source_format": source_format,
        "source_width": original_width,
        "source_height": original_height,
        "processed_width": processed.width,
        "processed_height": processed.height,
        "file_size_mb": round(byte_size / (1024 * 1024), 2),
        "profile": profile,
        "contrast_stddev": round(float(stats.stddev[0]), 1),
        "edge_density_pct": round((edge_pixels / total_pixels) * 100, 1),
        "max_side_px": MAX_SIDE,
    }
    return PreparedSketch(processed=processed, edges=edges, report=report)
