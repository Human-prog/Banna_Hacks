"""Hosted inference comparison utilities for VisionForge AI."""

from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image, ImageFilter, ImageOps

BASE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = BASE_DIR / "outputs" / "experiments"


def edge_iou(reference: Image.Image, generated: Image.Image) -> float:
    """Return a simple edge-overlap proxy, not a validated geometry score."""
    def edge_mask(image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
        return ImageOps.autocontrast(
            gray.filter(ImageFilter.FIND_EDGES)
        ).point(lambda pixel: 255 if pixel >= 80 else 0)

    first = edge_mask(reference)
    second = edge_mask(generated)
    second = second.resize(first.size, Image.Resampling.NEAREST)

    a = list(first.getdata())
    b = list(second.getdata())
    intersection = sum(x > 0 and y > 0 for x, y in zip(a, b))
    union = sum(x > 0 or y > 0 for x, y in zip(a, b))

    return round(intersection / union, 4) if union else 1.0


def run_comparison(
    sketch_path: str,
    prompt_guided: str,
    prompt_baseline: str,
    profile: str,
    provider: str,
    text_model: str,
    edit_model: str,
    token: str,
):
    """Generate both modes, save artifacts, and return gallery/report/export."""
    from app.processing import prepare_sketch

    if not token.strip():
        raise ValueError("HF_TOKEN is missing. Check your local .env file.")
    if not sketch_path:
        raise ValueError("Upload a sketch before starting an experiment.")

    prepared = prepare_sketch(sketch_path, profile)
    client = InferenceClient(
        provider=provider,
        api_key=token,
        timeout=240,
    )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENT_DIR / f"comparison_{stamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = EXPERIMENT_DIR / f"comparison_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    prepared.processed.save(run_dir / "sketch_processed.png")
    prepared.edges.save(run_dir / "sketch_edges.png")

    results = []
    records = []

    jobs = [
        ("Prompt-only baseline", text_model, prompt_baseline, False),
        ("Sketch-guided", edit_model, prompt_guided, True),
    ]

    for label, model, prompt, guided in jobs:
        started = time.perf_counter()
        try:
            if guided:
                image = client.image_to_image(
                    prepared.processed,
                    prompt=prompt,
                    model=model,
                )
            else:
                image = client.text_to_image(
                    prompt=prompt,
                    model=model,
                )
        except Exception as exc:
            # Keep the failed experiment's error useful without exposing credentials.
            raise RuntimeError(
                f"{label} failed using model {model}: "
                f"{type(exc).__name__}. Check provider access, model availability, "
                "network connectivity, and inference quota."
            ) from None

        elapsed = round(time.perf_counter() - started, 2)
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"{label} returned an unsupported image result.")

        image = image.convert("RGB")
        filename = "guided.png" if guided else "baseline.png"
        image.save(run_dir / filename, format="PNG", optimize=True)

        score = edge_iou(prepared.edges, image)
        record = {
            "mode": label,
            "model": model,
            "provider": provider,
            "generation_seconds": elapsed,
            "width": image.width,
            "height": image.height,
            "edge_overlap_proxy": score,
            "prompt": prompt,
        }
        records.append(record)
        results.append((str(run_dir / filename), label))

    metadata = {
        "experiment": "Prompt-only versus sketch-guided generation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": profile,
        "sketch_report": prepared.report,
        "results": records,
        "methodology_note": (
            "This comparison uses two inference modes and may use different models. "
            "The hosted interface does not guarantee a shared random seed. "
            "Edge overlap is a rough pixel-level proxy, not a measure of semantic "
            "similarity, geometric accuracy, or engineering correctness."
        ),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# VisionForge Experiment Lab",
        "",
        "**Experiment:** Prompt-only baseline versus sketch-guided generation",
        "",
        f"- **Date:** {metadata['created_at']}",
        f"- **Processing profile:** {profile}",
        f"- **Sketch size:** {prepared.processed.width} × {prepared.processed.height}",
        "",
        "| Measurement | Prompt-only baseline | Sketch-guided |",
        "|---|---:|---:|",
    ]
    baseline, guided = records
    report_lines.extend([
        f"| Model | `{baseline['model']}` | `{guided['model']}` |",
        f"| Generation time | {baseline['generation_seconds']} s | {guided['generation_seconds']} s |",
        f"| Output size | {baseline['width']} × {baseline['height']} | {guided['width']} × {guided['height']} |",
        f"| Edge overlap proxy | {baseline['edge_overlap_proxy']:.4f} | {guided['edge_overlap_proxy']:.4f} |",
        "",
        "## Interpretation and limitations",
        "",
        metadata["methodology_note"],
        "",
        "Review both images visually. The edge score can be affected by background, "
        "lighting, image composition, and small shifts. It must not be treated as "
        "proof of geometry preservation or engineering validity.",
        "",
    ])
    report = "\n".join(report_lines)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    bundle = run_dir / "experiment_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in (
            "sketch_processed.png",
            "sketch_edges.png",
            "baseline.png",
            "guided.png",
            "metadata.json",
            "report.md",
        ):
            archive.write(run_dir / filename, arcname=filename)

    return results, report, str(bundle)
