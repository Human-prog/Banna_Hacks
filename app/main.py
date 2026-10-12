"""VisionForge AI Ã¢â‚¬â€ sketch-to-concept studio."""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

try:
    from .local_backend import generate_local_image, local_backend_status, refine_prompt_with_ollama
except ImportError:  # support running `python app/main.py` directly
    from local_backend import generate_local_image, local_backend_status, refine_prompt_with_ollama


try:
    from .engineering_lab import evaluate_design, render_engineering_report, save_engineering_report
    from .local_ai import (
        local_runtime_status,
        inspect_sketch_with_local_vision,
        critique_concept_with_local_vision,
        refine_prompt_with_local_reasoner,
        draft_revision_prompt_with_local_reasoner,
    )
except ImportError:  # support both `python app/main.py` and `python -m app.main`
    from engineering_lab import evaluate_design, render_engineering_report, save_engineering_report
    from local_ai import (
        local_runtime_status,
        inspect_sketch_with_local_vision,
        critique_concept_with_local_vision,
        refine_prompt_with_local_reasoner,
        draft_revision_prompt_with_local_reasoner,
    )

try:
    from .experiment_lab import run_comparison
except ImportError:
    from experiment_lab import run_comparison

try:
    from .processing import PROFILES, PreparedSketch, prepare_sketch
except ImportError:  # support both `python app/main.py` and `python -m app.main`
    from processing import PROFILES, PreparedSketch, prepare_sketch

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
ENV_FILE = BASE_DIR / ".env"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ENV_FILE, override=True)

PROVIDER = os.getenv("HF_PROVIDER", "fal-ai").strip() or "fal-ai"
TEXT_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev").strip()
EDIT_MODEL = os.getenv("HF_EDIT_MODEL", "black-forest-labs/FLUX.2-klein-9B").strip()
GUIDED_MODE = "Sketch-guided (image-to-image)"
PROMPT_MODE = "Prompt-only fallback"

STYLE_GUIDES = {
    "Studio product render": (
        "Create a polished industrial-design studio render with realistic materials, coherent proportions, "
        "soft studio lighting, and a clean neutral background. Show one complete product in a three-quarter view."
    ),
    "Early prototype on desk": (
        "Show a believable early-stage hardware prototype on a tidy engineering workbench, with realistic "
        "prototype materials and restrained lighting. Keep the complete product clearly visible."
    ),
    "Exploded concept visualization": (
        "Create a clean exploded concept visualization where a few plausible major parts are separated "
        "slightly for visual clarity. Preserve the overall product silhouette and avoid implying verified internals."
    ),
}

def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:48] or "untitled-concept")


def build_prompt(
    name: str,
    description: str,
    priorities: list[str] | None,
    constraints: str,
    style: str,
    sketch_guided: bool,
) -> str:
    """Build a repeatable prompt from the user's brief and selected rendering style."""
    product_name = (name or "Untitled hardware concept").strip()[:120]
    purpose = (description or "").strip()[:2000]
    priority_text = ", ".join(priorities or []) or "balanced and practical design"
    constraint_text = (constraints or "No additional constraints provided.").strip()[:1500]
    visual_direction = STYLE_GUIDES.get(style, STYLE_GUIDES["Studio product render"])

    reference_instruction = (
        "Treat the supplied engineering sketch as the primary visual reference. Preserve its main silhouette, "
        "layout, relative proportions, and visible features. Refine linework into a coherent product concept; "
        "do not add unnecessary controls, ports, or unsupported details."
        if sketch_guided
        else "Create a product concept from the written brief. Do not invent logos or readable text."
    )

    return f"""Professional industrial-design visualization for: {product_name}.
Product purpose: {purpose}
Design priorities: {priority_text}.
Engineering constraints to respect visually: {constraint_text}
Visual direction: {visual_direction}
Reference handling: {reference_instruction}
Composition: one clear product, three-quarter view unless an exploded view is selected, realistic materials,
clear silhouette, sensible spacing, neutral background, crisp focus, polished presentation quality.
Avoid: watermarks, random lettering, decorative labels, duplicated products, floating unrelated objects,
needless features, tangled components, exaggerated proportions, or claims that the design is manufacture-ready.""".strip()


def _sketch_markdown(report: dict[str, Any] | None) -> str:
    if not report:
        return "No sketch was processed."
    return (
        "### Sketch processing report\n"
        f"- **Input:** {report['source_format']} Ã‚Â· {report['source_width']} Ãƒâ€” {report['source_height']} px Ã‚Â· "
        f"{report['file_size_mb']} MB\n"
        f"- **Prepared image:** {report['processed_width']} Ãƒâ€” {report['processed_height']} px "
        f"(longest edge capped at {report['max_side_px']} px)\n"
        f"- **Profile:** {report['profile']}\n"
        f"- **Grayscale contrast (standard deviation):** {report['contrast_stddev']}\n"
        f"- **Edge-pixel estimate:** {report['edge_density_pct']}%\n\n"
        "*Diagnostics describe image properties; they do not assess the technical correctness of the design.*"
    )


def preview_processing(sketch_path: str | None, profile: str) -> tuple[Image.Image | None, Image.Image | None, str]:
    if not sketch_path:
        return None, None, "Upload a sketch to inspect preprocessing and linework."
    try:
        prepared = prepare_sketch(sketch_path, profile)
        return prepared.processed, prepared.edges, _sketch_markdown(prepared.report)
    except ValueError as exc:
        return None, None, f"### Could not process sketch\n{exc}"
    except Exception as exc:  # keep malformed uploads from crashing the UI
        return None, None, f"### Processing error\n{type(exc).__name__}: {exc}"


def _recent_concepts(limit: int = 12) -> list[tuple[str, str]]:
    concepts: list[tuple[str, str]] = []
    if not RUNS_DIR.exists():
        return concepts
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        image_path = run_dir / "concept.png"
        metadata_path = run_dir / "metadata.json"
        if not image_path.is_file() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            caption = f"{metadata.get('product_name', 'Concept')} Ã‚Â· {metadata.get('created_at', '')}"
            concepts.append((str(image_path), caption))
        except (OSError, json.JSONDecodeError):
            continue
        if len(concepts) >= limit:
            break
    return concepts


def load_history() -> list[tuple[str, str]]:
    return _recent_concepts()


def _render_brief(metadata: dict[str, Any]) -> str:
    sketch = metadata.get("sketch_report")
    if sketch:
        sketch_section = _sketch_markdown(sketch)
    else:
        sketch_section = "No sketch was processed."

    return f"""# {metadata['product_name']}

> VisionForge AI Ã‚Â· Product concept record

## Product intent
{metadata['description']}

## Design priorities
{', '.join(metadata['priorities']) if metadata['priorities'] else 'Balanced practical design'}

## Constraints
{metadata['constraints'] or 'No additional constraints provided.'}

## Visualization settings
- **Mode:** {metadata['generation_mode']}
- **Style:** {metadata['render_style']}
- **Model:** `{metadata['model']}`
- **Provider:** `{metadata['provider']}`
- **Generation time:** {metadata['generation_seconds']:.1f} seconds
- **Output dimensions:** {metadata['output_width']} Ãƒâ€” {metadata['output_height']} px
- **Created:** {metadata['created_at']}

## Sketch diagnostics
{sketch_section}

## Prompt used
```text
{metadata['prompt']}
```

## Engineering review checklist
- [ ] Compare the generated silhouette with the original sketch.
- [ ] Verify dimensions, interfaces, clearances, and accessibility assumptions.
- [ ] Independently validate materials, electrical details, thermal behavior, and power needs.
- [ ] Create real CAD and schematics before prototyping or manufacturing.

## Limitations
This is an AI-generated visual concept. It is not validated CAD, an electrical schematic, a bill of materials, a safety assessment, or a manufacturing-ready design. Generative models can change or invent details and may not obey every constraint. Image diagnostics are descriptive only.
"""


def _save_export(
    generated: Image.Image,
    metadata: dict[str, Any],
    prepared: PreparedSketch | None,
) -> tuple[Path, Path, str]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{_slug(metadata['product_name'])}"
    run_dir = RUNS_DIR / run_name
    # Timestamp resolution plus an increment avoids overwriting runs generated in the same second.
    suffix = 2
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    image_path = run_dir / "concept.png"
    generated.convert("RGB").save(image_path, format="PNG", optimize=True)
    with Image.open(image_path) as saved_image:
        metadata["output_width"], metadata["output_height"] = saved_image.size

    included_files: list[Path] = [image_path]
    if prepared:
        processed_path = run_dir / "sketch_processed.png"
        edge_path = run_dir / "sketch_edges.png"
        prepared.processed.save(processed_path, format="PNG", optimize=True)
        prepared.edges.save(edge_path, format="PNG", optimize=True)
        included_files.extend([processed_path, edge_path])

    brief = _render_brief(metadata)
    brief_path = run_dir / "brief.md"
    brief_path.write_text(brief, encoding="utf-8")
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    included_files.extend([brief_path, metadata_path])

    bundle_path = run_dir / "visionforge_concept_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in included_files:
            bundle.write(file_path, arcname=file_path.name)
    return image_path, bundle_path, brief


def generate_concept(
    sketch_path: str | None,
    name: str,
    description: str,
    priorities: list[str] | None,
    constraints: str,
    render_style: str,
    generation_mode: str,
    profile: str,
    strength: float = 0.62,
) -> tuple[str | None, str, str, str | None, list[tuple[str, str]]]:
    """Generate through local Phi-4 Mini + local Stable Diffusion; no hosted API."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Describe what the product does before generating a concept.")

    is_guided = generation_mode == GUIDED_MODE
    if is_guided and not sketch_path:
        raise gr.Error("Sketch-guided mode needs an uploaded sketch. Upload one or select Prompt-only fallback.")

    prepared: PreparedSketch | None = None
    if sketch_path:
        try:
            prepared = prepare_sketch(sketch_path, profile)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    product_name = (name or "Untitled concept").strip()[:120] or "Untitled concept"
    original_prompt = build_prompt(
        product_name, description, priorities, constraints or "", render_style, is_guided
    )
    prompt, reasoning_note = refine_prompt_with_ollama(original_prompt)
    seed = int(time.time_ns() % 2_147_483_647)
    actual_mode = "img2img" if is_guided else "txt2img"

    try:
        generation_started = time.perf_counter()
        generated, backend = generate_local_image(
            prompt=prompt,
            mode=actual_mode,
            reference_image=prepared.processed if (is_guided and prepared) else None,
            strength=float(min(0.9, max(0.25, strength))),
            seed=seed,
            steps=int(os.getenv("VISIONFORGE_SD_STEPS", "16")),
        )
        elapsed = float(backend.get("generation_seconds") or (time.perf_counter() - generation_started))
        if not isinstance(generated, Image.Image) or generated.width < 1 or generated.height < 1:
            raise RuntimeError("The local image worker returned an invalid image.")
    except Exception as exc:
        raise gr.Error(
            f"Local generation failed: {type(exc).__name__}: {exc}. "
            "No hosted inference API was called. Check outputs/local_sd_worker.log for worker details."
        ) from exc

    model = str(backend.get("model", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    provider = str(backend.get("provider", "Local Stable Diffusion / DirectML"))
    metadata: dict[str, Any] = {
        "product_name": product_name,
        "description": description,
        "priorities": priorities or [],
        "constraints": (constraints or "").strip(),
        "render_style": render_style,
        "generation_mode": generation_mode,
        "model": model,
        "provider": provider,
        "device": backend.get("device", "DirectML"),
        "generation_seconds": round(elapsed, 2),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt": prompt,
        "original_prompt": original_prompt,
        "prompt_refinement": reasoning_note,
        "seed": backend.get("seed", seed),
        "inference_steps": backend.get("steps", int(os.getenv("VISIONFORGE_SD_STEPS", "16"))),
        "sketch_strength": float(min(0.9, max(0.25, strength))) if is_guided else None,
        "sketch_report": prepared.report if prepared else None,
        "inference_mode": "local-only; no paid or hosted inference request",
        "notice": "Visual concept only; not verified engineering or manufacturing documentation.",
    }
    image_path, bundle_path, brief = _save_export(generated, metadata, prepared)

    if is_guided:
        mode_text = (
            f"Sketch-guided img2img initialized from the uploaded sketch (strength {strength:.2f}). "
            "This is not ControlNet; geometry may drift."
        )
    else:
        mode_text = "Prompt-only text-to-image; the reference sketch is not passed into the model."
    status = f"""### Local concept generated successfully

- **Image model:** `{model}`
- **Runtime:** `{provider}` Â· `{metadata['device']}`
- **Generation time:** {elapsed:.1f} seconds
- **Output:** {metadata['output_width']} Ã— {metadata['output_height']} px
- **Mode:** {mode_text}
- **Local reasoning:** {reasoning_note}
- **Requested seed:** `{metadata['seed']}` (exact repeatability is not guaranteed on DirectML)
- **Saved to:** `outputs/runs/{image_path.parent.name}/`

**Privacy / cost:** Inference runs on this computer. No Hugging Face Inference API is called and no API token is needed. The public model weights download locally the first time.

**Engineering note:** This is an AI visual concept, not validated CAD, a verified circuit, or a manufacturing-ready design.
"""
    return str(image_path), status, brief, str(bundle_path), load_history()

def configuration_status() -> str:
    """Report local runtime status and make the no-hosted-inference path clear."""
    try:
        return local_backend_status()
    except Exception as exc:
        return (
            "### Local-only runtime\n"
            "- **Hosted inference:** Disabled in the generation path.\n"
            "- **API token:** Not required.\n"
            f"- **Status check:** {type(exc).__name__}: {exc}"
        )

def _selected_priorities(
    form_factor: list[str] | None,
    usability: list[str] | None,
    lifecycle: list[str] | None,
    engineering: list[str] | None,
) -> list[str]:
    """Combine categorized priority controls into one consistent metadata list."""
    result: list[str] = []
    for group in (form_factor, usability, lifecycle, engineering):
        for choice in group or []:
            if choice not in result:
                result.append(choice)
    return result


def _combine_constraints(required: str, envelope: str, avoid: str) -> str:
    """Keep must-haves, size/interfaces, and exclusions distinct in the prompt."""
    sections = []
    if (required or '').strip():
        sections.append(f"Required features: {(required or '').strip()}")
    if (envelope or '').strip():
        sections.append(f"Dimensions / interfaces: {(envelope or '').strip()}")
    if (avoid or '').strip():
        sections.append(f"Avoid / additional notes: {(avoid or '').strip()}")
    return "\n".join(sections)


def preview_prompt_ui(
    sketch_path: str | None,
    name: str,
    description: str,
    form_factor: list[str] | None,
    usability: list[str] | None,
    lifecycle: list[str] | None,
    engineering: list[str] | None,
    required: str,
    envelope: str,
    avoid: str,
    render_style: str,
    generation_mode: str,
) -> str:
    """Show users the exact structured prompt before spending provider credits."""
    if not (description or '').strip():
        return "Add a product description in **02 / Design brief** to preview the prompt."
    guided = generation_mode == GUIDED_MODE
    if guided and not sketch_path:
        return "Upload a sketch in **01 / Sketch** or choose **Prompt-only fallback** in this tab."
    return build_prompt(
        name,
        description,
        _selected_priorities(form_factor, usability, lifecycle, engineering),
        _combine_constraints(required, envelope, avoid),
        render_style,
        guided,
    )


def generate_from_studio(
    sketch_path, name, description,
    form_factor, usability, lifecycle, engineering,
    required, envelope, avoid, render_style, generation_mode, profile, image_strength,
):
    """Adapt grouped UI controls to the local-only generation pipeline."""
    result_image, status, brief, download_bundle, history = generate_concept(
        sketch_path,
        name,
        description,
        _selected_priorities(form_factor, usability, lifecycle, engineering),
        _combine_constraints(required, envelope, avoid),
        render_style,
        generation_mode,
        profile,
        image_strength,
    )
    return (
        result_image,
        status,
        brief,
        download_bundle,
        history,
        result_image,
        status,
    )

try:
    from .engineering_lab import (
        evaluate_design,
        render_engineering_report,
        save_engineering_report,
    )
    from .local_ai import (
        local_runtime_status,
        inspect_sketch_with_local_vision,
        critique_concept_with_local_vision,
        refine_prompt_with_local_reasoner,
        draft_revision_prompt_with_local_reasoner,
    )
except ImportError:
    from engineering_lab import (
        evaluate_design,
        render_engineering_report,
        save_engineering_report,
    )
    from local_ai import (
        local_runtime_status,
        inspect_sketch_with_local_vision,
        critique_concept_with_local_vision,
        refine_prompt_with_local_reasoner,
        draft_revision_prompt_with_local_reasoner,
    )


def evaluate_engineering_design_ui(
    enclosure_length, enclosure_width, enclosure_height, clearance,
    component_specs, observed_features, human_rating,
    description, form_factor, usability, lifecycle, engineering,
    required, envelope, avoid,
):
    priorities = _selected_priorities(
        form_factor, usability, lifecycle, engineering
    )
    try:
        result = evaluate_design(
            enclosure_length=enclosure_length,
            enclosure_width=enclosure_width,
            enclosure_height=enclosure_height,
            clearance=clearance,
            component_specs=component_specs,
            required_features=required or "",
            observed_features=observed_features or "",
            human_rating=human_rating,
            description=description or "",
            priorities=priorities,
            envelope_notes=envelope or "",
            avoid_notes=avoid or "",
        )
        bundle = save_engineering_report(result)
        return render_engineering_report(result), bundle
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(
            f"Engineering review failed: {type(exc).__name__}: {exc}"
        ) from exc


def local_runtime_status_ui():
    return local_runtime_status()


def inspect_sketch_locally_ui(
    sketch_path, profile, description, required, envelope, avoid
):
    if not sketch_path:
        return "Upload a sketch in 01 · Sketch first."
    context = "\n".join([
        description or "",
        required or "",
        envelope or "",
        avoid or "",
    ])
    try:
        return inspect_sketch_with_local_vision(sketch_path, context)
    except Exception as exc:
        return f"Local sketch analysis unavailable: {type(exc).__name__}: {exc}"


def refine_prompt_locally_ui(
    name, description, form_factor, usability, lifecycle, engineering,
    required, envelope, avoid, render_style, vision_inventory,
):
    if not (description or "").strip():
        return "Add a product description in 02 · Design brief first."

    priorities = _selected_priorities(
        form_factor, usability, lifecycle, engineering
    )
    constraints = _combine_constraints(
        required or "", envelope or "", avoid or ""
    )
    draft = build_prompt(
        name or "Untitled concept",
        description,
        priorities,
        constraints,
        render_style,
        True,
    )
    try:
        return refine_prompt_with_local_reasoner(
            product_description=description,
            base_prompt=draft,
            design_constraints=constraints,
            visual_inventory=vision_inventory or "",
        )
    except Exception as exc:
        return (
            f"Local prompt refinement unavailable: {type(exc).__name__}: {exc}"
            f"\n\nOffline fallback prompt draft:\n\n{draft}"
        )


def critique_concept_locally_ui(image_path, description, required, envelope, avoid):
    if not image_path:
        return "Upload a generated concept image first."

    requirements = "\n".join([
        f"Product purpose: {description or 'Not provided'}",
        f"Must-have features: {required or 'Not provided'}",
        f"Dimensions / interfaces: {envelope or 'Not provided'}",
        f"Avoid / extra notes: {avoid or 'Not provided'}",
    ])
    try:
        return critique_concept_with_local_vision(image_path, requirements)
    except Exception as exc:
        return f"Local image review unavailable: {type(exc).__name__}: {exc}"


def draft_revision_prompt_locally_ui(
    description, required, envelope, avoid, current_prompt, critique
):
    if not (current_prompt or "").strip():
        return "Enter the current image-generation prompt first."
    if not (critique or "").strip():
        return "Run a local concept review or enter critique notes first."

    constraints = _combine_constraints(
        required or "", envelope or "", avoid or ""
    )
    try:
        return draft_revision_prompt_with_local_reasoner(
            product_description=description or "",
            current_prompt=current_prompt,
            visual_critique=critique,
            requirements=constraints,
        )
    except Exception as exc:
        return f"Local revision drafting unavailable: {type(exc).__name__}: {exc}"


def run_experiment_from_studio(
    sketch_path, name, description,
    form_factor, usability, lifecycle, engineering,
    required, envelope, avoid, render_style, profile,
):
    """Compare prompt-only and sketch-guided local generations."""
    if not sketch_path:
        raise gr.Error("Upload a sketch in 01 · Sketch first.")
    if not (description or "").strip():
        raise gr.Error("Complete the product description in 02 · Design brief first.")

    try:
        try:
            from .local_backend import generate_local_image, refine_prompt_with_ollama
            from .processing import prepare_sketch
            from .experiment_lab import edge_iou
        except ImportError:
            from local_backend import generate_local_image, refine_prompt_with_ollama
            from processing import prepare_sketch
            from experiment_lab import edge_iou

        priorities = _selected_priorities(
            form_factor, usability, lifecycle, engineering
        )
        constraints = _combine_constraints(
            required or "", envelope or "", avoid or ""
        )

        # Keep the written prompt identical for a fair two-mode comparison.
        prompt = build_prompt(
            name or "Untitled concept",
            description,
            priorities,
            constraints,
            render_style,
            False,
        )

        prompt, reasoning_note = refine_prompt_with_ollama(prompt)
        prepared = prepare_sketch(sketch_path, profile)

        run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"local_experiment_{run_stamp}"
        suffix = 2
        while run_dir.exists():
            run_dir = RUNS_DIR / f"local_experiment_{run_stamp}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

        prepared.processed.save(run_dir / "sketch_processed.png")
        prepared.edges.save(run_dir / "sketch_edges.png")

        # Identical seed and step count; one local text-to-image call and
        # one local image-to-image call. No hosted inference provider used.
        baseline, baseline_info = generate_local_image(
            prompt=prompt,
            mode="txt2img",
            seed=42,
            steps=10,
        )

        guided, guided_info = generate_local_image(
            prompt=prompt,
            mode="img2img",
            reference_image=prepared.processed,
            strength=0.62,
            seed=42,
            steps=10,
        )

        baseline_path = run_dir / "baseline.png"
        guided_path = run_dir / "sketch_guided.png"
        baseline.save(baseline_path, format="PNG", optimize=True)
        guided.save(guided_path, format="PNG", optimize=True)

        baseline_score = edge_iou(prepared.processed, baseline)
        guided_score = edge_iou(prepared.processed, guided)

        metadata = {
            "experiment": "Local prompt-only vs sketch-guided",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": "Stable Diffusion 1.5",
            "runtime": "Local DirectML",
            "reasoning_model": "Ollama Phi-4 Mini",
            "reasoning_note": reasoning_note,
            "seed": 42,
            "steps": 10,
            "sketch_influence": 0.62,
            "prompt": prompt,
            "baseline": {
                "mode": "txt2img",
                "generation_seconds": baseline_info.get("generation_seconds"),
                "edge_overlap_proxy": baseline_score,
            },
            "sketch_guided": {
                "mode": "img2img",
                "generation_seconds": guided_info.get("generation_seconds"),
                "edge_overlap_proxy": guided_score,
            },
            "limitations": (
                "The edge-overlap value is a rough image proxy, not verified geometric "
                "accuracy. Image-to-image is not ControlNet and does not guarantee "
                "preservation of proportions or engineering correctness."
            ),
        }

        report = f"""# VisionForge Local Experiment

## Comparison setup
- **Image model:** Stable Diffusion 1.5
- **Runtime:** Local DirectML
- **Reasoning:** {reasoning_note}
- **Seed:** 42 for both runs
- **Inference steps:** 10 per run
- **Prompt:** Identical for both runs

## Results

| Measurement | Prompt-only baseline | Sketch-guided |
|---|---:|---:|
| Generation time | {baseline_info.get('generation_seconds', 'N/A')} s | {guided_info.get('generation_seconds', 'N/A')} s |
| Edge-overlap proxy | {baseline_score:.4f} | {guided_score:.4f} |
| Output dimensions | {baseline.width} × {baseline.height} | {guided.width} × {guided.height} |

## Local reasoning
{reasoning_note}

## Interpretation
Compare the two images visually. Image-to-image uses the prepared sketch as initialization, while text-to-image does not receive it.

The edge-overlap proxy measures rough pixel-level edge agreement only. Background, lighting, and composition can influence it; it is not a validated geometry or engineering score.

## Engineering review
- [ ] Check silhouette against the original sketch.
- [ ] Check every required feature and interface.
- [ ] Independently verify dimensions and clearances.
- [ ] Validate electrical, thermal, and mechanical details before prototyping.

**Privacy and cost:** Both image generations run through the local worker. This experiment does not call a hosted Hugging Face Inference API.
"""

        (run_dir / "report.md").write_text(report, encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        bundle = run_dir / "experiment_bundle.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename in (
                "sketch_processed.png",
                "sketch_edges.png",
                "baseline.png",
                "sketch_guided.png",
                "report.md",
                "metadata.json",
            ):
                archive.write(run_dir / filename, arcname=filename)

        return [
            (str(baseline_path), "Prompt-only baseline · seed 42"),
            (str(guided_path), "Sketch-guided img2img · seed 42"),
        ], report, str(bundle)

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            f"Local experiment failed: {type(exc).__name__}: {exc}. "
            "Check outputs/local_sd_worker.log for local image-worker details."
        ) from exc

CSS = """
.gradio-container {max-width: 1480px !important; margin: 0 auto !important; padding-bottom: 36px !important;}
#vf-hero {padding: 26px 28px; border: 1px solid var(--border-color-primary); border-radius: 20px; margin: 6px 0 16px; background: linear-gradient(120deg, var(--block-background-fill), var(--background-fill-secondary));}
#vf-hero h1 {letter-spacing: -0.045em; margin: 0 0 8px; font-size: clamp(2rem, 3vw, 3rem);}
#vf-kicker {font-size: .76rem; letter-spacing: .16em; font-weight: 800; opacity: .72; margin-bottom: 8px;}
#vf-lead {font-size: 1rem; line-height: 1.65; max-width: 900px; opacity: .84;}
#vf-workflow {display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 12px 0 22px;}
.vf-step {padding: 12px 14px; border-radius: 14px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); min-height: 72px;}
.vf-step-num {font-size: .73rem; letter-spacing: .12em; opacity: .7; font-weight: 800;}
.vf-step-title {font-weight: 750; margin-top: 4px;}
.vf-step-note {font-size: .78rem; opacity: .72; margin-top: 2px;}
.vf-section-note {opacity: .78; line-height: 1.55; max-width: 900px;}
.vf-footnote {font-size: .82rem; opacity: .75; line-height: 1.55;}
.vf-result {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 14px;}
button[role="tab"] {border-radius: 10px !important; font-weight: 650 !important;}
button.primary {font-weight: 750 !important;}
footer {display: none !important;}
@media (max-width: 760px) {.gradio-container {padding: 10px !important;} #vf-hero {padding: 18px;} #vf-workflow {grid-template-columns: repeat(2,minmax(0,1fr));}}
"""

with gr.Blocks(title="VisionForge AI Ã‚Â· Concept Studio") as demo:
    gr.Markdown(
        """<div id="vf-hero">
<div id="vf-kicker">HARDWARE CONCEPT LAB / V 0.2</div>
# VisionForge AI
<div id="vf-lead">Turn a rough engineering sketch into a structured product concept. Work through a guided flow, inspect how the image is prepared, set clear design constraints, then export a review-ready concept package.</div>
</div>
<div id="vf-workflow">
  <div class="vf-step"><div class="vf-step-num">STEP 01</div><div class="vf-step-title">Sketch</div><div class="vf-step-note">Upload and inspect source</div></div>
  <div class="vf-step"><div class="vf-step-num">STEP 02</div><div class="vf-step-title">Design brief</div><div class="vf-step-note">Define goals and limits</div></div>
  <div class="vf-step"><div class="vf-step-num">STEP 03</div><div class="vf-step-title">Configure</div><div class="vf-step-note">Review prompt and generate</div></div>
  <div class="vf-step"><div class="vf-step-num">STEP 04</div><div class="vf-step-title">Results</div><div class="vf-step-note">Review and export concept</div></div>
</div>"""
    )

    with gr.Tabs():
        with gr.Tab("01 - Sketch"):
            gr.Markdown("## Start with your reference\nUpload a photo, scan, or digital sketch. VisionForge prepares a clean reference image and displays an edge map before generation.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    sketch = gr.Image(
                        label="Engineering sketch",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=340,
                    )
                    profile = gr.Dropdown(
                        label="Processing profile",
                        choices=list(PROFILES),
                        value="Clean sketch",
                        info="Choose how linework is cleaned before preview and sketch-guided generation.",
                    )
                    gr.Markdown("<div class='vf-footnote'>Accepted image files: PNG, JPG/JPEG, or WEBP. Maximum upload size: 20 MB.</div>")
                with gr.Column(scale=6, min_width=300):
                    with gr.Row():
                        processed_preview = gr.Image(label="Prepared reference", interactive=False, height=265)
                        edges_preview = gr.Image(label="Edge map", interactive=False, height=265)
                    analysis_report = gr.Markdown("Upload a sketch to begin image diagnostics.")
                    with gr.Accordion("What processing does", open=False):
                        gr.Markdown("""- Applies image orientation metadata.
- Rejects files that are not valid images or are too large to handle safely.
- Scales the longest edge to at most 1024 px.
- Applies the selected cleanup profile and creates a visible edge map.

These checks describe the image only; they do not judge whether the engineering design is correct.""")
            gr.Markdown("**Next:** Open **02 Ã‚Â· Design brief** and describe what the product needs to do.")

        with gr.Tab("02 - Design brief"):
            gr.Markdown("## Define the product\nBe concrete. State what the product does, who uses it, and which features must remain present.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=280):
                    name = gr.Textbox(label="Project / product name", placeholder="e.g. Pocket-sized music player", max_lines=1)
                    description = gr.Textbox(
                        label="Product purpose and main functions",
                        placeholder="What is it Who is it for What should it do",
                        lines=6,
                    )
                    gr.Markdown("<div class='vf-footnote'>Tip: describe function before appearance. The model will use your visual style settings in the next step.</div>")
                with gr.Column(scale=6, min_width=300):
                    gr.Markdown("### Design priorities\nChoose the goals that matter most. Leaving a group blank is fine.")
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("**Form factor & mobility**")
                                form_factor = gr.CheckboxGroup(
                                    choices=["Compact", "Portable"],
                                    label="Size and movement",
                                )
                            with gr.Group():
                                gr.Markdown("**Usability**")
                                usability = gr.CheckboxGroup(
                                    choices=["Accessible controls", "Minimal parts"],
                                    label="Interaction and simplicity",
                                )
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("**Reliability & lifecycle**")
                                lifecycle = gr.CheckboxGroup(
                                    choices=["Durable", "Easy to repair", "Modular"],
                                    label="Long-term ownership",
                                )
                            with gr.Group():
                                gr.Markdown("**Engineering goals**")
                                engineering = gr.CheckboxGroup(
                                    choices=["Low power", "Low cost"],
                                    label="Practical constraints",
                                )
            with gr.Accordion("Required features and constraints", open=True):
                gr.Markdown("Separate must-have features from physical limits and things the concept should avoid.")
                with gr.Row(equal_height=False):
                    required = gr.Textbox(
                        label="Must-have features",
                        placeholder="USB-C, physical buttons, replaceable microSD card",
                        lines=3,
                    )
                    envelope = gr.Textbox(
                        label="Dimensions / interfaces",
                        placeholder="Approx. 100 Ãƒâ€” 45 mm, side-mounted headphone jack",
                        lines=3,
                    )
                    avoid = gr.Textbox(
                        label="Avoid / extra notes",
                        placeholder="No touchscreen, no glued-shut enclosure",
                        lines=3,
                    )
            gr.Markdown("**Next:** Open **03 Ã‚Â· Configure** to choose the render style and review the prompt.")

        with gr.Tab("03 - Configure & generate"):
            gr.Markdown("## Configure the generation\nPhi-4 Mini refines the prompt locally and Stable Diffusion generates on this computer. The first run downloads public model weights; no paid inference API is used.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    render_style = gr.Dropdown(
                        label="Visualization style",
                        choices=list(STYLE_GUIDES),
                        value="Studio product render",
                    )
                    generation_mode = gr.Radio(
                        label="Reference mode",
                        choices=[GUIDED_MODE, PROMPT_MODE],
                        value=GUIDED_MODE,
                        info="Both modes run locally. Sketch-guided initializes img2img with the sketch; prompt-only sends text only. The first run downloads model weights.",
                    )
                    image_strength = gr.Slider(
                        label="Sketch influence",
                        minimum=0.25, maximum=0.90, value=0.62, step=0.05,
                        info="Used for local img2img mode. Higher values allow more change from the sketch.",
                    )
                    prompt_preview = gr.Textbox(
                        label="Prompt preview",
                        value="Complete the design brief to preview the prompt.",
                        lines=13,
                        interactive=False,
                    )
                    refresh_prompt = gr.Button("Refresh prompt preview", variant="secondary")
                with gr.Column(scale=5, min_width=300):
                    gr.Markdown("### Ready to generate\nCheck your sketch, required features, and selected mode. You can revise any earlier tab before making a request.")
                    generate_button = gr.Button("Generate product concept", variant="primary", size="lg")
                    generation_status = gr.Markdown("Your generation status will appear here.")
                    quick_preview = gr.Image(label="Latest concept preview", interactive=False, height=350)
                    gr.Markdown("<div class='vf-footnote'>The generated render is a visual concept only. It is not validated CAD, a circuit schematic, a bill of materials, or a manufacturing-ready design.</div>")

        with gr.Tab("04 - Results & export"):
            gr.Markdown("## Review your concept\nCompare it against your starting sketch and constraints. Export the image and metadata to continue your project outside VisionForge.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, min_width=320):
                    result_image = gr.Image(label="Generated product concept", interactive=False, height=540)
                    download_bundle = gr.File(label="Concept package (.zip)")
                with gr.Column(scale=5, min_width=300):
                    status = gr.Markdown("Generate a concept in **03 Ã‚Â· Configure & generate** to see its report here.")
                    with gr.Accordion("Concept brief and engineering review", open=True):
                        brief = gr.Markdown("The design record will appear here after generation.")

        with gr.Tab("05 - Project library"):
            gr.Markdown("## Recent local concepts\nEach completed run is stored under `outputs/runs/` on this computer. Refresh the view to reload recent saved images.")
            refresh_history = gr.Button("Refresh library", variant="secondary")
            history = gr.Gallery(label="Saved concepts", columns=3, height=470, object_fit="contain")
            gr.Markdown("<div class='vf-footnote'>The library is stored locally. It is not a cloud project manager, and concepts are not uploaded anywhere by this gallery.</div>")

        with gr.Tab("07 - Experiment Lab"):
            gr.Markdown("## Compare generation modes\n\nCompare the prompt-only baseline against sketch-guided image-to-image using the local Stable Diffusion backend. Both generations run on this computer. No hosted inference provider, API token, or paid inference request is used.")
            gr.Markdown(
                "**Before running:** upload a sketch in **01 Â· Sketch** and complete the brief "
                "in **02 Â· Design brief**. This experiment uses those existing inputs."
            )
            run_experiment_button = gr.Button(
                "Run baseline vs sketch-guided experiment",
                variant="primary",
                size="lg",
            )
            experiment_gallery = gr.Gallery(
                label="Side-by-side experiment outputs",
                columns=2,
                height=480,
                object_fit="contain",
            )
            experiment_report = gr.Markdown(
                "Run an experiment to see timing, model names, and the edge-overlap proxy."
            )
            experiment_bundle = gr.File(label="Download experiment bundle (.zip)")
            gr.Markdown(
                "**Metric warning:** edge overlap is a rough image-processing proxy. "
                "It does not prove the generated product preserves dimensions or is technically correct."
            )
        with gr.Tab("Engineering & Local AI"):
            gr.Markdown("## Evidence-first engineering tools\nUse deterministic Python checks for measurable fit, then optionally connect local Ollama models for sketch inventory and revision suggestions. The checker works without a token, cloud inference, or Ollama.")

            with gr.Accordion("Constraint-aware component fit checker", open=True):
                gr.Markdown("Enter the **external enclosure size** and the expected component dimensions in millimeters. Each component is checked independently in six axis-aligned orientations; this does not pack components together or detect collisions. Must-have features are reused from tab 02, and feature coverage is manually marked below.")
                with gr.Row():
                    enclosure_length = gr.Number(label="Enclosure length (mm)", value=100, minimum=0.1)
                    enclosure_width = gr.Number(label="Enclosure width (mm)", value=45, minimum=0.1)
                    enclosure_height = gr.Number(label="Enclosure height (mm)", value=15, minimum=0.1)
                    component_clearance = gr.Number(label="Clearance per side (mm)", value=1.5, minimum=0)
                component_specs = gr.Textbox(
                    label="Component dimensions Ã¢â‚¬â€ one CSV row per part",
                    value="Display, 48, 28, 3\nBattery, 40, 25, 7\nPCB, 55, 30, 1.6",
                    lines=5,
                    info="Format: name, length_mm, width_mm, height_mm. These are estimates you provide, not extracted from the sketch.",
                )
                with gr.Row():
                    observed_features = gr.Textbox(
                        label="Features you manually confirmed in the concept",
                        placeholder="Display, USB-C port, two buttons",
                        lines=3,
                    )
                    human_visual_rating = gr.Slider(
                        label="Your visual review (1 = poor, 5 = strong)",
                        minimum=1, maximum=5, step=1, value=3,
                    )
                engineering_check_button = gr.Button("Run engineering screening", variant="primary")
                engineering_report = gr.Markdown("Engineering screening results will appear here.")
                engineering_download = gr.File(label="Download engineering review (.zip)")

            with gr.Accordion("Local AI status", open=True):
                gr.Markdown("Ollama is optional. The models listed here run on your computer. No API key is required, and the app will not download a model automatically.")
                refresh_local_ai_button = gr.Button("Check local Ollama models", variant="secondary")
                local_ai_status_output = gr.Markdown("Select **Check local Ollama models** to check whether Ollama is running and which models are installed.")

            with gr.Accordion("Sketch component inventory Ã¢â‚¬â€ optional Qwen2.5-VL", open=False):
                gr.Markdown("Uses the sketch already uploaded in **01 Ã‚Â· Sketch**. Suggested model: `qwen2.5vl:3b`. Results are model observations and may be mistaken; compare them with the drawing.")
                inspect_sketch_button = gr.Button("Inspect uploaded sketch locally")
                vision_inventory_output = gr.Textbox(label="Local vision findings (editable)", lines=12, interactive=True)

            with gr.Accordion("Prompt planning Ã¢â‚¬â€ optional Phi-4 Mini", open=False):
                gr.Markdown("Uses the existing design brief, optional Qwen findings, and a local Phi-4 Mini model to refine a prompt. This writes text only; it does not generate an image.")
                refine_prompt_button = gr.Button("Refine prompt with local Phi-4 Mini", variant="secondary")
                refined_local_prompt = gr.Textbox(label="Locally refined image-generation prompt", lines=12, interactive=True)

            with gr.Accordion("Self-improvement loop Ã¢â‚¬â€ review an existing concept", open=False):
                gr.Markdown("Upload a concept image you already have. Qwen can provide a cautious visual critique; Phi-4 Mini can draft a revision prompt. No new image is generated in this workflow.")
                concept_image_for_review = gr.Image(label="Concept image to review", type="filepath", sources=["upload", "clipboard"], height=300)
                review_concept_button = gr.Button("Review concept against my requirements")
                concept_critique_output = gr.Textbox(label="Local visual critique (editable)", lines=10, interactive=True)
                current_prompt_for_revision = gr.Textbox(label="Current image-generation prompt", placeholder="Paste the prompt that created this concept image.", lines=5)
                draft_revision_button = gr.Button("Draft targeted revision with Phi-4 Mini", variant="primary")
                revision_prompt_output = gr.Textbox(label="Revision prompt draft", lines=10, interactive=True)
            gr.Markdown("**Honesty note:** Qwen/Phi suggestions are model-generated and can be wrong. The numeric fit check and checklist calculations are Python rules; neither proves collision-free packaging, manufacturability, electrical correctness, or safety.")
        with gr.Tab("System & limits"):
            gr.Markdown("## Runtime and process notes")
            config = gr.Markdown(configuration_status())
            refresh_config = gr.Button("Refresh local configuration check")
            refresh_config.click(fn=configuration_status, outputs=config)
            with gr.Accordion("Image processing pipeline", open=False):
                gr.Markdown("""1. Validate file size, image dimensions, and minimum resolution.
2. Apply EXIF orientation and resize to a maximum longest edge of 1024 px.
3. Apply the selected Pillow-only cleanup profile.
4. Create an edge map and report basic image measurements.
5. In sketch-guided mode, send the processed image and structured prompt to the local Stable Diffusion img2img worker (not ControlNet).
6. Save the generated image, prompt, metadata, diagnostics, review checklist, and a ZIP bundle for each run.""")
            with gr.Accordion("Important limitations", open=True):
                gr.Markdown("""Image diagnostics are basic pixel measurements, not an AI engineering review. The generator may alter proportions, omit features, or invent internals. A generated render is **not** CAD, a circuit schematic, validated dimensions, a bill of materials, a safety assessment, or a manufacturing plan. Review all engineering details independently. Provider access, quotas, and pricing depend on the configured service.""")

    # Keep image diagnostics responsive no matter which tab currently contains the controls.
    sketch.change(
        fn=preview_processing,
        inputs=[sketch, profile],
        outputs=[processed_preview, edges_preview, analysis_report],
    )
    profile.change(
        fn=preview_processing,
        inputs=[sketch, profile],
        outputs=[processed_preview, edges_preview, analysis_report],
    )

    prompt_inputs = [
        sketch, name, description, form_factor, usability, lifecycle, engineering,
        required, envelope, avoid, render_style, generation_mode,
    ]
    refresh_prompt.click(fn=preview_prompt_ui, inputs=prompt_inputs, outputs=prompt_preview)
    for component in [name, description, form_factor, usability, lifecycle, engineering, required, envelope, avoid, render_style, generation_mode, sketch]:
        component.change(fn=preview_prompt_ui, inputs=prompt_inputs, outputs=prompt_preview)

    generate_button.click(
        fn=generate_from_studio,
        inputs=[
            sketch, name, description, form_factor, usability, lifecycle, engineering,
            required, envelope, avoid, render_style, generation_mode, profile, image_strength,
        ],
        outputs=[result_image, status, brief, download_bundle, history, quick_preview, generation_status],
    )
    engineering_check_button.click(
        fn=evaluate_engineering_design_ui,
        inputs=[
            enclosure_length, enclosure_width, enclosure_height, component_clearance,
            component_specs, observed_features, human_visual_rating,
            description, form_factor, usability, lifecycle, engineering,
            required, envelope, avoid,
        ],
        outputs=[engineering_report, engineering_download],
    )
    refresh_local_ai_button.click(fn=local_runtime_status_ui, outputs=local_ai_status_output)
    inspect_sketch_button.click(
        fn=inspect_sketch_locally_ui,
        inputs=[sketch, profile, description, required, envelope, avoid],
        outputs=vision_inventory_output,
    )
    refine_prompt_button.click(
        fn=refine_prompt_locally_ui,
        inputs=[name, description, form_factor, usability, lifecycle, engineering, required, envelope, avoid, render_style, vision_inventory_output],
        outputs=refined_local_prompt,
    )
    review_concept_button.click(
        fn=critique_concept_locally_ui,
        inputs=[concept_image_for_review, description, required, envelope, avoid],
        outputs=concept_critique_output,
    )
    draft_revision_button.click(
        fn=draft_revision_prompt_locally_ui,
        inputs=[description, required, envelope, avoid, current_prompt_for_revision, concept_critique_output],
        outputs=revision_prompt_output,
    )
    refresh_history.click(fn=load_history, outputs=history)
    run_experiment_button.click(
        fn=run_experiment_from_studio,
        inputs=[
            sketch, name, description,
            form_factor, usability, lifecycle, engineering,
            required, envelope, avoid, render_style, profile,
        ],
        outputs=[experiment_gallery, experiment_report, experiment_bundle],
    )
    demo.load(fn=load_history, outputs=history)


if __name__ == "__main__":
    demo.queue(max_size=6).launch(
        theme=gr.themes.Soft(),
        css=CSS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
