"""VisionForge AI — sketch-to-concept studio."""

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
from huggingface_hub import InferenceClient
from PIL import Image

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
load_dotenv(ENV_FILE)

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

CSS = """
.gradio-container {max-width: 1440px !important; margin: 0 auto !important;}
#vf-hero {padding: 22px 24px; border: 1px solid var(--border-color-primary); border-radius: 18px; margin-bottom: 12px;}
#vf-hero h1 {letter-spacing: -0.04em; margin-bottom: 4px;}
.vf-muted {opacity: 0.78;}
"""


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
        f"- **Input:** {report['source_format']} · {report['source_width']} × {report['source_height']} px · "
        f"{report['file_size_mb']} MB\n"
        f"- **Prepared image:** {report['processed_width']} × {report['processed_height']} px "
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
            caption = f"{metadata.get('product_name', 'Concept')} · {metadata.get('created_at', '')}"
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

> VisionForge AI · Product concept record

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
- **Output dimensions:** {metadata['output_width']} × {metadata['output_height']} px
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
) -> tuple[str | None, str, str, str | None, list[tuple[str, str]]]:
    description = (description or "").strip()
    if not description:
        raise gr.Error("Describe what the product does before generating a concept.")

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise gr.Error("HF_TOKEN is missing. Add it to the local .env file; never paste it into the app or commit it to Git.")

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
    prompt = build_prompt(product_name, description, priorities, constraints or "", render_style, is_guided)
    model = EDIT_MODEL if is_guided else TEXT_MODEL

    try:
        client = InferenceClient(provider=PROVIDER, api_key=token, timeout=240)
        started = time.perf_counter()
        if is_guided:
            assert prepared is not None
            generated = client.image_to_image(prepared.processed, prompt=prompt, model=model)
        else:
            generated = client.text_to_image(prompt=prompt, model=model)
        elapsed = time.perf_counter() - started
        if not isinstance(generated, Image.Image):
            raise RuntimeError("The inference provider returned an unexpected image format.")
        if generated.width < 1 or generated.height < 1:
            raise RuntimeError("The inference provider returned an empty image.")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if is_guided:
            hint = (
                " Check that your Hugging Face token has Inference Providers permission and that the selected "
                f"model ({EDIT_MODEL}) is available through {PROVIDER}. You can switch to Prompt-only fallback "
                "to test text-to-image separately. Provider quotas or billing may apply."
            )
        else:
            hint = (
                " Check that your Hugging Face token has Inference Providers permission and that the selected "
                f"model ({TEXT_MODEL}) is available through {PROVIDER}. Provider quotas or billing may apply."
            )
        # Do not print or attach the token to errors, logs, or metadata.
        if token:
            detail = detail.replace(token, "[redacted token]")
        raise gr.Error(f"Generation failed — {detail}.{hint}") from exc

    metadata: dict[str, Any] = {
        "product_name": product_name,
        "description": description,
        "priorities": priorities or [],
        "constraints": (constraints or "").strip(),
        "render_style": render_style,
        "generation_mode": generation_mode,
        "model": model,
        "provider": PROVIDER,
        "generation_seconds": round(elapsed, 2),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt": prompt,
        "sketch_report": prepared.report if prepared else None,
        "notice": "Visual concept only; not verified engineering or manufacturing documentation.",
    }
    image_path, bundle_path, brief = _save_export(generated, metadata, prepared)

    if is_guided:
        mode_text = "Sketch-guided image-to-image: the processed sketch was sent as the reference image."
    else:
        mode_text = "Prompt-only text-to-image: the generated image was not conditioned on the uploaded sketch."
    status = f"""### Concept generated successfully

- **Model:** `{model}`
- **Provider:** `{PROVIDER}`
- **Time:** {elapsed:.1f} seconds
- **Output:** {metadata['output_width']} × {metadata['output_height']} px
- **Mode:** {mode_text}
- **Saved to:** `outputs/runs/{image_path.parent.name}/`

**Engineering note:** Review the output against the sketch and constraints. The concept is not validated CAD or a verified electronics design.
"""
    return str(image_path), status, brief, str(bundle_path), load_history()


def configuration_status() -> str:
    token_exists = bool(os.getenv("HF_TOKEN", "").strip())
    token_line = "Token detected in environment; its value is never displayed." if token_exists else "**HF_TOKEN is not configured.** Add it to your local `.env` file."
    return (
        "### Runtime configuration\n"
        f"- **Provider:** `{PROVIDER}`\n"
        f"- **Sketch-guided model:** `{EDIT_MODEL}`\n"
        f"- **Prompt-only model:** `{TEXT_MODEL}`\n"
        f"- **Credential:** {token_line}\n\n"
        "This is a local configuration check only; it does not make an inference request. Provider availability, quotas, and pricing can change."
    )


with gr.Blocks(title="VisionForge AI · Concept Studio", css=CSS) as demo:
    gr.Markdown(
        """<div id="vf-hero">
# ◈ VisionForge AI
**SKETCH → PROCESS → CONCEPT**

Turn a rough hardware sketch and a design brief into a polished visual concept. Inspect image preprocessing, keep constraints attached to the result, and export a review-ready bundle.
</div>"""
    )

    with gr.Tabs():
        with gr.Tab("Concept studio"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320):
                    gr.Markdown("### 01 / Sketch input")
                    sketch = gr.Image(
                        label="Engineering sketch",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )
                    profile = gr.Dropdown(
                        label="Preprocessing profile",
                        choices=list(PROFILES),
                        value="Clean sketch",
                        info="Changes affect the reference image used for sketch-guided generation.",
                    )
                    with gr.Accordion("Inspect image processing", open=True):
                        with gr.Row():
                            processed_preview = gr.Image(label="Prepared sketch", interactive=False, height=210)
                            edges_preview = gr.Image(label="Edge map", interactive=False, height=210)
                        analysis_report = gr.Markdown("Upload a sketch to inspect preprocessing and linework.")

                    gr.Markdown("### 02 / Product brief")
                    name = gr.Textbox(label="Product name", placeholder="e.g. Pocket-sized music player", max_lines=1)
                    description = gr.Textbox(
                        label="What are you designing?",
                        placeholder="Describe the purpose, user, and main functions.",
                        lines=4,
                    )
                    priorities = gr.CheckboxGroup(
                        label="Design priorities",
                        choices=[
                            "Compact", "Low cost", "Easy to repair", "Accessible controls",
                            "Low power", "Modular", "Portable", "Durable", "Minimal parts",
                        ],
                    )
                    constraints = gr.Textbox(
                        label="Hard constraints / required features",
                        placeholder="e.g. USB-C, physical buttons, replaceable microSD, 100 × 45 mm envelope",
                        lines=3,
                    )
                    render_style = gr.Dropdown(
                        label="Visualization style",
                        choices=list(STYLE_GUIDES),
                        value="Studio product render",
                    )
                    generation_mode = gr.Radio(
                        label="Generation mode",
                        choices=[GUIDED_MODE, PROMPT_MODE],
                        value=GUIDED_MODE,
                        info="Sketch-guided mode sends the processed image to an image-to-image model. Prompt-only is a fallback.",
                    )
                    generate_button = gr.Button("Generate concept", variant="primary", size="lg")

                with gr.Column(scale=6, min_width=360):
                    gr.Markdown("### 03 / Generated concept")
                    result_image = gr.Image(label="Concept render", interactive=False, height=460)
                    status = gr.Markdown("Complete the product brief and generate a concept.")
                    download_bundle = gr.File(label="Download concept package (.zip)")
                    with gr.Accordion("Concept brief and engineering review", open=True):
                        brief = gr.Markdown("The generated design brief will appear here.")

            gr.Markdown("### Recent concepts")
            history = gr.Gallery(label="Previous local generations", columns=3, height=320)

        with gr.Tab("Runtime & limitations"):
            gr.Markdown(
                """### Runtime configuration

The app reads `HF_TOKEN` from the local `.env` file. It does not show the token or include it in exports.
The default provider is `fal-ai`; sketch-guided and prompt-only generation use separate configurable model IDs.

### Image-processing pipeline

1. Validate file size, image dimensions, and minimum resolution.
2. Apply EXIF orientation and resize to a maximum longest edge of 1024 px.
3. Apply the selected Pillow-only cleanup profile.
4. Create an edge map and report image dimensions, grayscale contrast, and estimated edge density.
5. In sketch-guided mode, send the processed sketch and structured prompt to the image-to-image endpoint.
6. Save the image, prompt, metadata, diagnostics, review checklist, and a ZIP bundle for each run.

### Important limits

Image diagnostics are basic pixel measurements, not AI engineering review. The generator may alter proportions, omit features, or invent internals. A generated render is **not** CAD, a circuit schematic, validated dimensions, a bill of materials, or a manufacturing plan. Review all engineering details independently. Model/provider access, quotas, and pricing depend on the configured service.
"""
            )
            config = gr.Markdown(configuration_status())
            refresh_config = gr.Button("Refresh local configuration check")
            refresh_config.click(fn=configuration_status, outputs=config)

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
    generate_button.click(
        fn=generate_concept,
        inputs=[sketch, name, description, priorities, constraints, render_style, generation_mode, profile],
        outputs=[result_image, status, brief, download_bundle, history],
    )
    demo.load(fn=load_history, outputs=history)


if __name__ == "__main__":
    demo.queue(max_size=6).launch(
        theme=gr.themes.Soft(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
