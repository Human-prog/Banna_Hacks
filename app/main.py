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
    profile: str,
):
    """Adapt grouped UI controls to the core generation pipeline."""
    return generate_concept(
        sketch_path,
        name,
        description,
        _selected_priorities(form_factor, usability, lifecycle, engineering),
        _combine_constraints(required, envelope, avoid),
        render_style,
        generation_mode,
        profile,
    )


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

with gr.Blocks(title="VisionForge AI · Concept Studio") as demo:
    gr.Markdown(
        """<div id="vf-hero">
<div id="vf-kicker">HARDWARE CONCEPT LAB / V 0.2</div>
# ◈ VisionForge AI
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
        with gr.Tab("01 · Sketch"):
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
            gr.Markdown("**Next:** Open **02 · Design brief** and describe what the product needs to do.")

        with gr.Tab("02 · Design brief"):
            gr.Markdown("## Define the product\nBe concrete. State what the product does, who uses it, and which features must remain present.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=280):
                    name = gr.Textbox(label="Project / product name", placeholder="e.g. Pocket-sized music player", max_lines=1)
                    description = gr.Textbox(
                        label="Product purpose and main functions",
                        placeholder="What is it? Who is it for? What should it do?",
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
                        placeholder="Approx. 100 × 45 mm, side-mounted headphone jack",
                        lines=3,
                    )
                    avoid = gr.Textbox(
                        label="Avoid / extra notes",
                        placeholder="No touchscreen, no glued-shut enclosure",
                        lines=3,
                    )
            gr.Markdown("**Next:** Open **03 · Configure** to choose the render style and review the prompt.")

        with gr.Tab("03 · Configure & generate"):
            gr.Markdown("## Configure the generation\nReview the prompt before sending a request to the configured inference provider. Provider access and quotas may apply.")
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
                        info="Sketch-guided sends the prepared sketch to image-to-image. Prompt-only generates from text and does not use the sketch as a model input.",
                    )
                    prompt_preview = gr.Textbox(
                        label="Prompt preview",
                        value="Complete the design brief to preview the prompt.",
                        lines=13,
                        interactive=False,
                    )
                    refresh_prompt = gr.Button("Refresh prompt preview", variant="secondary")
                with gr.Column(scale=5, min_width=300):
                    gr.Markdown("### Ready to generate?\nCheck your sketch, required features, and selected mode. You can revise any earlier tab before making a request.")
                    generate_button = gr.Button("Generate product concept", variant="primary", size="lg")
                    generation_status = gr.Markdown("Your generation status will appear here.")
                    quick_preview = gr.Image(label="Latest concept preview", interactive=False, height=350)
                    gr.Markdown("<div class='vf-footnote'>The generated render is a visual concept only. It is not validated CAD, a circuit schematic, a bill of materials, or a manufacturing-ready design.</div>")

        with gr.Tab("04 · Results & export"):
            gr.Markdown("## Review your concept\nCompare it against your starting sketch and constraints. Export the image and metadata to continue your project outside VisionForge.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, min_width=320):
                    result_image = gr.Image(label="Generated product concept", interactive=False, height=540)
                    download_bundle = gr.File(label="Concept package (.zip)")
                with gr.Column(scale=5, min_width=300):
                    status = gr.Markdown("Generate a concept in **03 · Configure & generate** to see its report here.")
                    with gr.Accordion("Concept brief and engineering review", open=True):
                        brief = gr.Markdown("The design record will appear here after generation.")

        with gr.Tab("05 · Project library"):
            gr.Markdown("## Recent local concepts\nEach completed run is stored under `outputs/runs/` on this computer. Refresh the view to reload recent saved images.")
            refresh_history = gr.Button("Refresh library", variant="secondary")
            history = gr.Gallery(label="Saved concepts", columns=3, height=470, object_fit="contain")
            gr.Markdown("<div class='vf-footnote'>The library is stored locally. It is not a cloud project manager, and concepts are not uploaded anywhere by this gallery.</div>")

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
5. In sketch-guided mode, send the processed image and structured prompt to the configured image-to-image endpoint.
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
            required, envelope, avoid, render_style, generation_mode, profile,
        ],
        outputs=[result_image, status, brief, download_bundle, history, quick_preview, generation_status],
    )
    refresh_history.click(fn=load_history, outputs=history)
    demo.load(fn=load_history, outputs=history)


if __name__ == "__main__":
    demo.queue(max_size=6).launch(
        theme=gr.themes.Soft(),
        css=CSS,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
