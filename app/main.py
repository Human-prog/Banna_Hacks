import os
import time
from pathlib import Path
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image, ImageOps, ImageFilter

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")


def edge_preview(sketch_path):
    if not sketch_path:
        return None
    image = Image.open(sketch_path).convert("RGB")
    image.thumbnail((768, 768))
    edges = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES)
    return ImageOps.autocontrast(edges)


def generate_concept(sketch, name, description, priorities, constraints):
    if not sketch:
        raise gr.Error("Upload your engineering sketch first.")
    if not (description or "").strip():
        raise gr.Error("Describe the product first.")

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise gr.Error("HF_TOKEN is missing from your .env file.")

    name = (name or "Untitled concept").strip()
    priority_text = ", ".join(priorities or []) or "Balanced practical design"
    constraints = (constraints or "").strip() or "No extra constraints specified"

    prompt = f"""
Professional industrial design concept render of {name}.
Purpose: {description.strip()}.
Design priorities: {priority_text}.
Constraints: {constraints}.
Create a single isolated consumer electronics product in a three-quarter view,
with realistic materials, coherent proportions, visible practical controls,
and a clean neutral studio background. No labels, text, or watermark.
""".strip()

    try:
        client = InferenceClient(
            provider="fal-ai",
            api_key=token,
            timeout=180,
        )
        started = time.time()
        image = client.text_to_image(
            prompt,
            model=MODEL_ID,
        )
        elapsed = time.time() - started

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"concept_{timestamp}.png"
        image.save(output_path)

        brief = f"""# {name}

## Product intent
{description}

## Design priorities
{priority_text}

## Constraints
{constraints}

## Generation record
- Model: `{MODEL_ID}`
- Provider: `fal-ai`
- Generation time: {elapsed:.1f} seconds
- Method: Prompt-only text-to-image
- Image: `{output_path.as_posix()}`

## Engineering review
- [ ] Check that the main silhouette follows the sketch.
- [ ] Review control and port placement.
- [ ] Check dimensions and material assumptions.
- [ ] Validate electrical and mechanical details independently.

**Limitation:** This is a visual concept, not validated CAD or a verified engineering design.
"""
        status = f"""### Generation successful

- **Model:** `{MODEL_ID}`
- **Provider:** `fal-ai`
- **Time:** {elapsed:.1f} seconds
- **Method:** Prompt-only text-to-image
- **Saved image:** `{output_path.as_posix()}`

The sketch preview is shown for reference. It is not used to condition this generation.
"""
        return str(output_path), edge_preview(sketch), status, brief

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {type(exc).__name__}: {exc}")


with gr.Blocks(title="VisionForge AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# ? VisionForge AI
### From rough engineering sketch to product concept

Generate an AI concept image, document the design constraints, and prepare the result for evaluation.
""")

    with gr.Row():
        with gr.Column():
            sketch = gr.Image(
                label="Engineering sketch",
                type="filepath",
                sources=["upload", "clipboard"],
            )
            name = gr.Textbox(
                label="Product name",
                placeholder="Pocket-sized music player",
            )
            description = gr.Textbox(
                label="Product description",
                placeholder="What does it do, and who is it for?",
                lines=4,
            )
            priorities = gr.CheckboxGroup(
                label="Design priorities",
                choices=[
                    "Compact",
                    "Low cost",
                    "Easy to repair",
                    "Accessible controls",
                    "Low power",
                    "Modular",
                    "Portable",
                ],
            )
            constraints = gr.Textbox(
                label="Constraints",
                placeholder="USB-C, physical buttons, replaceable microSD card",
                lines=2,
            )
            generate = gr.Button("Generate product concept", variant="primary")

        with gr.Column():
            result_image = gr.Image(label="Generated concept")
            edges_image = gr.Image(label="Sketch edge preview")
            status = gr.Markdown("Your generation status will appear here.")
            brief = gr.Markdown("Your design brief will appear here.")

    generate.click(
        fn=generate_concept,
        inputs=[sketch, name, description, priorities, constraints],
        outputs=[result_image, edges_image, status, brief],
    )

    gr.Markdown("""
---
**Important:** Generated concepts may invent or misplace features. They are not verified CAD,
electrical schematics, manufacturing plans, or safety assessments.
""")

if __name__ == "__main__":
    demo.launch()

