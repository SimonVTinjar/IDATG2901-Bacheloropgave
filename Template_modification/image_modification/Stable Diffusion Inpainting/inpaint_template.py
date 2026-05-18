from pathlib import Path
from PIL import Image
import json
import torch
from diffusers import StableDiffusionInpaintPipeline


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

TEMPLATE_PATH = BASE_DIR.parent / "Prob Done v2.png"
MASK_PATH = BASE_DIR.parent / "template_mask.png"
JSON_PATH = BASE_DIR.parent / "template_boxes.json"

OUTPUT_PATH = BASE_DIR / "ai_edited_output.png"

# Vanlig Stable Diffusion inpainting.
# NB: Denne redigerer bildet, men er ikke perfekt på eksakt tekst.
MODEL_NAME = "runwayml/stable-diffusion-inpainting"


# -------------------------------------------------
# PROMPT
# -------------------------------------------------

def build_prompt(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = []

    for region in data["boxes"]:
        label = region.get("label", "field")
        value = region.get("value", "")
        bbox = region.get("bbox", [])

        if value:
            parts.append(f"{label} at bbox {bbox} should contain '{value}'")

    values_text = "; ".join(parts)

    prompt = (
        "Edit only the white masked regions in the image. "
        "Do not change anything outside the mask. "
        "Fill each masked field with realistic content matching the original document style, "
        "same font style, same color, same lighting, same texture, same perspective. "
        f"Required field contents: {values_text}. "
        "The result should look like a natural edited version of the same image."
    )

    return prompt


def main():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Fant ikke templatebildet: {TEMPLATE_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Fant ikke masken: {MASK_PATH}")

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Fant ikke JSON-filen: {JSON_PATH}")

    image = Image.open(TEMPLATE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("RGB")

    if mask.size != image.size:
        print(f"Resizer mask fra {mask.size} til {image.size}")
        mask = mask.resize(image.size)

    prompt = build_prompt(JSON_PATH)

    negative_prompt = (
        "changes outside mask, changed background, changed layout, extra boxes, "
        "extra text outside mask, blurry, distorted, artifacts, wrong perspective"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Laster inpainting-modell...")
    print("Device:", device)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)

    # Mindre minnebruk
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    print("Prompt:")
    print(prompt)
    print()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8.0,
        num_inference_steps=40,
        strength=0.95
    ).images[0]

    result.save(OUTPUT_PATH)

    print("Ferdig.")
    print("Lagret redigert bilde:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
