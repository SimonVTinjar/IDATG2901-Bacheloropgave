from pathlib import Path
from PIL import Image, ImageDraw
import json
import torch
from diffusers import StableDiffusionInpaintPipeline


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

TEMPLATE_PATH = BASE_DIR.parent / "Prob Done v2.png"
JSON_PATH = BASE_DIR.parent / "template_named_boxes.json"

OUTPUT_PATH = BASE_DIR / "ai_edited_output_per_box.png"
DEBUG_CROP_DIR = BASE_DIR / "debug_ai_crops"

MODEL_NAME = "runwayml/stable-diffusion-inpainting"

# Hvor mye ekstra område rundt boksen modellen får se.
# Øk hvis resultatet trenger mer kontekst.
CROP_MARGIN = 80

# Stable Diffusion liker størrelser som er delelige på 8.
MULTIPLE_OF = 8

# For store crops kan GPU/CPU bli treg.
# Hvis crop blir større enn dette, skaleres den ned under AI-editing
# og skaleres tilbake før den limes inn.
MAX_AI_SIDE = 768

# Hvor sterkt modellen skal endre masken.
STRENGTH = 0.95

# Hvis True, lagrer debug-crops.
SAVE_DEBUG_CROPS = True


# -------------------------------------------------
# HJELPEFUNKSJONER
# -------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value, low, high):
    return max(low, min(high, value))


def expand_bbox(bbox, image_size, margin):
    x1, y1, x2, y2 = bbox
    w, h = image_size

    return [
        clamp(x1 - margin, 0, w),
        clamp(y1 - margin, 0, h),
        clamp(x2 + margin, 0, w),
        clamp(y2 + margin, 0, h),
    ]


def make_crop_mask(crop_size, bbox, crop_box, padding=8):
    """
    Lager lokal mask for crop.
    Svart = ikke endre.
    Hvit = området som skal redigeres.
    """
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = bbox

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    cw, ch = crop_size
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)

    lx1 = clamp(lx1 - padding, 0, cw)
    ly1 = clamp(ly1 - padding, 0, ch)
    lx2 = clamp(lx2 + padding, 0, cw)
    ly2 = clamp(ly2 + padding, 0, ch)

    draw.rectangle([lx1, ly1, lx2, ly2], fill=255)
    return mask


def resize_for_ai(image, mask, multiple_of=8, max_side=768):
    """
    Skalerer crop til en størrelse som diffusion-modellen tåler.
    Returnerer image_for_ai, mask_for_ai, original_size.
    """
    original_size = image.size
    w, h = original_size

    scale = 1.0
    largest = max(w, h)

    if largest > max_side:
        scale = max_side / largest

    new_w = max(multiple_of, int(w * scale))
    new_h = max(multiple_of, int(h * scale))

    new_w = (new_w // multiple_of) * multiple_of
    new_h = (new_h // multiple_of) * multiple_of

    if new_w < multiple_of:
        new_w = multiple_of
    if new_h < multiple_of:
        new_h = multiple_of

    if (new_w, new_h) != original_size:
        image = image.resize((new_w, new_h), Image.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

    return image, mask, original_size


def build_prompt(region):
    label = region.get("label", "field")
    value = region.get("value", "")

    if value:
        content = f"The masked {label} field must contain exactly: {value}."
    else:
        content = f"Fill the masked {label} field with realistic content."

    prompt = (
        "Edit only the masked region. "
        "Keep all unmasked pixels unchanged. "
        f"{content} "
        "Match the same document style, same ink color, same font style, "
        "same texture, same lighting, same perspective, and same background. "
        "The result should look like the original document was naturally printed this way."
    )

    return prompt


def paste_only_masked_area(base_img, edited_crop, crop_box, mask):
    """
    Limer bare maskerte piksler tilbake i fullbildet.
    Dermed endres ikke resten av bildet.
    """
    x1, y1, x2, y2 = crop_box

    if edited_crop.size != (x2 - x1, y2 - y1):
        edited_crop = edited_crop.resize((x2 - x1, y2 - y1), Image.LANCZOS)

    if mask.size != edited_crop.size:
        mask = mask.resize(edited_crop.size, Image.NEAREST)

    base_img.paste(edited_crop, (x1, y1), mask)
    return base_img


# -------------------------------------------------
# HOVEDPROGRAM
# -------------------------------------------------

def main():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Fant ikke template: {TEMPLATE_PATH}")

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Fant ikke JSON: {JSON_PATH}")

    DEBUG_CROP_DIR.mkdir(exist_ok=True)

    data = load_json(JSON_PATH)
    regions = data.get("boxes", [])

    if not regions:
        raise ValueError("Fant ingen boxes i template_boxes.json")

    full_image = Image.open(TEMPLATE_PATH).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Laster Stable Diffusion inpainting...")
    print("Device:", device)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    result_full = full_image.copy()

    negative_prompt = (
        "changed unmasked area, changed background, changed layout, extra objects, "
        "extra text outside the mask, blurry, distorted, artifacts, low quality"
    )

    print(f"Redigerer {len(regions)} bokser, én og én.")

    for i, region in enumerate(regions):
        bbox = region["bbox"]
        label = region.get("label", "field")
        value = region.get("value", "")

        crop_box = expand_bbox(bbox, full_image.size, CROP_MARGIN)
        x1, y1, x2, y2 = crop_box

        crop = result_full.crop(crop_box)
        mask = make_crop_mask(crop.size, bbox, crop_box, padding=8)

        prompt = build_prompt(region)

        crop_for_ai, mask_for_ai, original_crop_size = resize_for_ai(
            crop,
            mask,
            multiple_of=MULTIPLE_OF,
            max_side=MAX_AI_SIDE
        )

        if SAVE_DEBUG_CROPS:
            crop.save(DEBUG_CROP_DIR / f"{i:03d}_{label}_input.png")
            mask.save(DEBUG_CROP_DIR / f"{i:03d}_{label}_mask.png")

        print(f"[{i+1}/{len(regions)}] {label} -> {value} | crop={crop.size} ai={crop_for_ai.size}")

        edited = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop_for_ai,
            mask_image=mask_for_ai.convert("RGB"),
            guidance_scale=8.0,
            num_inference_steps=40,
            strength=STRENGTH
        ).images[0]

        if edited.size != original_crop_size:
            edited = edited.resize(original_crop_size, Image.LANCZOS)

        if SAVE_DEBUG_CROPS:
            edited.save(DEBUG_CROP_DIR / f"{i:03d}_{label}_edited.png")

        # Lim bare masken tilbake, ikke hele cropen.
        result_full = paste_only_masked_area(
            base_img=result_full,
            edited_crop=edited,
            crop_box=crop_box,
            mask=mask
        )

    result_full.save(OUTPUT_PATH)

    print()
    print("Ferdig.")
    print("Lagret:", OUTPUT_PATH)
    print("Debug crops:", DEBUG_CROP_DIR)


if __name__ == "__main__":
    main()
