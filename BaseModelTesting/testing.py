import os, json, time, re
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from huggingface_hub import hf_hub_download

# Optional OCR/scoring
USE_OCR = True
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    USE_OCR = False

try:
    from rapidfuzz.distance import Levenshtein
    def cer(pred: str, target: str) -> float:
        if len(target) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return Levenshtein.distance(pred, target) / max(1, len(target))
except Exception:
    def cer(pred: str, target: str) -> float:
        return float("nan")


@dataclass
class ModelSpec:
    name: str
    repo: str
    kind: str  # "sd15" or "sdxl"
    single_file: Optional[str] = None


PROMPTS: List[Dict[str, str]] = [
    {
        "id": "combined_prompt_original",
        "prompt": (
            "Top-down photo of white paper on a table. Clean handwriting: "
            "“Dette er en test av tekst. Alle bokstaver må være riktige.” "
        ),
        "target_text": (
            "Dette er en test av tekst.\n"
            "Alle bokstaver må være riktige.\n\n"
        ),
    }
]

MODELS: List[ModelSpec] = [
    ModelSpec("sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "sd15"),
    ModelSpec("epicrealism", "emilianJR/epiCRealism", "sd15"),
    ModelSpec("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl"),
    ModelSpec("juggernaut_xl_v9", "RunDiffusion/Juggernaut-XL-v9", "sdxl", single_file="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"),
    # FLUX not included in this script (separate script below)
]

# ======= Tuning defaults =======
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

DEFAULTS = {
    "sd15": {"height": 512, "width": 768, "steps": 35, "cfg": 7.0},
    "sdxl": {"height": 832, "width": 1216, "steps": 35, "cfg": 5.5},
}

NEGATIVE = (
    "lowres, blurry, illegible text, gibberish, misspelled, watermark, logo, "
    "extra text, distorted letters, artifacts, perspective, shadow"
)


def sanitize(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120]


def load_pipe(model: ModelSpec):
    torch_dtype = torch.float16

    if model.kind == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            model.repo,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    elif model.kind == "sdxl":
        if model.single_file:
            ckpt_path = hf_hub_download(
                repo_id=model.repo,
                filename=model.single_file,
            )
            pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model.repo,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

        # SDXL black-image fix: decode with fp32 VAE
        pipe.vae.config.force_upcast = True

    else:
        raise ValueError(f"Unknown model kind: {model.kind}")

    pipe = pipe.to("cuda")

    # Memory/perf helpers
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.enable_attention_slicing()

    # New API (avoid deprecated enable_vae_slicing)
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    return pipe

def run_ocr(img: Image.Image) -> str:
    if not USE_OCR:
        return ""
    # Improve OCR readability by upscaling a bit
    w, h = img.size
    img2 = img.resize((int(w * 1.5), int(h * 1.5)))
    text = pytesseract.image_to_string(img2, lang="eng")  # eng works OK for ÆØÅ sometimes; depends on your tesseract
    # Normalize whitespace
    text = text.replace("\r\n", "\n").strip()
    return text


def main():
    out_dir = "runs"
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, f"results_{int(time.time())}.jsonl")
    print(f"Writing results to: {results_path}")
    print(f"OCR enabled: {USE_OCR}")

    for m in MODELS:
        print(f"\n=== Loading {m.name}: {m.repo} ===")
        pipe = load_pipe(m)

        d = DEFAULTS[m.kind]
        height, width, steps, cfg = d["height"], d["width"], d["steps"], d["cfg"]

        model_dir = os.path.join(out_dir, sanitize(m.name))
        os.makedirs(model_dir, exist_ok=True)

        for p in PROMPTS:
            prompt_id = p["id"]
            prompt = p["prompt"]
            target = p["target_text"]

            prompt_dir = os.path.join(model_dir, sanitize(prompt_id))
            os.makedirs(prompt_dir, exist_ok=True)

            for seed in tqdm(SEEDS, desc=f"{m.name}:{prompt_id}"):
                gen = torch.Generator(device="cuda").manual_seed(seed)

                with torch.inference_mode():
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        height=height,
                        width=width,
                        generator=gen,
                    ).images[0]

                img_name = f"{sanitize(prompt_id)}_seed{seed}.png"
                img_path = os.path.join(prompt_dir, img_name)
                image.save(img_path)

                ocr_text = run_ocr(image)
                score = cer(ocr_text, target) if USE_OCR else float("nan")

                rec = {
                    "model": m.name,
                    "repo": m.repo,
                    "kind": m.kind,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "height": height,
                    "width": width,
                    "steps": steps,
                    "cfg": cfg,
                    "image_path": img_path,
                    "target_text": target,
                    "ocr_text": ocr_text,
                    "cer": score,
                }

                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Free VRAM between models
        del pipe
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
