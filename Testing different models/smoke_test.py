import os
import json
import torch

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

# Disse importene finnes i diffusers 0.36.0, men noen modeller kan fortsatt være gated
from diffusers import StableDiffusion3Pipeline, FluxPipeline, ZImagePipeline

PROMPT = "photorealistic car on a winding forest road, light rain, wet asphalt reflections, overcast sky"
NEG = "cgi, render, 3d, cartoon, illustration, text, watermark, logo, deformed, bad geometry"

OUT_DIR = "smoke_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def cuda_ok() -> bool:
    return torch.cuda.is_available()

def pick_dtype(prefer: str | None):
    """
    prefer: "bf16" / "fp16" / None
    RTX 40-series støtter bf16, men fp16 er ofte tryggest.
    """
    if prefer == "bf16" and cuda_ok():
        return torch.bfloat16
    return torch.float16

def make_generator(device: str, seed: int = 0):
    if device == "cuda":
        return torch.Generator("cuda").manual_seed(seed)
    return torch.Generator("cpu").manual_seed(seed)

def save_img(img, model_key: str):
    out_path = os.path.join(OUT_DIR, f"{model_key}.png")
    img.save(out_path)
    print(f"✅ saved: {out_path}")

def load_and_run(model_key: str, model_id: str, family: str, dtype_pref: str | None):
    device = "cuda" if cuda_ok() else "cpu"
    dtype = pick_dtype(dtype_pref)
    gen = make_generator(device, seed=0)

    print(f"\n=== {model_key} ({family}) ===")
    print(f"model_id: {model_id}")
    print(f"device: {device}, dtype: {dtype}")

    try:
        if family == "sd1x" or family == "sd1x_finetune":
            # SD1.x / fine-tunes
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,  # valgfritt (kan fjernes)
            ).to(device)

            img = pipe(
                PROMPT,
                negative_prompt=NEG,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=gen,
                height=512,
                width=768,
            ).images[0]
            save_img(img, model_key)

        elif family == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)

            img = pipe(
                PROMPT,
                negative_prompt=NEG,
                num_inference_steps=30,
                guidance_scale=4.5,
                generator=gen,
                height=768,
                width=1024,
            ).images[0]
            save_img(img, model_key)

        elif family == "sd3":
            # SD3 / SD3.5
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
            if device == "cuda":
                pipe = pipe.to("cuda")

            img = pipe(
                prompt=PROMPT,
                negative_prompt=NEG,
                num_inference_steps=28,
                guidance_scale=6.0,
                generator=make_generator("cpu", 0),  # SD3 kan være CPU-gen OK
                height=1024,
                width=1024,
            ).images[0]
            save_img(img, model_key)

        elif family == "flux":
            # FLUX: ofte gated; CPU-offload hjelper mye på 4070 Ti
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            if device == "cuda":
                pipe.enable_model_cpu_offload()

            img = pipe(
                PROMPT,
                guidance_scale=3.5,
                num_inference_steps=30,
                max_sequence_length=256,
                generator=make_generator("cpu", 0),
                height=768,
                width=1024,
            ).images[0]
            save_img(img, model_key)

        elif family == "other":
            # Z-Image-Turbo kan gå via ZImagePipeline; andre "other" prøver vi AutoPipeline først.
            if "Z-Image" in model_id or "Z-Image-Turbo" in model_id:
                pipe = ZImagePipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                ).to(device)

                img = pipe(
                    PROMPT,
                    num_inference_steps=8,
                    guidance_scale=3.5,
                    generator=gen,
                    height=768,
                    width=1024,
                ).images[0]
                save_img(img, model_key)
            else:
                # Qwen/Qwen-Image osv – prøv AutoPipeline
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                ).to(device)

                img = pipe(
                    PROMPT,
                    negative_prompt=NEG,
                    num_inference_steps=30,
                    guidance_scale=4.5,
                    generator=gen,
                    height=768,
                    width=1024,
                ).images[0]
                save_img(img, model_key)

        else:
            raise ValueError(f"Unknown family: {family}")

    except Exception as e:
        print(f"❌ failed: {type(e).__name__}: {e}")

def main():
    with open("models.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for m in data["models"]:
        load_and_run(
            model_key=m["id"],
            model_id=m["model_id"],
            family=m.get("family", "other"),
            dtype_pref=m.get("dtype"),
        )

if __name__ == "__main__":
    main()
