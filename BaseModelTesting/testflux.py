import os, time, json
import torch
from tqdm import tqdm

# Flux pipelines live under diffusers; exact class can vary by diffusers version
from diffusers import FluxPipeline

MODEL_REPO = "black-forest-labs/FLUX.2-klein-9B"

PROMPTS = [
    (
        "combined_prompt_original",
        (
            "Top-down photo of white paper on a table. Clean handwriting: "
            "“Dette er en test av tekst. Alle bokstaver må være riktige.” "
        )
    )
]

SEEDS = [11, 22, 33, 44]
OUT_DIR = "runs_flux"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, f"results_{int(time.time())}.jsonl")

    print(f"Loading: {MODEL_REPO}")
    pipe = FluxPipeline.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16,
    ).to("cuda")

    # memory helpers
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()

    for pid, prompt in PROMPTS:
        for seed in tqdm(SEEDS, desc=f"flux:{pid}"):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # FLUX distilled often expects very low steps; start with 4
                image = pipe(
                    prompt=prompt,
                    generator=gen,
                    num_inference_steps=4,
                    height=832,
                    width=1216,
                ).images[0]


            path = os.path.join(OUT_DIR, f"{pid}_seed{seed}.png")
            image.save(path)

            rec = {
                "model": "flux2_klein_9b",
                "repo": MODEL_REPO,
                "prompt_id": pid,
                "seed": seed,
                "steps": 4,
                "image_path": path,
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
