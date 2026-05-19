"""
Batch-generering av seddelbilder via ComfyUI API med LoRA.
Kjør ComfyUI først, deretter: python generate_batch_lora.py
"""

import json
import urllib.request
import time
import sys

# --- Innstillinger ---
COMFYUI_URL = "http://127.0.0.1:8000"
NUM_IMAGES = 200
START_SEED = 1
# PROMPT_TEXT = "A perfect, flawless US 100 dollar banknote, front side, centered, flat lay on white surface. Every letter perfectly legible, every number correctly printed, symmetrical layout, museum-quality reproduction photograph, 8K resolution"
PROMPT_TEXT = "A photorealistic 20 Thai Baht banknote, front side, flat lay, sharp focus, high detail, 8K resolution"

# Kjør én av gangen
# LORA_NAME    = "my_second_lora_000000750.safetensors"  # Trening 2 (USD)
# OUTPUT_PREFIX = "Lora-trening2-batch"

LORA_NAME    = "thai_lora.safetensors"               # Trening 4 (Thai Baht)
OUTPUT_PREFIX = "Lora-trening4-batch"

LORA_STRENGTH = 1.0
# ---------------------

def build_workflow(seed: int) -> dict:
    return {
        "9": {
            "inputs": {
                "filename_prefix": OUTPUT_PREFIX,
                "images": ["57:8", 0]
            },
            "class_type": "SaveImage"
        },
        "57:30": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default"
            },
            "class_type": "CLIPLoader"
        },
        "57:29": {
            "inputs": {
                "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "57:33": {
            "inputs": {
                "conditioning": ["57:27", 0]
            },
            "class_type": "ConditioningZeroOut"
        },
        "57:8": {
            "inputs": {
                "samples": ["57:3", 0],
                "vae": ["57:29", 0]
            },
            "class_type": "VAEDecode"
        },
        "57:28": {
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        "57:62": {
            "inputs": {
                "lora_name": LORA_NAME,
                "strength_model": LORA_STRENGTH,
                "model": ["57:28", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "57:11": {
            "inputs": {
                "shift": 3,
                "model": ["57:62", 0]
            },
            "class_type": "ModelSamplingAuraFlow"
        },
        "57:13": {
            "inputs": {
                "width": 1344,
                "height": 576,
                "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage"
        },
        "57:27": {
            "inputs": {
                "text": PROMPT_TEXT,
                "clip": ["57:30", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "57:3": {
            "inputs": {
                "seed": seed,
                "steps": 30,
                "cfg": 2,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["57:11", 0],
                "positive": ["57:27", 0],
                "negative": ["57:33", 0],
                "latent_image": ["57:13", 0]
            },
            "class_type": "KSampler"
        }
    }


def queue_prompt(workflow: dict) -> str:
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def get_queue_remaining() -> int:
    with urllib.request.urlopen(f"{COMFYUI_URL}/queue") as resp:
        data = json.loads(resp.read())
    return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))


def main():
    print(f"LoRA: {LORA_NAME} (strength={LORA_STRENGTH})")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print(f"Antall bilder: {NUM_IMAGES}")
    print("-" * 50)

    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats")
    except Exception:
        print("FEIL: ComfyUI kjører ikke på", COMFYUI_URL)
        sys.exit(1)

    MAX_QUEUE = 10

    for i in range(NUM_IMAGES):
        seed = START_SEED + i

        while get_queue_remaining() >= MAX_QUEUE:
            time.sleep(2)

        workflow = build_workflow(seed)
        prompt_id = queue_prompt(workflow)

        pct = (i + 1) / NUM_IMAGES * 100
        print(f"[{i+1:4d}/{NUM_IMAGES}] seed={seed} | id={prompt_id[:8]}... | {pct:.1f}%")

    print("\nAlle jobber sendt. Venter på at køen tømmes...")
    while get_queue_remaining() > 0:
        print(f"  {get_queue_remaining()} jobber gjenstår...", end="\r")
        time.sleep(3)

    print(f"\nFerdig! {NUM_IMAGES} bilder generert med prefix '{OUTPUT_PREFIX}'")


if __name__ == "__main__":
    main()