"""
Batch-generering av seddelbilder via ComfyUI API.
Trening 3 - Z-Image Base med USD LoRA.
Kjør ComfyUI først, deretter: python generate_batch_lora_t3.py
"""

import json
import urllib.request
import time
import sys

# --- Innstillinger ---
COMFYUI_URL = "http://127.0.0.1:8000"
NUM_IMAGES = 200
START_SEED = 1
PROMPT_TEXT = "A perfect, flawless US 100 dollar banknote, front side, centered, flat lay on white surface. Every letter perfectly legible, every number correctly printed, symmetrical layout, museum-quality reproduction photograph, 8K resolution"
OUTPUT_PREFIX = "Lora-trening3-batch"
LORA_NAME = "my_zimage_a100_v1_000001800.safetensors"
LORA_STRENGTH = 1.0
# ---------------------

def build_workflow(seed: int) -> dict:
    return {
        "9": {
            "inputs": {
                "filename_prefix": OUTPUT_PREFIX,
                "images": ["76:65", 0]
            },
            "class_type": "SaveImage"
        },
        "76:62": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default"
            },
            "class_type": "CLIPLoader"
        },
        "76:63": {
            "inputs": {
                "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "76:71": {
            "inputs": {
                "text": "",
                "clip": ["76:62", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "76:65": {
            "inputs": {
                "samples": ["76:69", 0],
                "vae": ["76:63", 0]
            },
            "class_type": "VAEDecode"
        },
        "76:66": {
            "inputs": {
                "unet_name": "z_image_bf16.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        "76:95": {
            "inputs": {
                "lora_name": LORA_NAME,
                "strength_model": LORA_STRENGTH,
                "model": ["76:66", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "76:70": {
            "inputs": {
                "shift": 3,
                "model": ["76:95", 0]
            },
            "class_type": "ModelSamplingAuraFlow"
        },
        "76:68": {
            "inputs": {
                "width": 1344,
                "height": 576,
                "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage"
        },
        "76:67": {
            "inputs": {
                "text": PROMPT_TEXT,
                "clip": ["76:62", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "76:69": {
            "inputs": {
                "seed": seed,
                "steps": 30,
                "cfg": 2,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["76:70", 0],
                "positive": ["76:67", 0],
                "negative": ["76:71", 0],
                "latent_image": ["76:68", 0]
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
