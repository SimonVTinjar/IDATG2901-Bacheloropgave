"""
Batch-generering av seddelbilder via ComfyUI API.
Kjør ComfyUI først, deretter: python generate_batch.py
"""

import json
import urllib.request
import urllib.parse
import time
import sys

# --- Innstillinger ---
COMFYUI_URL = "http://127.0.0.1:8000"
NUM_IMAGES   = 500
START_SEED   = 1
PROMPT_TEXT  = "A perfect, flawless US 100 dollar banknote, front side, centered, 2013 series, blue security ribbon, copper inkwell with bell, large portrait of Benjamin Franklin. Every letter perfectly legible, every number correctly printed, symmetrical layout, museum-quality reproduction photograph, 8K resolution. Full frame, no border, no shadow, banknote fills entire image."
OUTPUT_PREFIX = "New-dataset_batch"

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
        "57:13": {
            "inputs": {
                "width": 1344,
                "height": 576,
                "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage"
        },
        "57:11": {
            "inputs": {
                "shift": 3,
                "model": ["57:28", 0]
            },
            "class_type": "ModelSamplingAuraFlow"
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


def wait_for_completion(prompt_id: str, timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return True
        time.sleep(1)
    return False


def main():
    print(f"Starter generering av {NUM_IMAGES} bilder")
    print(f"Prompt: {PROMPT_TEXT}")
    print(f"Seeds: {START_SEED} -> {START_SEED + NUM_IMAGES - 1}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print("-" * 50)

    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats")
    except Exception:
        print("FEIL: ComfyUI ser ikke ut til å kjøre på", COMFYUI_URL)
        print("Start ComfyUI og prøv igjen.")
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

    while get_queue_remaining() > 0:
        remaining = get_queue_remaining()
        print(f"  {remaining} jobber gjenstår...", end="\r")
        time.sleep(3)

    print(f"\n{NUM_IMAGES} bilder generert.")
    print(f"Bildene ligger i ComfyUI sin output-mappe med prefix '{OUTPUT_PREFIX}'")


if __name__ == "__main__":
    main()
