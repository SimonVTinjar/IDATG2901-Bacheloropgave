"""
BLIP Image Captioning for Banknote Analysis
============================================
Generates captions from banknote images using Salesforce BLIP model.
Use the output as prompts for Z-Image Turbo generation.

Setup:
    pip install transformers torch pillow

Usage:
    python blip_caption.py <image_path>
    python blip_caption.py <image_path> --conditional "a photograph of"
    python blip_caption.py <folder_path>      # process all images in folder
"""

import sys
import os
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import json
from datetime import datetime


def load_model():
    print("Loading BLIP model (first run downloads ~1GB)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU")
    else:
        print("Using CPU (slower)")
    
    return processor, model


def caption_image(processor, model, image_path, conditional_text=None):
    """Generate caption for a single image.
    
    Args:
        conditional_text: Optional starting text like "a photograph of"
                         This guides the model to generate more specific captions.
                         If None, generates unconditional caption.
    """
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    
    results = {}
    
    # Unconditional caption (let model describe freely)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,           # beam search for better quality
        repetition_penalty=1.5  # avoid repetitive descriptions
    )
    results["unconditional"] = processor.decode(out[0], skip_special_tokens=True)
    
    # Conditional captions with different prompts
    conditional_prompts = [
        "a photograph of",
        "a detailed image of",
        "a banknote showing",
        "a close-up of a US dollar bill featuring",
        "this image contains",
    ]
    
    if conditional_text:
        conditional_prompts.insert(0, conditional_text)
    
    for prompt in conditional_prompts:
        inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            repetition_penalty=1.5
        )
        caption = processor.decode(out[0], skip_special_tokens=True)
        results[f"conditional: '{prompt}'"] = caption
    
    return results


def process_path(path, conditional_text=None):
    processor, model = load_model()
    
    path = Path(path)
    all_results = {}
    
    if path.is_file():
        image_files = [path]
    elif path.is_dir():
        image_files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        )
        print(f"Found {len(image_files)} images in {path}")
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
    
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")
        
        results = caption_image(processor, model, img_path, conditional_text)
        all_results[str(img_path)] = results
        
        for prompt_type, caption in results.items():
            print(f"\n  [{prompt_type}]")
            print(f"  → {caption}")
    
    # Save results to JSON
    output_file = f"blip_captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Also save a simple text file with just the best captions for easy copy-paste
    prompt_file = f"generated_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(prompt_file, 'w') as f:
        f.write("# BLIP-generated prompts for Z-Image Turbo\n")
        f.write("# Use these as prompts in ComfyUI\n\n")
        for img_path, results in all_results.items():
            f.write(f"# Source: {Path(img_path).name}\n")
            for prompt_type, caption in results.items():
                f.write(f"# [{prompt_type}]\n")
                f.write(f"{caption}\n\n")
    print(f"Prompts saved to: {prompt_file}")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python blip_caption.py <image_or_folder>")
        print("  python blip_caption.py <image_or_folder> --conditional 'a photograph of'")
        sys.exit(1)
    
    path = sys.argv[1]
    conditional = None
    
    if "--conditional" in sys.argv:
        idx = sys.argv.index("--conditional")
        if idx + 1 < len(sys.argv):
            conditional = sys.argv[idx + 1]
    
    process_path(path, conditional)