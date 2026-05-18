from src.pipeline import UncondLatentDiffusionPipeline
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "./1-output"  # Path to the directory containing the model

# 🔥 mappe for output
output_dir = "generated_images256x256"
os.makedirs(output_dir, exist_ok=True)

# load model
pipeline = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)

num_images = 200
batch_size = 10

image_counter = 0

while image_counter < num_images:
    images = pipeline(
        num_inference_steps=200,
        height=256,
        width=256,
        batch_size=batch_size
    ).images

    for img in images:
        if image_counter >= num_images:
            break

        img.save(os.path.join(output_dir, f"generated_{image_counter}.png"))
        image_counter += 1

print("Ferdig!")