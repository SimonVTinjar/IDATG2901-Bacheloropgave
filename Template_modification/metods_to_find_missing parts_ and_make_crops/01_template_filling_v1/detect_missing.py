from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2

from pathlib import Path

DIFF_THRESHOLD = 0.02
MIN_AREA = 20

BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = (2984, 7200)

MODEL_PATH = BASE_DIR.parent / "models" / "autoencoders" / "template_autoencoder_v1.pth"
INPUT_IMAGE = BASE_DIR.parent / "Prob Done v2.png"

OUTPUT_BOX = BASE_DIR / "detected_error_box.png"
OUTPUT_MASK = BASE_DIR / "error_mask.png"

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)


def postprocess_mask(diff_gray, threshold=DIFF_THRESHOLD):
    mask = (diff_gray > threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_largest_box(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    best_box = None
    best_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= MIN_AREA and area > best_area:
            best_area = area
            best_box = (x, y, x + w, y + h)

    return best_box


def main():
    model, device = load_model()

    img = Image.open(INPUT_IMAGE).convert("RGB")
    resized_img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    x = image_to_tensor(img).to(device)

    with torch.no_grad():
        recon = model(x)[0].cpu().permute(1, 2, 0).numpy()

    orig_np = np.array(resized_img).astype(np.float32) / 255.0
    diff = np.abs(orig_np - recon)
    diff_gray = diff.mean(axis=2)

    diff_vis = (diff_gray * 1200).clip(0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save("diff_gray.png")
    print("Lagret diff_gray.png")

    mask = postprocess_mask(diff_gray)
    box = find_largest_box(mask)

    vis = (orig_np * 255).astype(np.uint8).copy()

    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print("Fant mulig manglende område:", box)
    else:
        print("Fant ikke noe tydelig område.")

    Image.fromarray(vis).save(OUTPUT_BOX)
    Image.fromarray(mask).save(OUTPUT_MASK)

    print(f"Lagret {OUTPUT_BOX}")
    print(f"Lagret {OUTPUT_MASK}")


if __name__ == "__main__":
    main()