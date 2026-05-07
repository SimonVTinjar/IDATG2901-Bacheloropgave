from pathlib import Path
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "unet_error_detector.pth"

INPUT_IMAGE = BASE_DIR / "test.png"
OUTPUT_MASK = BASE_DIR / "predicted_mask.png"
OUTPUT_BOX = BASE_DIR / "predicted_box.png"

IMAGE_SIZE = (512, 512)
THRESHOLD = 0.5
MIN_AREA = 100


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        m = self.middle(self.pool3(e3))

        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def find_boxes(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= MIN_AREA:
            boxes.append((x, y, x + w, y + h, area))

    return boxes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    original = Image.open(INPUT_IMAGE).convert("RGB")
    original_w, original_h = original.size

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    x = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob > THRESHOLD).astype(np.uint8) * 255

    # Rydd litt støy
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Skaler masken tilbake til original størrelse
    mask_original = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    boxes = find_boxes(mask_original)

    vis = np.array(original).copy()

    for box in boxes:
        x1, y1, x2, y2, area = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        print("Fant feilområde:", [x1, y1, x2, y2], "area:", area)

    Image.fromarray(mask_original).save(OUTPUT_MASK)
    Image.fromarray(vis).save(OUTPUT_BOX)

    print("Lagret:", OUTPUT_MASK)
    print("Lagret:", OUTPUT_BOX)


if __name__ == "__main__":
    main()