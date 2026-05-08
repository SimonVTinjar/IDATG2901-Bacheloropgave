from pathlib import Path
import json
import random

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -------------------------------------------------
# KONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

METADATA_PATH = BASE_DIR / "training_data_reference" / "metadata.jsonl"
MODEL_OUTPUT = BASE_DIR / "local_box_editor_ref_unet.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# DATASET
# -------------------------------------------------

class BoxEditReferenceDataset(Dataset):
    def __init__(self, metadata_path):
        self.items = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

        random.shuffle(self.items)

        self.rgb_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        self.gray_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        input_img = Image.open(item["input"]).convert("RGB")
        target_img = Image.open(item["target"]).convert("RGB")
        mask_img = Image.open(item["mask"]).convert("L")
        guide_img = Image.open(item["text_guide"]).convert("L")
        ref_img = Image.open(item["reference"]).convert("RGB")

        x_input = self.rgb_tf(input_img)
        x_target = self.rgb_tf(target_img)
        x_mask = self.gray_tf(mask_img)
        x_guide = self.gray_tf(guide_img)
        x_ref = self.rgb_tf(ref_img)

        x = torch.cat([x_input, x_mask, x_guide, x_ref], dim=0)
        y = x_target

        return x, y


# -------------------------------------------------
# MODELL
# -------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=8, out_ch=3):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        m = self.mid(p3)

        u3 = self.up3(m)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


# -------------------------------------------------
# LOSS
# -------------------------------------------------

def masked_loss(pred, target, x):
    input_rgb = x[:, 0:3, :, :]
    mask = x[:, 3:4, :, :]

    l1 = nn.L1Loss(reduction="none")
    loss_all = l1(pred, target)

    # inni masken
    mask_den = (mask.sum() * 3.0) + 1e-8
    loss_mask = (loss_all * mask).sum() / mask_den

    # utenfor masken
    outside = 1.0 - mask
    outside_den = (outside.sum() * 3.0) + 1e-8
    loss_outside = (l1(pred, input_rgb) * outside).sum() / outside_den

    return loss_mask * 20.0 + loss_outside * 1.0


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Fant ikke metadata: {METADATA_PATH}")

    dataset = BoxEditReferenceDataset(METADATA_PATH)

    if len(dataset) == 0:
        raise ValueError("Datasettet er tomt.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = UNetSmall(in_ch=8, out_ch=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Samples:", len(dataset))
    print("Starter trening...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = masked_loss(pred, y, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={avg_loss:.6f}")

    torch.save({
        "model_state": model.state_dict(),
        "image_size": IMAGE_SIZE,
        "note": "Reference-based editor: input RGB + mask + text_guide + reference RGB"
    }, MODEL_OUTPUT)

    print()
    print("Lagret modell:", MODEL_OUTPUT)


if __name__ == "__main__":
    main()