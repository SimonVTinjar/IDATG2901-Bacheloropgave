from pathlib import Path
from PIL import Image, ImageDraw
import json
import torch
import torch.nn as nn
from torchvision import transforms


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

TEMPLATE_PATH = BASE_DIR / "Prob Done v2.png"
JSON_PATH = BASE_DIR / "template_named_boxes.json"
MODEL_PATH = BASE_DIR.parent / "models" / "editors" / "local_box_editor_unet.pth"

OUTPUT_PATH = BASE_DIR / "local_model_edited_output.png"

CROP_MARGIN = 40
MASK_PADDING = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# MODELLEN MÅ MATCHE 05_train_local_editor.py
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
    def __init__(self, in_ch=5, out_ch=3):
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
# HJELPEFUNKSJONER
# -------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v, low, high):
    return max(low, min(high, v))


def expand_bbox(bbox, image_size, margin):
    x1, y1, x2, y2 = bbox
    w, h = image_size

    return [
        clamp(x1 - margin, 0, w),
        clamp(y1 - margin, 0, h),
        clamp(x2 + margin, 0, w),
        clamp(y2 + margin, 0, h),
    ]


def make_crop_mask(crop_size, bbox, crop_box, padding=4):
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = bbox

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    cw, ch = crop_size
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)

    lx1 = clamp(lx1 - padding, 0, cw)
    ly1 = clamp(ly1 - padding, 0, ch)
    lx2 = clamp(lx2 + padding, 0, cw)
    ly2 = clamp(ly2 + padding, 0, ch)

    draw.rectangle([lx1, ly1, lx2, ly2], fill=255)
    return mask


def create_text_guide(crop_size, bbox, crop_box, text):
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = bbox

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    guide = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(guide)

    if not text:
        return guide

    font = ImageDraw.ImageDraw(guide).getfont()

    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    bw = max(1, lx2 - lx1)
    bh = max(1, ly2 - ly1)

    tx = lx1 + max(0, (bw - tw) // 2)
    ty = ly1 + max(0, (bh - th) // 2)

    draw.text((tx, ty), text, fill=255, font=font)
    return guide


def tensor_to_pil(t):
    t = t.detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(t)
    return img


def paste_only_masked_area(base_img, edited_crop, crop_box, mask):
    x1, y1, x2, y2 = crop_box

    if edited_crop.size != (x2 - x1, y2 - y1):
        edited_crop = edited_crop.resize((x2 - x1, y2 - y1), Image.LANCZOS)

    if mask.size != edited_crop.size:
        mask = mask.resize(edited_crop.size, Image.NEAREST)

    base_img.paste(edited_crop, (x1, y1), mask)
    return base_img


# -------------------------------------------------
# INFERENCE
# -------------------------------------------------

def main():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Fant ikke template: {TEMPLATE_PATH}")

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Fant ikke JSON: {JSON_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fant ikke modellen: {MODEL_PATH}. Kjør 05_train_local_editor.py først."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    image_size = checkpoint.get("image_size", 256)

    model = UNetSmall(in_ch=5, out_ch=3).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tf_rgb = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    tf_gray = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    data = load_json(JSON_PATH)
    regions = data.get("boxes", [])

    result = Image.open(TEMPLATE_PATH).convert("RGB")

    print("Device:", DEVICE)
    print(f"Redigerer {len(regions)} bokser med lokal modell.")

    for i, region in enumerate(regions):
        bbox = region["bbox"]
        value = region.get("value", "")
        label = region.get("label", "field")

        crop_box = expand_bbox(bbox, result.size, CROP_MARGIN)
        crop = result.crop(crop_box)
        mask = make_crop_mask(crop.size, bbox, crop_box, MASK_PADDING)
        guide = create_text_guide(crop.size, bbox, crop_box, value)

        x_rgb = tf_rgb(crop)
        x_mask = tf_gray(mask)
        x_guide = tf_gray(guide)

        x = torch.cat([x_rgb, x_mask, x_guide], dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x)[0]

        edited = tensor_to_pil(pred)
        edited = edited.resize(crop.size, Image.LANCZOS)

        result = paste_only_masked_area(result, edited, crop_box, mask)

        print(f"[{i+1}/{len(regions)}] {label} -> {value}")

    result.save(OUTPUT_PATH)

    print()
    print("Ferdig.")
    print("Lagret:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
