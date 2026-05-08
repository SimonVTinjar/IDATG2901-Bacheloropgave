import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


BASE_DIR = Path(__file__).parent
METADATA_PATH = BASE_DIR / "dataset_missing_type" / "metadata.json"
MODEL_PATH = BASE_DIR / "box_type_position_missing_classifier.pth"

IMAGE_SIZE = (300, 500)
BATCH_SIZE = 8
EPOCHS = 700
LR = 0.001


class BoxDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.labels = sorted(list(set(item["label"] for item in self.items)))
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        img = Image.open(item["crop_path"]).convert("RGB")
        img = self.transform(img)

        x1, y1, x2, y2 = item["bbox"]
        img_w, img_h = item["image_size"]

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        position = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        label = torch.tensor(self.label_to_id[item["label"]], dtype=torch.long)

        return img, position, label


class ImagePositionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        image_dim = backbone.fc.in_features

        self.position_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, position):
        img_feat = self.image_features(image)
        img_feat = torch.flatten(img_feat, 1)

        pos_feat = self.position_net(position)

        combined = torch.cat([img_feat, pos_feat], dim=1)
        return self.classifier(combined)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BoxDataset(METADATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Klasser:", dataset.labels)

    model = ImagePositionClassifier(num_classes=len(dataset.labels)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, positions, labels in loader:
            images = images.to(device)
            positions = positions.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, positions)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total else 0
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss:.4f} acc={acc:.2%}")

    torch.save({
        "model_state": model.state_dict(),
        "classes": dataset.labels
    }, MODEL_PATH)

    print("Lagret:", MODEL_PATH)


if __name__ == "__main__":
    train()