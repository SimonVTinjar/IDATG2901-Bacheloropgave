import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim


BASE_DIR = Path(__file__).parent
IMAGE_SIZE = (600, 1416)  # (height, width)
DATA_DIR = BASE_DIR / "data" / "small"
MODEL_PATH = BASE_DIR / "autoencoder.pth"


class FullImageDataset(Dataset):
    def __init__(self, folder):
        self.paths = [
            str(folder / f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


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
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Ser etter bilder i:", DATA_DIR.resolve())

    dataset = FullImageDataset(DATA_DIR)
    if len(dataset) == 0:
        raise ValueError(f"Ingen bilder funnet i {DATA_DIR}")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        model.train()
        total_loss = 0.0

        for images in loader:
            images = images.to(device)

            optimizer.zero_grad()
            recon = model(images)
            loss = loss_fn(recon, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/200 - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Lagret modell til {MODEL_PATH}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train()