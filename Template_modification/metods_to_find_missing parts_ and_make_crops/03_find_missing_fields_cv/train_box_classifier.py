from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR.parent / "models" / "classifiers" / "box_classifier_filled_missing_v1.pth"

IMAGE_SIZE = (500, 1416)
BATCH_SIZE = 8
EPOCHS = 20
LR = 0.001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

    print("Klasser:", dataset.classes)
    # Forventet: ['filled', 'missing']

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss:.4f} accuracy={acc:.2%}")

    torch.save({
        "model_state": model.state_dict(),
        "classes": dataset.classes
    }, MODEL_PATH)

    print("Lagret modell:", MODEL_PATH)


if __name__ == "__main__":
    main()