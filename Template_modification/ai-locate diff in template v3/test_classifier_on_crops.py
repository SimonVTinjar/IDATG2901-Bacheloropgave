from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "box_classifier.pth"
TEST_DIR = BASE_DIR / "dataset" / "missing"

IMAGE_SIZE = (128, 128)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, classes


def predict(model, classes, image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    return classes[pred_idx], probs[pred_idx].item()


def main():
    model, classes = load_model()

    for path in TEST_DIR.glob("*.png"):
        label, conf = predict(model, classes, path)
        print(f"{path.name}: {label} ({conf:.2%})")


if __name__ == "__main__":
    main()