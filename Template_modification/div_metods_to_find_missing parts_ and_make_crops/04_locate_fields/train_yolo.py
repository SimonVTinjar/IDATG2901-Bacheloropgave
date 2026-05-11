from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "YOLO_training"
DATASET_YAML = BASE_DIR / "dataset.yaml"

yaml_text = f"""
path: "{DATASET_DIR.as_posix()}"

train: images
val: images

names:
  0: serial
  1: serial_2
  2: code
  3: top_left_code
  4: top_right_code
  5: series
  6: signature_1
  7: signature_2
  8: noise
"""

DATASET_YAML.write_text(yaml_text.strip(), encoding="utf-8")

print("Bruker dataset.yaml:")
print(DATASET_YAML.read_text(encoding="utf-8"))

model = YOLO(str(Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolov8n.pt"))

model.train(
    data=str(DATASET_YAML),
    epochs=50,
    imgsz=640
)