from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR.parent

RUNS_DIR = TEMPLATE_DIR / "outputs" / "yolo_runs_archive" / "runs" / "detect"

best_models = list(RUNS_DIR.glob("train*/weights/best.pt"))

if not best_models:
    raise FileNotFoundError(f"Fant ingen best.pt under: {RUNS_DIR}")

MODEL_PATH = max(best_models, key=lambda p: p.stat().st_mtime)

print(f"Bruker YOLO-modell: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

TEST_IMAGE = BASE_DIR.parent / "Prob Done v2.png"

if not TEST_IMAGE.exists():
    raise FileNotFoundError(f"Fant ikke testbildet: {TEST_IMAGE}")

results = model(str(TEST_IMAGE), show=True)

for r in results:
    print(r.boxes)