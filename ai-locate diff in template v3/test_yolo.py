from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

results = model("test.png", show=True)

for r in results:
    print(r.boxes)