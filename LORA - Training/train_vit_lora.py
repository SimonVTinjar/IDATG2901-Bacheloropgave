import os
import glob
import numpy as np
import torch

from multiprocessing import freeze_support
from functools import partial
from PIL import Image

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import evaluate
from sklearn.metrics import confusion_matrix, classification_report


def build_dataset(data_root: str) -> tuple[DatasetDict, str]:
    train_dir = os.path.join(data_root, "training")
    val_dir = os.path.join(data_root, "validation")
    test_dir = os.path.join(data_root, "test")

    train_dd = load_dataset("imagefolder", data_dir=train_dir)
    val_dd = load_dataset("imagefolder", data_dir=val_dir)
    test_dd = load_dataset("imagefolder", data_dir=test_dir)

    train_split = list(train_dd.keys())[0]
    val_split = list(val_dd.keys())[0]
    test_split = list(test_dd.keys())[0]

    dataset = DatasetDict({
        "train": train_dd[train_split],
        "validation": val_dd[val_split],
        "test": test_dd[test_split],
    })

    # Hold examples as python objects (PIL images) for our collate_fn
    dataset["train"].set_format(type="python")
    dataset["validation"].set_format(type="python")
    dataset["test"].set_format(type="python")

    print("Num train:", len(dataset["train"]))
    print("Num val:", len(dataset["validation"]))
    print("Num test:", len(dataset["test"]))
    print("Train features:", dataset["train"].features)
    print("First example keys:", dataset["train"][0].keys())
    print("Test example keys:", dataset["test"][0].keys())

    return dataset, test_dir


def collate_fn(batch, processor):
    images = []
    has_label = "label" in batch[0]

    for x in batch:
        img = x["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    inputs = processor(images=images, return_tensors="pt")

    if has_label:
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        inputs["labels"] = labels

    return inputs


def predict_unlabeled_folder(model, processor, folder, id2label, batch_size=64, threshold=0.60):
    """
    Runs inference on an unlabeled folder and saves (path, predicted label, confidence).
    Also emits 'UNCERTAIN' if confidence < threshold.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, "**", e), recursive=True)

    paths = sorted(paths)
    if len(paths) == 0:
        print(f"[WARN] No images found under: {folder}")
        return []

    device = next(model.parameters()).device
    model.eval()

    results = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        conf, pred = torch.max(probs, dim=-1)

        for p, pr, cf in zip(batch_paths, pred.tolist(), conf.tolist()):
            pred_label = id2label[pr]
            final_label = pred_label if cf >= threshold else "UNCERTAIN"
            results.append({
                "path": p,
                "pred_id": pr,
                "pred_label": pred_label,
                "confidence": float(cf),
                "final_label": final_label,
            })

    return results


def main():
    # ---- CUDA check
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA device: None")

    # ---- PATH (change if needed)
    DATA_ROOT = r"C:\Users\benja\Desktop\BachelorOppgaveRepo\IDATG2901-Bacheloropgave\LORA - Training\Indian currency dataset v1"

    # ---- Load dataset
    dataset, test_dir = build_dataset(DATA_ROOT)

    # ---- Labels
    labels = dataset["train"].features["label"].names
    print("Classes:", labels)

    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}

    # ---- Model + processor
    base_model_name = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(base_model_name, use_fast=True)

    collator = partial(collate_fn, processor=processor)

    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ---- LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Metrics (validation only)
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=y)

    # ---- Training args
    args = TrainingArguments(
        output_dir="vit-lora-inr",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        warmup_ratio=0.05,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

        logging_steps=50,
        fp16=True,
        report_to="none",

        remove_unused_columns=False,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],  # labeled -> metrics work
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    # ---- Train
    trainer.train()

    # ---- Final validation report (optional extra)
    print("\n=== FINAL VALIDATION EVAL ===")
    val_metrics = trainer.evaluate(dataset["validation"])
    print(val_metrics)

    # ---- Unlabeled test: run predictions + save CSV
    print("\n=== UNLABELED TEST PREDICTIONS ===")
    # ensure model on correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)

    rows = predict_unlabeled_folder(
        model=trainer.model,
        processor=processor,
        folder=test_dir,
        id2label=id2label,
        batch_size=64,
        threshold=0.60,
    )

    # Save CSV without pandas (so you don't need extra deps)
    out_csv = "test_predictions.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("path,pred_id,pred_label,confidence,final_label\n")
        for r in rows:
            # Quote path in case of commas
            path = r["path"].replace('"', '""')
            f.write(f"\"{path}\",{r['pred_id']},{r['pred_label']},{r['confidence']:.6f},{r['final_label']}\n")

    print(f"Saved {len(rows)} predictions to: {out_csv}")

    # Simple summary: counts per predicted label
    counts = {}
    for r in rows:
        counts[r["final_label"]] = counts.get(r["final_label"], 0) + 1
    print("Prediction counts (final_label):")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    # ---- Save LoRA adapter + processor
    trainer.model.save_pretrained("vit-lora-adapter")
    processor.save_pretrained("vit-lora-adapter")
    print("\nSaved LoRA adapter to: vit-lora-adapter")


if __name__ == "__main__":
    freeze_support()
    main()
