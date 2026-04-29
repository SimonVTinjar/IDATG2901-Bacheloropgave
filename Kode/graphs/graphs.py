import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


loss_file = Path("loss_1.json")
lr_file = Path("lr_1.json")

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)


def load_tensorboard_json(path, value_name):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data, columns=["wall_time", "step", value_name])

    # Fjerner duplikate steps hvis TensorBoard har eksportert flere runs/restarts
    df = (
        df.sort_values(["step", "wall_time"])
        .drop_duplicates(subset=["step"], keep="last")
        .reset_index(drop=True)
    )

    return df


loss_df = load_tensorboard_json(loss_file, "loss")
lr_df = load_tensorboard_json(lr_file, "learning_rate")

# Glattet loss-kurve
loss_df["loss_smoothed"] = loss_df["loss"].rolling(window=50, min_periods=1).mean()


# -------- LOSS GRAPH --------
plt.figure(figsize=(10, 5))
plt.plot(loss_df["step"], loss_df["loss"], alpha=0.35, label="Loss")
plt.plot(loss_df["step"], loss_df["loss_smoothed"], label="Loss, rolling mean 50")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "loss_graph.svg")   # best for rapport
plt.savefig(output_dir / "loss_graph.pdf")   # også bra i LaTeX
plt.savefig(output_dir / "loss_graph.png", dpi=300)
plt.close()


# -------- LEARNING RATE GRAPH --------
plt.figure(figsize=(10, 5))
plt.plot(lr_df["step"], lr_df["learning_rate"], label="Learning rate")
plt.xlabel("Training step")
plt.ylabel("Learning rate")
plt.title("Learning rate schedule")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(output_dir / "learning_rate_graph.svg")
plt.savefig(output_dir / "learning_rate_graph.pdf")
plt.savefig(output_dir / "learning_rate_graph.png", dpi=300)
plt.close()


print(f"Ferdig! Grafer lagret i: {output_dir}")