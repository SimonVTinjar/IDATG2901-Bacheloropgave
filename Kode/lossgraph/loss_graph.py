import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("lossgraph_1152x480_run4.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data, columns=["wall_time", "step", "loss"])

# Fjerner duplikate steps hvis TensorBoard har eksportert flere forsøk/restarts
df = (
    df.sort_values(["step", "wall_time"])
    .drop_duplicates(subset="step", keep="last")
    .reset_index(drop=True)
)

df["loss_smoothed"] = df["loss"].rolling(window=50, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["loss"], alpha=0.35, label="Loss")
plt.plot(df["step"], df["loss_smoothed"], label="Loss, rolling mean 50")

plt.xlabel("Training step")
plt.ylabel("Loss")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Training loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("loss_graph_1152x480_run4.pdf")
plt.savefig("loss_graph_1152x480_run4.png", dpi=300)
plt.show()