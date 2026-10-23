"""Per-class sensitivity / +P of the main models on every test set."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
CLASSES = ["N", "S", "V"]
MODELS = [("lr", "Logistic regression", "#1565C0"),
          ("cnn", "CNN", "#7FA7D9"),
          ("ensemble", "Ensemble", "#0A2E5C"),
          ("ensemble_af", "Ensemble + AF-aware rule", "#F4A100")]
SPLITS = [(s, t) for s, t in [("ds2", "MIT-BIH DS2 (22 unseen patients)"), ("svdb", "SVDB (78 patients)"),
                              ("incart", "INCART (75 recordings)")] if (R / f"{s}_lr.json").exists()]

fig, axes = plt.subplots(len(SPLITS), 2, figsize=(10, 3.1 * len(SPLITS) + 0.5), sharey=True, squeeze=False)
x = np.arange(len(CLASSES))
w = 0.2
for row, (split, title) in enumerate(SPLITS):
    for col, (metric, mname) in enumerate([("sensitivity", "Sensitivity"), ("ppv", "Positive predictivity (+P)")]):
        ax = axes[row, col]
        for i, (key, label, color) in enumerate(MODELS):
            pc = json.loads((R / f"{split}_{key}.json").read_text())["per_class"]
            ax.bar(x + (i - 1.5) * w, [pc[c][metric] for c in CLASSES], w, label=label, color=color)
        ax.set_xticks(x, CLASSES)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{mname} — {title}", fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(R / "model_comparison.png", dpi=150)
print("saved results/model_comparison.png")
