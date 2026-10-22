"""Bar chart comparing per-class sensitivity / +P of the three models on DS2."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
lr = json.loads((R / "logistic_regression.json").read_text())["per_class"]
gb = json.loads((R / "gradient_boosting.json").read_text())["per_class"]
cnn = json.loads((R / "cnn_seeds.json").read_text())
classes = ["N", "S", "V", "F"]
models = {
    "Logistic regression (features)": ({c: lr[c]["sensitivity"] for c in classes}, {c: lr[c]["ppv"] for c in classes}, None),
    "Gradient boosting (features)": ({c: gb[c]["sensitivity"] for c in classes}, {c: gb[c]["ppv"] for c in classes}, None),
    "CNN (waveform + features, 3 seeds)": ({c: cnn[f"{c}_sensitivity"]["mean"] for c in classes},
                                           {c: cnn[f"{c}_ppv"]["mean"] for c in classes},
                                           ({c: cnn[f"{c}_sensitivity"]["std"] for c in classes},
                                            {c: cnn[f"{c}_ppv"]["std"] for c in classes})),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
x = np.arange(len(classes)); w = 0.26
colors = ["#1565C0", "#7FA7D9", "#F4A100"]
for ax, k, title in [(axes[0], 0, "Sensitivity"), (axes[1], 1, "Positive predictivity (+P)")]:
    for i, (name, vals) in enumerate(models.items()):
        v = [vals[k][c] for c in classes]
        err = [vals[2][k][c] for c in classes] if vals[2] else None
        ax.bar(x + (i - 1) * w, v, w, yerr=err, capsize=3, label=name, color=colors[i])
    ax.set_xticks(x, classes); ax.set_title(title, fontsize=11); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle("MIT-BIH DS2 — 22 unseen patients", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(R / "model_comparison.png", dpi=150)
print("saved")
