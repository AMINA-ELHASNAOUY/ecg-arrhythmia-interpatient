"""AAMI-style metrics (sensitivity / positive predictivity per class) and plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .data import CLASSES


def report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        se = tp / cm[i].sum() if cm[i].sum() else 0.0
        ppv = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
        per_class[c] = {"support": int(cm[i].sum()), "sensitivity": round(float(se), 4),
                        "ppv": round(float(ppv), 4), "f1": round(float(f1), 4)}
    return {
        "accuracy": round(float(np.trace(cm) / cm.sum()), 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def macro_f1(y_true, y_pred, classes=CLASSES):
    pc = report(y_true, y_pred)["per_class"]
    return float(np.mean([pc[c]["f1"] for c in classes]))


def save_report(rep, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rep, indent=2))

    cm = np.asarray(rep["confusion_matrix"])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    split = "SVDB, external database" if name.startswith("svdb") else "DS2, unseen patients"
    ax.set_title(f"{name.replace('ds2_', '').replace('svdb_', '')} ({split})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=150)
    plt.close(fig)


def table(rows):
    """Markdown table: rows = [(model name, report), ...]."""
    head = "| Model | Acc | Macro-F1 | " + " | ".join(f"{c} Se | {c} +P" for c in CLASSES) + " |"
    sep = "|" + "---|" * (3 + 2 * len(CLASSES))
    lines = [head, sep]
    for name, r in rows:
        cells = [f"{r['per_class'][c]['sensitivity']:.1%} | {r['per_class'][c]['ppv']:.1%}" for c in CLASSES]
        lines.append(f"| {name} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
