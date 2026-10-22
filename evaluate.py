"""Final evaluation: feature model, CNN (3 seeds averaged) and their ensemble on
  * DS2  - 22 unseen MIT-BIH patients
  * SVDB - 78 patients from a different database (MIT-BIH Supraventricular Arrhythmia Database)
plus a per-patient error table for DS2.

The ensemble weight is chosen on the DS1 validation patients only. For that reason the
logistic regression here is trained on the DS1 training patients only (not on validation).
Run after train.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLASSES, load_external, load_splits
from src.metrics import macro_f1, report, save_report, table
from src.model import BeatNet

RESULTS = Path(__file__).resolve().parent / "results"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def cnn_proba(models, x, f, dev, bs=2048):
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(dev)
        fb = torch.from_numpy(f[i:i + bs]).to(dev)
        out.append(torch.stack([m(xb, fb).softmax(1) for m in models]).mean(0).cpu())
    return torch.cat(out).numpy()


def per_patient(y, rec, preds):
    """preds: {model name: predicted labels}. One row per record."""
    rows = []
    for r in np.unique(rec):
        m = rec == r
        row = {"record": int(r), "beats": int(m.sum())}
        for c_i, c in enumerate(CLASSES):
            row[f"n_{c}"] = int((y[m] == c_i).sum())
        for name, p in preds.items():
            for c_i, c in [(1, "S"), (2, "V")]:
                tp = int(((y[m] == c_i) & (p[m] == c_i)).sum())
                row[f"{name}_{c}_tp"] = tp
                row[f"{name}_{c}_fn"] = int(((y[m] == c_i) & (p[m] != c_i)).sum())
                row[f"{name}_{c}_fp"] = int(((y[m] != c_i) & (p[m] == c_i)).sum())
            row[f"{name}_acc"] = round(float((y[m] == p[m]).mean()), 4)
        rows.append(row)
    return rows


def main():
    dev = device()
    d = load_splits()
    ext = load_external()
    norm = np.load(RESULTS / "feature_norm.npz")
    mu, sd = norm["mu"], norm["sd"]

    # feature model (trained on DS1 training patients only)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(d["train_f"], d["train_y"])

    models = []
    for p in sorted(RESULTS.glob("beatnet_seed*.pt")):
        m = BeatNet().to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        models.append(m)
    print(f"{len(models)} CNN seeds loaded")

    def probs(split_x, split_f):
        fn = ((split_f - mu) / sd).astype(np.float32)
        return lr.predict_proba(split_f), cnn_proba(models, split_x, fn, dev)

    sets = {"val": (d["val_x"], d["val_f"], d["val_y"]),
            "DS2": (d["test_x"], d["test_f"], d["test_y"]),
            "SVDB": (ext["ext_x"], ext["ext_f"], ext["ext_y"])}
    P = {k: probs(x, f) for k, (x, f, _) in sets.items()}

    # ensemble weight chosen on validation patients (N/S/V macro-F1; val has only 16 F beats)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    scores = [macro_f1(sets["val"][2], (w * P["val"][0] + (1 - w) * P["val"][1]).argmax(1), ["N", "S", "V"])
              for w in grid]
    w = float(grid[int(np.argmax(scores))])
    print(f"ensemble weight on logistic regression (chosen on validation): {w}")

    reports, rows = {}, {}
    for split in ["DS2", "SVDB"]:
        y = sets[split][2]
        p_lr, p_cnn = P[split]
        preds = {"lr": p_lr.argmax(1), "cnn": p_cnn.argmax(1), "ensemble": (w * p_lr + (1 - w) * p_cnn).argmax(1)}
        for name, p in preds.items():
            rep = report(y, p)
            reports[(split, name)] = rep
            save_report(rep, RESULTS, f"{split.lower()}_{name}")
        rows[split] = preds

    pretty = {"lr": "Logistic regression (features)", "cnn": "CNN (3-seed average)",
              "ensemble": f"Ensemble (w_LR = {w})"}
    md = ["# Final evaluation", "",
          f"Ensemble = {w} x logistic regression + {1 - w:.2f} x CNN probabilities (weight chosen on DS1 validation patients).", ""]
    for split, title in [("DS2", "DS2 — 22 unseen MIT-BIH patients"),
                         ("SVDB", "SVDB — 78 patients from a second database (external)")]:
        md += [f"## {title}", "", table([(pretty[n], reports[(split, n)]) for n in ["lr", "cnn", "ensemble"]]), ""]
        n = np.bincount(sets[split][2], minlength=4)
        md += ["Support: " + " · ".join(f"{c} {k:,}" for c, k in zip(CLASSES, n)), ""]

    # per-patient table (DS2)
    pp = per_patient(sets["DS2"][2], d["test_rec"], rows["DS2"])
    (RESULTS / "per_patient_ds2.json").write_text(json.dumps(pp, indent=2))
    md += ["## Per-patient errors on DS2 (ensemble)", "",
           "| Record | Beats | S beats | S found | S missed | false S | V beats | V found | V missed | false V | Accuracy |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(pp, key=lambda r: -(r["ensemble_S_fn"] + r["ensemble_S_fp"] + r["ensemble_V_fn"] + r["ensemble_V_fp"])):
        md.append(f"| {r['record']} | {r['beats']} | {r['n_S']} | {r['ensemble_S_tp']} | {r['ensemble_S_fn']} | "
                  f"{r['ensemble_S_fp']} | {r['n_V']} | {r['ensemble_V_tp']} | {r['ensemble_V_fn']} | "
                  f"{r['ensemble_V_fp']} | {r['ensemble_acc']:.1%} |")
    (RESULTS / "final_results.md").write_text("\n".join(md) + "\n")
    (RESULTS / "ensemble.json").write_text(json.dumps({"weight_lr": w, "val_grid": dict(zip(map(str, grid), scores))}, indent=2))

    # per-patient error plot: where do S/V mistakes come from?
    recs = [str(r["record"]) for r in pp]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(recs))
    parts = [("ensemble_S_fn", "missed S", "#F4A100"), ("ensemble_S_fp", "false S", "#FBD38D"),
             ("ensemble_V_fn", "missed V", "#0A2E5C"), ("ensemble_V_fp", "false V", "#1565C0")]
    bottom = np.zeros(len(recs))
    for key, label, col in parts:
        v = np.array([r[key] for r in pp])
        ax.bar(x, v, bottom=bottom, label=label, color=col)
        bottom += v
    ax.set_xticks(x, recs, rotation=60, fontsize=8)
    ax.set_ylabel("beats")
    ax.set_title("Where the ensemble's S/V errors come from (DS2, per patient)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_patient_errors.png", dpi=150)

    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
