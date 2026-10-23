"""Final evaluation on
  * DS2    - 22 unseen MIT-BIH patients
  * SVDB   - 78 patients, MIT-BIH Supraventricular Arrhythmia Database (external)
  * INCART - 75 recordings, St Petersburg INCART 12-lead database (external, if downloaded)

Main pipeline (lead A only): logistic regression, CNN (3 seeds averaged), their ensemble, and the ensemble
with the AF-aware rule. Multi-lead experiment: the same logistic regression / CNN / ensemble with both leads.
Everything that is chosen (ensemble weight, AF detector, AF threshold) is chosen on DS1 patients only.
The logistic regressions are trained on the 18 DS1 training patients so the 4 validation patients stay
unseen for choosing the ensemble weight. Run after train.py.
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

from src.af import AFDetector, af_metrics, apply_af_rule
from src.data import CLASSES, INCART_DIR, N_LEAD_A_FEATURES, load_external, load_splits
from src.metrics import macro_f1, report, save_report, table
from src.model import BeatNet

RESULTS = Path(__file__).resolve().parent / "results"
MODEL_NAMES = {
    "lr": "Logistic regression",
    "cnn": "CNN (3-seed average)",
    "ensemble": "Ensemble (LR + CNN)",
    "ensemble_af": "**Ensemble + AF-aware rule**",
    "lr_2lead": "Two leads: logistic regression",
    "cnn_2lead": "Two leads: CNN",
    "ensemble_2lead": "Two leads: ensemble",
}
A = N_LEAD_A_FEATURES


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


def per_patient(y, rec, pred):
    rows = []
    for r in np.unique(rec):
        m = rec == r
        row = {"record": str(r), "beats": int(m.sum())}
        for c_i, c in enumerate(CLASSES):
            row[f"n_{c}"] = int((y[m] == c_i).sum())
        for c_i, c in [(1, "S"), (2, "V")]:
            row[f"{c}_tp"] = int(((y[m] == c_i) & (pred[m] == c_i)).sum())
            row[f"{c}_fn"] = int(((y[m] == c_i) & (pred[m] != c_i)).sum())
            row[f"{c}_fp"] = int(((y[m] != c_i) & (pred[m] == c_i)).sum())
        row["accuracy"] = round(float((y[m] == pred[m]).mean()), 4)
        rows.append(row)
    return rows


def main():
    dev = device()
    d = load_splits()
    sets = {"val": d, "DS2": d}
    split_key = {"val": "val", "DS2": "test"}
    externals = {"SVDB": "svdb"}
    if (INCART_DIR / "RECORDS").exists() and all(
            all((INCART_DIR / f"{r}.{e}").exists() for e in ["hea", "atr", "dat"])
            for r in (INCART_DIR / "RECORDS").read_text().split()):
        externals["INCART"] = "incart"
    for name, db in externals.items():
        sets[name] = load_external(db)
        split_key[name] = "ext"

    def lr_model(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        return m.fit(d["train_f"][:, cols], d["train_y"])

    def cnn_models(tag, in_ch, n_feat):
        out = []
        for p in sorted(RESULTS.glob(f"beatnet{tag}_seed*.pt")):
            m = BeatNet(in_ch=in_ch, n_feat=n_feat).to(dev)
            m.load_state_dict(torch.load(p, map_location=dev))
            m.eval()
            out.append(m)
        return out

    configs = {  # name: (feature columns, CNN channels, CNN weights tag)
        "": (slice(0, A), slice(0, 2), ""),
        "_2lead": (slice(0, None), slice(0, 4), "_2lead"),
    }
    P = {name: {} for name in sets}
    for suffix, (cols, ch, tag) in configs.items():
        lr = lr_model(cols)
        nf = d["train_f"][:, cols].shape[1]
        models = cnn_models(tag, ch.stop, nf)
        norm = np.load(RESULTS / f"feature_norm{tag}.npz")
        print(f"config '{suffix or 'lead A'}': {len(models)} CNN seeds loaded")
        for name, data in sets.items():
            k = split_key[name]
            f = data[f"{k}_f"][:, cols]
            P[name][f"lr{suffix}"] = lr.predict_proba(f)
            P[name][f"cnn{suffix}"] = cnn_proba(models, np.ascontiguousarray(data[f"{k}_x"][:, ch]),
                                                ((f - norm["mu"]) / norm["sd"]).astype(np.float32), dev)

    af = AFDetector().fit(d["train_r"], d["train_af"], d["train_rec"])
    print("AF detector:", af.summary())
    for name, data in sets.items():
        P[name]["af"] = af.predict(data[f"{split_key[name]}_r"])

    # ---- ensemble weights on validation patients (N/S/V macro-F1; val has only 16 F beats)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    yv = d["val_y"]
    weights, val_grid = {}, {}
    for suffix in configs:
        sc = [macro_f1(yv, (w_ * P["val"][f"lr{suffix}"] + (1 - w_) * P["val"][f"cnn{suffix}"]).argmax(1), ["N", "S", "V"])
              for w_ in grid]
        weights[suffix] = float(grid[int(np.argmax(sc))])
        val_grid[suffix or "lead_A"] = dict(zip(map(str, grid), sc))
    w = weights[""]
    print(f"ensemble weights on logistic regression (chosen on validation): {weights}")

    reports, preds = {}, {}
    for name in [n for n in sets if n != "val"]:
        data, k = sets[name], split_key[name]
        y = data[f"{k}_y"]
        p = P[name]
        ens = (w * p["lr"] + (1 - w) * p["cnn"]).argmax(1)
        w2 = weights["_2lead"]
        preds[name] = {"lr": p["lr"].argmax(1), "cnn": p["cnn"].argmax(1), "ensemble": ens,
                       "ensemble_af": apply_af_rule(ens, p["af"]),
                       "lr_2lead": p["lr_2lead"].argmax(1), "cnn_2lead": p["cnn_2lead"].argmax(1),
                       "ensemble_2lead": (w2 * p["lr_2lead"] + (1 - w2) * p["cnn_2lead"]).argmax(1)}
        for model, pr in preds[name].items():
            rep = report(y, pr)
            reports[(name, model)] = rep
            save_report(rep, RESULTS, f"{name.lower()}_{model}")

    af_eval = {"detector": af.summary(), "DS2": af_metrics(P["DS2"]["af"], d["test_af"]),
               "note": "SVDB and INCART have no rhythm annotations; the detector is scored on MIT-BIH DS2 only."}
    (RESULTS / "af_detector.json").write_text(json.dumps(af_eval, indent=2))
    (RESULTS / "ensemble.json").write_text(json.dumps({"weight_lr": weights, "val_grid": val_grid}, indent=2))

    # ---- markdown summary
    md = ["# Final evaluation", "",
          f"Main pipeline uses lead A only. Ensemble = {w} x logistic regression + {1 - w:.2f} x CNN probabilities "
          f"(two-lead ensemble: {weights['_2lead']}), weights chosen on DS1 validation patients.",
          f"AF detector: {af.name.replace('_', ' ')}, threshold {af.threshold}, chosen by leave-patient-out CV on DS1 "
          f"(CV F1 {af.cv[af.name]['cv_f1']}). On DS2: AF sensitivity {af_eval['DS2']['sensitivity']:.1%}, "
          f"+P {af_eval['DS2']['ppv']:.1%}, specificity {af_eval['DS2']['specificity']:.1%}.", ""]
    titles = {"DS2": "DS2 — 22 unseen MIT-BIH patients", "SVDB": "SVDB — 78 patients, second database (external)",
              "INCART": "INCART — 75 recordings, third database (external)"}
    for name in [n for n in sets if n != "val"]:
        k = split_key[name]
        md += [f"## {titles[name]}", "", table([(MODEL_NAMES[m], reports[(name, m)]) for m in MODEL_NAMES]), ""]
        n = np.bincount(sets[name][f"{k}_y"], minlength=4)
        md += ["Support: " + " · ".join(f"{c} {v:,}" for c, v in zip(CLASSES, n)), ""]

    pp = per_patient(d["test_y"], d["test_rec"], preds["DS2"]["ensemble_af"])
    (RESULTS / "per_patient_ds2.json").write_text(json.dumps(pp, indent=2))
    md += ["## Per-patient errors on DS2 (ensemble + AF rule)", "",
           "| Record | Beats | S beats | S found | S missed | false S | V beats | V found | V missed | false V | Accuracy |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(pp, key=lambda r: -(r["S_fn"] + r["S_fp"] + r["V_fn"] + r["V_fp"])):
        md.append(f"| {r['record']} | {r['beats']} | {r['n_S']} | {r['S_tp']} | {r['S_fn']} | {r['S_fp']} | "
                  f"{r['n_V']} | {r['V_tp']} | {r['V_fn']} | {r['V_fp']} | {r['accuracy']:.1%} |")
    (RESULTS / "final_results.md").write_text("\n".join(md) + "\n")

    # ---- per-patient error plot (before vs after the AF rule)
    before = {r["record"]: r for r in per_patient(d["test_y"], d["test_rec"], preds["DS2"]["ensemble"])}
    recs = [r["record"] for r in pp]
    x = np.arange(len(recs))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - 0.2, [before[r]["S_fp"] for r in recs], 0.4, color="#FBD38D", label="false S, ensemble")
    ax.bar(x + 0.2, [r["S_fp"] for r in pp], 0.4, color="#F4A100", label="false S, ensemble + AF rule")
    ax.bar(x + 0.2, [r["S_fn"] for r in pp], 0.4, bottom=[r["S_fp"] for r in pp], color="#0A2E5C", label="missed S, ensemble + AF rule")
    ax.set_xticks(x, recs, rotation=60, fontsize=8)
    ax.set_ylabel("beats")
    ax.set_title("S-beat errors per DS2 patient, before and after the AF-aware rule", fontsize=10)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_patient_errors.png", dpi=150)

    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
