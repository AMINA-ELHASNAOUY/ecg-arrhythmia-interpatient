"""Recording-level analysis: false fusion alarms and when (not) to trust a recording's result.

Questions:
  1. Why are most errors false fusion (F) alarms, and does the fusion gate fix them?
  2. Can we tell, without labels, which recordings the model will get wrong?
  3. At recording level, does the pipeline get the ectopic burden right?
Also records one idea that did not work: rescaling the F probability (chosen on DS1) instead of the gate.

Uses the per-beat predictions saved by evaluate.py (data/predictions.npz) for the main pipeline
(ensemble + AF rule + fusion gate). The reliability threshold is chosen on DS1 patients only, from
out-of-fold logistic regression predictions; the three test sets are only used to report results.
Outputs: results/reliability.md, results/reliability.json, results/reliability.png, results/false_fusion.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.af import apply_af_rule
from src.data import AAMI, AFTER, BEFORE, DATA_DIR, MITDB_DIR, N_LEAD_A_FEATURES, load_splits, read_record
from src.metrics import report
from src.reliability import (ACC_TARGET, CONSISTENCY_COL, TAU, burden_category, choose_threshold, fusion_gate,
                             per_recording)

RESULTS = Path(__file__).resolve().parent / "results"
NAVY, BLUE, SKY, AMBER = "#0A2E5C", "#1565C0", "#E3F0FF", "#F4A100"
SET_STYLE = {"DS2": (NAVY, "o", "MIT-BIH DS2 (22)"), "SVDB": (BLUE, "s", "SVDB (78)"),
             "INCART": (AMBER, "^", "INCART (75)")}


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def ds1_out_of_fold(d):
    """Out-of-fold LR probabilities for the 18 DS1 training patients (6 patient groups) + LR on the 4 validation patients."""
    A = N_LEAD_A_FEATURES
    f, y, rec = d["train_f"][:, :A], d["train_y"], d["train_rec"]
    oof = np.zeros((len(y), 4))
    for tr, te in GroupKFold(6).split(f, y, rec):
        m = lr_model().fit(f[tr], y[tr])
        p = np.zeros((len(te), 4))
        p[:, m.classes_] = m.predict_proba(f[te])
        oof[te] = p
    val = lr_model().fit(f, y).predict_proba(d["val_f"][:, :A])
    return (np.concatenate([oof, val]), np.concatenate([y, d["val_y"]]),
            np.concatenate([rec, d["val_rec"]]).astype(str), np.concatenate([d["train_f"], d["val_f"]]))


def load_predictions():
    z = np.load(DATA_DIR / "predictions.npz")
    sets = {}
    for k in z.files:
        name, key = k.split("__")
        sets.setdefault(name, {})[key] = z[k]
    return {n: sets[n] for n in ["DS2", "SVDB", "INCART"] if n in sets}


def beat_symbols(rec):
    """Original MIT-BIH beat symbols of the beats kept for a record (same order as the beat arrays)."""
    xa, xb, ref, syms, _ = read_record(MITDB_DIR, int(rec), "mitdb")
    lab = np.array([AAMI.get(s_, -1) for s_ in syms])
    ks = [k for k in range(1, len(ref) - 1) if ref[k] - BEFORE >= 0 and ref[k] + AFTER <= min(len(xa), len(xb))]
    return np.array(syms)[ks][lab[ks] >= 0]


SYMBOL_NAMES = {"N": "normal", "R": "right bundle branch block", "L": "left bundle branch block", "e": "atrial escape",
                "j": "nodal escape"}


def pooled_accuracy(rows, mask):
    n = np.array([r["beats"] for r in rows])[mask]
    a = np.array([r["accuracy"] for r in rows])[mask]
    return float((a * n).sum() / n.sum()) if mask.any() else float("nan")


def main():
    d = load_splits()
    sets = load_predictions()
    out, md = {}, []

    # ---- 1. DS1: choose the reliability threshold; check the fusion gate does no harm on training patients
    prob, y, rec, f = ds1_out_of_fold(d)
    raw = prob.argmax(1)
    gated = fusion_gate(prob, raw, rec)
    ds1_rows = per_recording(y, rec, gated, f[:, CONSISTENCY_COL])
    threshold, grid = choose_threshold(ds1_rows)
    r0, r1 = report(y, raw), report(y, gated)
    alphas = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    scale = lambda p, a: p * np.array([1, 1, 1, a])
    alpha_ds1 = {a: report(y, scale(prob, a).argmax(1))["macro_f1"] for a in alphas}
    best_alpha = max(alpha_ds1, key=alpha_ds1.get)
    out["ds1"] = {"threshold": threshold, "youden_grid": grid,
                  "gate_check": {"macro_f1": [r0["macro_f1"], r1["macro_f1"]],
                                 "F_sensitivity": [r0["per_class"]["F"]["sensitivity"], r1["per_class"]["F"]["sensitivity"]],
                                 "F_ppv": [r0["per_class"]["F"]["ppv"], r1["per_class"]["F"]["ppv"]]},
                  "f_scaling": {"ds1_macro_f1": alpha_ds1, "chosen_alpha": best_alpha}}

    # ---- 2. test sets
    gate_rows, rel_rows, burden_rows, all_rows = [], [], [], {}
    for name, s in sets.items():
        before = apply_af_rule(s["prob"].argmax(1), s["af_pred"])
        after = s["pred"]
        y = s["y"]
        rb, ra = report(y, before), report(y, after)
        gate_rows.append((name, rb, ra, int(((before == 3) & (y != 3)).sum()), int(((after == 3) & (y != 3)).sum())))
        rs = report(y, apply_af_rule(scale(s["prob"], best_alpha).argmax(1), s["af_pred"]))
        out.setdefault("f_scaling_test", {})[name] = {"macro_f1": rs["macro_f1"], "F": rs["per_class"]["F"]}

        rows = per_recording(y, s["rec"], after, s["consistency"], s["af_pred"])
        all_rows[name] = rows
        c = np.array([r["consistency"] for r in rows])
        acc = np.array([r["accuracy"] for r in rows])
        flag, bad = c < threshold, acc < ACC_TARGET
        auc = roc_auc_score(bad, -c) if 0 < bad.sum() < len(bad) else float("nan")
        rel = {"recordings": len(rows), "flagged": int(flag.sum()), "failing": int(bad.sum()),
               "failing_flagged": int((flag & bad).sum()), "false_flags": int((flag & ~bad).sum()),
               "accuracy_flagged": pooled_accuracy(rows, flag), "accuracy_trusted": pooled_accuracy(rows, ~flag),
               "auroc": float(auc)}
        rel_rows.append((name, rel))
        agree = {}
        for c_ in ["V", "S"]:
            ok = np.array([burden_category(r[f"true_{c_}_burden"]) == burden_category(r[f"pred_{c_}_burden"]) for r in rows])
            agree[c_] = {"all": float(ok.mean()), "trusted": float(ok[~flag].mean()) if (~flag).any() else float("nan"),
                         "flagged": float(ok[flag].mean()) if flag.any() else float("nan")}
        burden_rows.append((name, agree))
        out[name] = {"fusion_gate": {"before": {k: rb[k] for k in ["accuracy", "macro_f1"]} | {"F": rb["per_class"]["F"]},
                                     "after": {k: ra[k] for k in ["accuracy", "macro_f1"]} | {"F": ra["per_class"]["F"]},
                                     "false_F": [gate_rows[-1][3], gate_rows[-1][4]]},
                     "reliability": rel, "burden_category_agreement": agree, "recordings": rows}
    # ---- which N beats become false F? (DS2, before the gate, original MIT-BIH symbols)
    if "DS2" in sets:
        s = sets["DS2"]
        pre_gate = apply_af_rule(s["prob"].argmax(1), s["af_pred"])
        out["false_F_by_symbol"] = {}
        for r in np.unique(s["rec"]):
            m = s["rec"] == r
            fF = (pre_gate[m] == 3) & (s["y"][m] == 0)
            if fF.sum() < 100:
                continue
            sym = beat_symbols(r)
            n_beats = sym[s["y"][m] == 0]
            out["false_F_by_symbol"][str(r)] = {
                "false_F": int(fF.sum()),
                "N_class_beats": {k: int((n_beats == k).sum()) for k in np.unique(n_beats)},
                "false_F_beats": {k: int((sym[fF] == k).sum()) for k in np.unique(sym[fF])}}
    (RESULTS / "reliability.json").write_text(json.dumps(out, indent=2))

    # ---- markdown
    g = out["ds1"]["gate_check"]
    md += ["# Recording-level analysis", "",
           "Main pipeline = ensemble + AF rule + fusion gate. Generated by `analysis.py`.", "",
           "## 1. Fusion gate", "",
           f"F predictions are dropped in recordings where fewer than {TAU:.0%} of beats are predicted V "
           "(a fusion beat needs a ventricular beat). The threshold is fixed in advance.", "",
           f"Check on DS1 (out-of-fold logistic regression): macro-F1 {g['macro_f1'][0]:.3f} → {g['macro_f1'][1]:.3f}, "
           f"F Se {g['F_sensitivity'][0]:.1%} → {g['F_sensitivity'][1]:.1%}, F +P {g['F_ppv'][0]:.1%} → {g['F_ppv'][1]:.1%}.", "",
           "| Test set | False F alarms | Accuracy | Macro-F1 | F Se | F +P |", "|---|---|---|---|---|---|"]
    for name, rb, ra, fb, fa in gate_rows:
        md.append(f"| {name} | {fb:,} → {fa:,} | {rb['accuracy']:.1%} → {ra['accuracy']:.1%} | "
                  f"{rb['macro_f1']:.3f} → {ra['macro_f1']:.3f} | {rb['per_class']['F']['sensitivity']:.1%} → "
                  f"{ra['per_class']['F']['sensitivity']:.1%} | {rb['per_class']['F']['ppv']:.1%} → {ra['per_class']['F']['ppv']:.1%} |")
    fs = out["ds1"]["f_scaling"]
    md += ["", f"**What did not work: rescaling the F probability.** Multiplying the F probability by a factor chosen on DS1 "
           f"(out-of-fold, best factor {fs['chosen_alpha']}: DS1 macro-F1 {fs['ds1_macro_f1'][1]:.3f} → "
           f"{fs['ds1_macro_f1'][fs['chosen_alpha']]:.3f}) does not transfer:", "",
           "| Test set | Macro-F1 (AF rule, no gate) | Macro-F1 with F rescaled | F Se with F rescaled |", "|---|---|---|---|"]
    for name, rb, ra, fb, fa in gate_rows:
        t_ = out["f_scaling_test"][name]
        md.append(f"| {name} | {rb['macro_f1']:.3f} | {t_['macro_f1']:.3f} | {t_['F']['sensitivity']:.1%} |")
    if "false_F_by_symbol" in out:
        md += ["", "**Which beats become false F alarms?** AAMI's N class merges several conduction patterns. "
               "DS2 recordings with at least 100 false F alarms (before the gate), by original MIT-BIH beat type:", "",
               "| Record | N-class beats in the recording | Beats called F |", "|---|---|---|"]
        fmt = lambda d_: ", ".join(f"{v:,} {SYMBOL_NAMES.get(k, k)}" for k, v in sorted(d_.items(), key=lambda kv: -kv[1]))
        for r, v in sorted(out["false_F_by_symbol"].items(), key=lambda kv: -kv[1]["false_F"]):
            md.append(f"| {r} | {fmt(v['N_class_beats'])} | {fmt(v['false_F_beats'])} |")
    md += ["", "## 2. Reliability flag", "",
           f"Consistency = median correlation between each beat and the patient's median beat (lead A). "
           f"A recording is flagged when consistency < **{threshold}** (chosen on DS1 patients only, "
           f"Youden's J for accuracy < {ACC_TARGET:.0%}). No labels and no model are needed to compute it.", "",
           "| Test set | Recordings | Flagged | Failing (acc < 90%) | Failing and flagged | Flagged but fine | "
           "Accuracy, trusted | Accuracy, flagged | AUROC |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rel_rows:
        md.append(f"| {name} | {r['recordings']} | {r['flagged']} | {r['failing']} | {r['failing_flagged']} | "
                  f"{r['false_flags']} | {r['accuracy_trusted']:.1%} | {r['accuracy_flagged']:.1%} | {r['auroc']:.2f} |")
    md += ["", "## 3. Recording-level burden", "",
           "Does the predicted ectopic burden fall in the same category as the true one (<1%, 1–10%, ≥10% of beats)?", "",
           "| Test set | V burden: all | V: trusted | V: flagged | S burden: all | S: trusted | S: flagged |",
           "|---|---|---|---|---|---|---|"]
    for name, a in burden_rows:
        md.append(f"| {name} | {a['V']['all']:.0%} | {a['V']['trusted']:.0%} | {a['V']['flagged']:.0%} | "
                  f"{a['S']['all']:.0%} | {a['S']['trusted']:.0%} | {a['S']['flagged']:.0%} |")
    if "DS2" in all_rows:
        md += ["", "## DS2 recording summaries", "",
               "What a reader would see first for each recording. Burdens are % of beats (predicted / true).", "",
               "| Record | Beats | Consistency | Status | AF burden | V burden | S burden | Beat accuracy |",
               "|---|---|---|---|---|---|---|---|"]
        for r in sorted(all_rows["DS2"], key=lambda r: r["consistency"]):
            status = "⚠ check" if r["consistency"] < threshold else "ok"
            md.append(f"| {r['record']} | {r['beats']} | {r['consistency']:.3f} | {status} | {r['af_burden']:.0%} | "
                      f"{r['pred_V_burden']:.1%} / {r['true_V_burden']:.1%} | {r['pred_S_burden']:.1%} / {r['true_S_burden']:.1%} | "
                      f"{r['accuracy']:.1%} |")
    (RESULTS / "reliability.md").write_text("\n".join(md) + "\n")

    # ---- figure: consistency vs accuracy
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, rows in all_rows.items():
        col, mk, label = SET_STYLE[name]
        x = np.clip([1 - r["consistency"] for r in rows], 1e-3, None)
        ax.scatter(x, [r["accuracy"] for r in rows], s=28, c=col, marker=mk, alpha=0.8, label=label, edgecolors="white", linewidths=0.4)
    ax.set_xscale("log")
    ax.axvline(1 - threshold, color="#555", ls="--", lw=1)
    ax.axhline(ACC_TARGET, color="#bbb", lw=0.8)
    ax.text(1 - threshold, 0.02, f"  flag: consistency < {threshold}\n  (chosen on DS1)", fontsize=8, color="#333", va="bottom")
    ax.set_xlabel("1 − beat consistency (log scale)   ·   consistency = median correlation of a beat with the patient's median beat",
                  fontsize=8.5)
    ax.set_ylabel("beat accuracy of the recording")
    ax.set_ylim(0, 1.02)
    ax.set_title("Low beat consistency predicts which recordings the model gets wrong — no labels needed", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "reliability.png", dpi=150)
    plt.close(fig)

    # ---- figure: false fusion alarms vs real fusion (DS2, lead A)
    if "DS2" in sets:
        s = sets["DS2"]
        x = np.load(DATA_DIR / "mitdb_annotated.npz")["test_x"][:, 0].astype(np.float32)
        pre_gate = apply_af_rule(s["prob"].argmax(1), s["af_pred"])
        t = (np.arange(x.shape[1]) - 90) / 360 * 1000
        panels = [("117", (s["y"] == 0) & (pre_gate == 3), "record 117: normal beats called F"),
                  ("212", (s["y"] == 0) & (pre_gate == 3), "record 212: normal beats called F"),
                  ("213", s["y"] == 3, "record 213: real fusion beats")]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
        rng = np.random.default_rng(0)
        for ax, (r, sel, title) in zip(axes, panels):
            m = s["rec"] == r
            n_v = int((s["y"][m] == 2).sum())
            note = f"{n_v} V beats in the recording" if n_v else "no V beats in the recording"
            tmpl = np.median(x[m], 0)
            idx = np.flatnonzero(m & sel)
            for i in rng.choice(idx, min(40, len(idx)), replace=False):
                ax.plot(t, x[i], color=AMBER, alpha=0.25, lw=0.8)
            ax.plot(t, tmpl, color=NAVY, lw=2, label="patient's median beat")
            ax.plot([], [], color=AMBER, lw=1.5, label="beats called F")
            ax.set_title(f"{title}\n({len(idx)} beats, {note})", fontsize=9)
            ax.set_xlabel("ms from R peak")
            ax.set_xlim(-250, 460)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("lead A (z-normalised)")
        axes[0].legend(fontsize=7, frameon=False, loc="upper right")
        fig.suptitle("False fusion alarms look like real fusion beat by beat; the difference is the context "
                     "(real fusion needs ventricular beats)", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(RESULTS / "false_fusion.png", dpi=150)
        plt.close(fig)

    print("\n".join(md))


if __name__ == "__main__":
    main()
