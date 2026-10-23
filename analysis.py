"""Recording-level analysis: false fusion alarms, conduction patterns, and when (not) to trust a result.

Questions:
  1. Does the fusion gate still help, and which beats are still called F by mistake?
  2. Which conduction patterns does the multi-template step find, and are they real?
     (No pattern may be made of ventricular beats, or V beats would start to look normal.)
  3. Can we tell, without labels, which recordings the model will get wrong, and why?
  4. At recording level, does the pipeline get the ectopic burden right?

Uses the per-beat predictions saved by evaluate.py (data/predictions.npz) for the main pipeline
(ensemble + AF rule + fusion gate). Every threshold is chosen on DS1 patients only, from out-of-fold
logistic regression predictions; the three test sets are only used to report results.
Outputs: results/reliability.md, results/reliability.json, results/reliability.png, results/templates.png
"""
import json
from collections import Counter
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
from src.data import (AAMI, AFTER, BEFORE, DATA_DIR, MITDB_DIR, N_LEAD_A_FEATURES, beats_path, load_splits,
                      read_record)
from src.metrics import report
from src.reliability import (ACC_TARGET, FIT_COL, RR_PRE_COL, TAU, burden_category, causes, choose_threshold,
                             fusion_gate, per_recording)

RESULTS = Path(__file__).resolve().parent / "results"
NAVY, BLUE, SKY, AMBER = "#0A2E5C", "#1565C0", "#E3F0FF", "#F4A100"
SET_STYLE = {"DS2": (NAVY, "o", "MIT-BIH DS2 (22)"), "SVDB": (BLUE, "s", "SVDB (78)"),
             "INCART": (AMBER, "^", "INCART (75)")}
GRIDS = {"unstable": (True, np.round(np.arange(0.01, 0.501, 0.01), 2)),
         "early": (True, np.round(np.arange(0.02, 0.501, 0.01), 2)),
         "consistency": (False, np.round(np.arange(0.80, 0.9951, 0.005), 3))}
SYMBOL_NAMES = {"N": "normal", "R": "right bundle branch block", "L": "left bundle branch block", "e": "atrial escape",
                "j": "nodal escape", "A": "atrial premature", "a": "aberrated atrial premature", "J": "nodal premature",
                "S": "supraventricular premature", "V": "ventricular", "E": "ventricular escape", "F": "fusion"}


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def ds1_out_of_fold(d):
    """Out-of-fold LR probabilities for the 18 DS1 training patients (6 patient groups) + LR on the 4 validation
    patients, with everything the per-recording signals need."""
    A = N_LEAD_A_FEATURES
    f, y, rec = d["train_f"][:, :A], d["train_y"], d["train_rec"]
    oof = np.zeros((len(y), 4))
    for tr, te in GroupKFold(6).split(f, y, rec):
        m = lr_model().fit(f[tr], y[tr])
        p = np.zeros((len(te), 4))
        p[:, m.classes_] = m.predict_proba(f[te])
        oof[te] = p
    val = lr_model().fit(f, y).predict_proba(d["val_f"][:, :A])
    cat = lambda k: np.concatenate([d[f"train_{k}"], d[f"val_{k}"]])
    return {"prob": np.concatenate([oof, val]), "y": cat("y"), "rec": cat("rec").astype(str),
            "fit": cat("f")[:, FIT_COL], "rr_pre": cat("f")[:, RR_PRE_COL], "consistency": cat("consist"),
            "n_patterns": cat("n_tmpl")}


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


def fmt_symbols(counts, top=3):
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{v:,} {SYMBOL_NAMES.get(k, k)}" for k, v in items)


def pooled_accuracy(rows, mask):
    n = np.array([r["beats"] for r in rows])[mask]
    a = np.array([r["accuracy"] for r in rows])[mask]
    return float((a * n).sum() / n.sum()) if mask.any() else float("nan")


def flag_stats(rows, flag):
    bad = np.array([r["accuracy"] for r in rows]) < ACC_TARGET
    return {"flagged": int(flag.sum()), "failing": int(bad.sum()), "failing_flagged": int((flag & bad).sum()),
            "false_flags": int((flag & ~bad).sum()), "accuracy_flagged": pooled_accuracy(rows, flag),
            "accuracy_trusted": pooled_accuracy(rows, ~flag)}


def auroc(rows, key, higher_is_worse):
    bad = np.array([r["accuracy"] for r in rows]) < ACC_TARGET
    v = np.array([r[key] for r in rows]) * (1 if higher_is_worse else -1)
    return float(roc_auc_score(bad, v)) if 0 < bad.sum() < len(bad) else float("nan")


def main():
    d = load_splits()
    sets = load_predictions()
    out, md = {}, []

    # ---- DS1 (training patients, out-of-fold): gate check, F rescaling, reliability thresholds
    o = ds1_out_of_fold(d)
    raw = o["prob"].argmax(1)
    gated = fusion_gate(o["prob"], raw, o["rec"])
    ds1_rows = per_recording(o["y"], o["rec"], gated, o["fit"], o["rr_pre"], o["consistency"], o["n_patterns"])
    thresholds = {k: choose_threshold(ds1_rows, k, hw, grid)[0] for k, (hw, grid) in GRIDS.items()}
    r0, r1 = report(o["y"], raw), report(o["y"], gated)
    alphas = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    scale = lambda p, a: p * np.array([1, 1, 1, a])
    alpha_ds1 = {a: report(o["y"], scale(o["prob"], a).argmax(1))["macro_f1"] for a in alphas}
    best_alpha = max(alpha_ds1, key=alpha_ds1.get)
    out["ds1"] = {"thresholds": thresholds,
                  "gate_check": {"macro_f1": [r0["macro_f1"], r1["macro_f1"]],
                                 "F_sensitivity": [r0["per_class"]["F"]["sensitivity"], r1["per_class"]["F"]["sensitivity"]],
                                 "F_ppv": [r0["per_class"]["F"]["ppv"], r1["per_class"]["F"]["ppv"]]},
                  "f_scaling": {"ds1_macro_f1": alpha_ds1, "chosen_alpha": best_alpha}}

    # ---- conduction patterns found on MIT-BIH (checked against the original beat annotations)
    patterns = {}
    for split in ["train", "val", "test"]:
        for r in np.unique(d[f"{split}_rec"]):
            m = d[f"{split}_rec"] == r
            n_pat = int(d[f"{split}_n_tmpl"][m][0])
            if n_pat < 2:
                continue
            sym, tm = beat_symbols(r), d[f"{split}_tmpl"][m]
            patterns[str(r)] = {"split": {"train": "DS1", "val": "DS1 hold-out", "test": "DS2"}[split],
                                "patterns": [dict(Counter(sym[tm == k].tolist())) for k in range(n_pat)]}
    v_share = [p_.get("V", 0) / max(sum(p_.values()), 1) for v in patterns.values() for p_ in v["patterns"]]
    out["patterns"] = {"recordings": patterns, "max_ventricular_share_of_a_pattern": float(max(v_share, default=0))}

    # ---- test sets
    gate_rows, all_rows = [], {}
    for name, s in sets.items():
        before = apply_af_rule(s["prob"].argmax(1), s["af_pred"])
        after, y = s["pred"], s["y"]
        rb, ra = report(y, before), report(y, after)
        gate_rows.append((name, rb, ra, int(((before == 3) & (y != 3)).sum()), int(((after == 3) & (y != 3)).sum())))
        rs = report(y, apply_af_rule(scale(s["prob"], best_alpha).argmax(1), s["af_pred"]))

        rows = per_recording(y, s["rec"], after, s["fit"], s["rr_pre"], s["consistency"], s["n_patterns"], s["af_pred"])
        for r in rows:
            r["causes"] = causes(r, thresholds)
            r["warning"] = r["unstable"] > thresholds["unstable"]
        all_rows[name] = rows
        warn = np.array([r["warning"] for r in rows])
        old = np.array([r["consistency"] < thresholds["consistency"] for r in rows])
        early = np.array([r["early"] > thresholds["early"] for r in rows])
        multi = np.array([r["patterns"] > 1 for r in rows])
        groups = {"no warning, no early beats": ~warn & ~early, "frequent early beats only": ~warn & early,
                  "unstable beats": warn}
        agree = {}
        for c_ in ["V", "S"]:
            ok = np.array([burden_category(r[f"true_{c_}_burden"]) == burden_category(r[f"pred_{c_}_burden"])
                           for r in rows])
            agree[c_] = {"all": float(ok.mean()), "trusted": float(ok[~warn].mean()) if (~warn).any() else float("nan"),
                         "warned": float(ok[warn].mean()) if warn.any() else float("nan")}
        out[name] = {
            "fusion_gate": {"before": {k: rb[k] for k in ["accuracy", "macro_f1"]} | {"F": rb["per_class"]["F"]},
                            "after": {k: ra[k] for k in ["accuracy", "macro_f1"]} | {"F": ra["per_class"]["F"]},
                            "false_F": [gate_rows[-1][3], gate_rows[-1][4]]},
            "f_scaling": {"macro_f1": rs["macro_f1"], "F": rs["per_class"]["F"]},
            "warning": flag_stats(rows, warn) | {"auroc": auroc(rows, "unstable", True)},
            "consistency_flag": flag_stats(rows, old) | {"auroc": auroc(rows, "consistency", False)},
            "groups": {g: {"recordings": int(m.sum()), "accuracy": pooled_accuracy(rows, m)} for g, m in groups.items()},
            "several_patterns": {"recordings": int(multi.sum()), "accuracy": pooled_accuracy(rows, multi)},
            "burden_category_agreement": agree, "recordings": rows}

    # ---- remaining false F alarms on DS2 (main pipeline, before the gate), by original beat type
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
            out["false_F_by_symbol"][str(r)] = {"false_F": int(fF.sum()),
                                                "false_F_beats": dict(Counter(sym[fF].tolist()))}
    (RESULTS / "reliability.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- markdown
    g, th = out["ds1"]["gate_check"], thresholds
    md += ["# Recording-level analysis", "",
           "Main pipeline = ensemble + AF rule + fusion gate, on multi-template features. Generated by `analysis.py`.",
           "Every threshold below is chosen on DS1 patients only (out-of-fold logistic regression).", "",
           "## 1. Conduction patterns (multi-template features)", "",
           "Recordings where more than one beat pattern was found, with the original MIT-BIH beat types in each pattern "
           "(labels are used only for this check, never to find the patterns):", "",
           "| Record | Set | Pattern 1 | Pattern 2 | Pattern 3 |", "|---|---|---|---|---|"]
    for r, v in patterns.items():
        cells = [fmt_symbols(p_) for p_ in v["patterns"]] + ["–"] * (3 - len(v["patterns"]))
        md.append(f"| {r} | {v['split']} | " + " | ".join(cells) + " |")
    md += ["", f"Largest share of ventricular beats in any pattern: {out['patterns']['max_ventricular_share_of_a_pattern']:.1%} "
           "(V beats never form a pattern of their own).", "",
           "## 2. Fusion gate", "",
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
           f"(best factor {fs['chosen_alpha']}: DS1 macro-F1 {fs['ds1_macro_f1'][1]:.3f} → "
           f"{fs['ds1_macro_f1'][fs['chosen_alpha']]:.3f}) does not transfer:", "",
           "| Test set | Macro-F1 (AF rule, no gate) | Macro-F1 with F rescaled | F Se with F rescaled |", "|---|---|---|---|"]
    for name, rb, ra, fb, fa in gate_rows:
        t_ = out[name]["f_scaling"]
        md.append(f"| {name} | {rb['macro_f1']:.3f} | {t_['macro_f1']:.3f} | {t_['F']['sensitivity']:.1%} |")
    if out.get("false_F_by_symbol"):
        md += ["", "Remaining DS2 recordings with at least 100 false F alarms (before the gate), by original beat type:", "",
               "| Record | False F alarms | Beat types called F |", "|---|---|---|"]
        for r, v in sorted(out["false_F_by_symbol"].items(), key=lambda kv: -kv[1]["false_F"]):
            md.append(f"| {r} | {v['false_F']:,} | {fmt_symbols(v['false_F_beats'])} |")

    md += ["", "## 3. Reliability warning and its causes", "",
           f"- **Unstable beats** (the warning): share of on-time beats that match none of the patient's own beat patterns "
           f"(correlation < 0.9). Warning above **{th['unstable']:.0%}**.",
           f"- **Frequent early beats** (information): share of beats arriving > 15% early. Shown above **{th['early']:.0%}**.",
           "- **Several beat patterns** (information): the patient switches between beat shapes.",
           f"- For comparison, the first version of the flag: median correlation with a single median beat < **{th['consistency']}**.", "",
           "A recording \"fails\" when its beat accuracy is below 90%. No labels and no model are needed to compute any signal.", "",
           "| Test set | Signal | Flagged | Failing | Failing and flagged | Flagged but fine | Accuracy, not flagged | Accuracy, flagged | AUROC |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name in sets:
        for label, key in [("first version: consistency", "consistency_flag"), ("**unstable beats**", "warning")]:
            r = out[name][key]
            md.append(f"| {name} | {label} | {r['flagged']} | {r['failing']} | {r['failing_flagged']} | {r['false_flags']} | "
                      f"{r['accuracy_trusted']:.1%} | {r['accuracy_flagged']:.1%} | {r['auroc']:.2f} |")
    md += ["", "Accuracy by cause (pooled over beats):", "",
           "| Test set | No warning, no early beats | Frequent early beats only | Unstable beats | Several beat patterns |",
           "|---|---|---|---|---|"]
    for name in sets:
        gr = out[name]["groups"]
        cell = lambda v: f"{v['accuracy']:.1%} ({v['recordings']})" if v["recordings"] else "– (0)"
        md.append(f"| {name} | {cell(gr['no warning, no early beats'])} | {cell(gr['frequent early beats only'])} | "
                  f"{cell(gr['unstable beats'])} | {cell(out[name]['several_patterns'])} |")
    md += ["", "## 4. Recording-level burden", "",
           "Does the predicted ectopic burden fall in the same category as the true one (<1%, 1–10%, ≥10% of beats)?", "",
           "| Test set | V burden: all | V: no warning | V: warning | S burden: all | S: no warning | S: warning |",
           "|---|---|---|---|---|---|---|"]
    for name in sets:
        a = out[name]["burden_category_agreement"]
        md.append(f"| {name} | {a['V']['all']:.0%} | {a['V']['trusted']:.0%} | {a['V']['warned']:.0%} | "
                  f"{a['S']['all']:.0%} | {a['S']['trusted']:.0%} | {a['S']['warned']:.0%} |")
    if "DS2" in all_rows:
        md += ["", "## DS2 recording summaries", "",
               "What a reader would see first for each recording. Burdens are % of beats (predicted / true).", "",
               "| Record | Beats | Status | Notes | AF burden | V burden | S burden | Beat accuracy |",
               "|---|---|---|---|---|---|---|---|"]
        for r in sorted(all_rows["DS2"], key=lambda r: -r["unstable"]):
            status = "⚠ check" if r["warning"] else "ok"
            notes = ", ".join(r["causes"]) or "–"
            md.append(f"| {r['record']} | {r['beats']} | {status} | {notes} | {r['af_burden']:.0%} | "
                      f"{r['pred_V_burden']:.1%} / {r['true_V_burden']:.1%} | {r['pred_S_burden']:.1%} / {r['true_S_burden']:.1%} | "
                      f"{r['accuracy']:.1%} |")
    (RESULTS / "reliability.md").write_text("\n".join(md) + "\n")

    # ---- figure: first-version flag vs unstable-beats warning
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [("consistency", lambda r: max(1 - r["consistency"], 1e-3), 1 - th["consistency"],
               "first version: 1 − consistency with a single median beat (log)"),
              ("unstable", lambda r: max(r["unstable"], 1e-3), th["unstable"],
               "unstable beats: share of on-time beats matching no pattern (log)")]
    for ax, (key, fx, line, xlabel) in zip(axes, panels):
        for name, rows in all_rows.items():
            col, mk, label = SET_STYLE[name]
            ax.scatter([fx(r) for r in rows], [r["accuracy"] for r in rows], s=26, c=col, marker=mk, alpha=0.8,
                       label=label, edgecolors="white", linewidths=0.4)
        ax.set_xscale("log")
        ax.axvline(line, color="#555", ls="--", lw=1)
        ax.axhline(ACC_TARGET, color="#bbb", lw=0.8)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.grid(alpha=0.25)
        aucs = " · ".join(f"{n} {out[n]['warning' if key == 'unstable' else 'consistency_flag']['auroc']:.2f}" for n in all_rows)
        ax.set_title(f"AUROC for failing recordings: {aucs}", fontsize=8.5)
    axes[0].set_ylabel("beat accuracy of the recording")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Which recordings will the model get wrong? Two label-free signals (dashed: threshold chosen on DS1)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "reliability.png", dpi=150)
    plt.close(fig)

    # ---- figure: beat patterns found in two DS2 recordings + a V-heavy recording with a single pattern
    x = np.load(beats_path("mitdb"))["test_x"][:, 0].astype(np.float32)
    t = (np.arange(x.shape[1]) - BEFORE) / 360 * 1000
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    colors = [NAVY, AMBER, BLUE]
    for ax, r in zip(axes, ["212", "231", "233"]):
        m = d["test_rec"].astype(str) == r
        if not m.any():
            continue
        tm, sym = d["test_tmpl"][m], beat_symbols(r)
        xs = x[m]
        for k in range(int(d["test_n_tmpl"][m][0])):
            idx = np.flatnonzero(tm == k)
            for i in rng.choice(idx, min(25, len(idx)), replace=False):
                ax.plot(t, xs[i], color=colors[k], alpha=0.12, lw=0.8)
            top = Counter(sym[idx].tolist()).most_common(1)[0][0]
            ax.plot(t, np.median(xs[idx], 0), color=colors[k], lw=2,
                    label=f"pattern {k + 1}: {len(idx):,} beats, mostly {SYMBOL_NAMES.get(top, top)}")
        n_pat = int(d["test_n_tmpl"][m][0])
        ax.set_title(f"record {r}: {n_pat} beat pattern{'s' if n_pat > 1 else ''} found", fontsize=9)
        ax.set_xlabel("ms from R peak")
        ax.set_xlim(-250, 460)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("lead A (z-normalised)")
    fig.suptitle("Beat patterns found without labels: conduction changes are separated, frequent V beats are not "
                 "turned into a 'normal' pattern", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "templates.png", dpi=150)
    plt.close(fig)

    print("\n".join(md))


if __name__ == "__main__":
    main()
