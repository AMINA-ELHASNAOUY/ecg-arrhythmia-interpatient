"""Three extensions of the main study (run after evaluate.py and `train.py --realtime`):

1. Noise stress test. Does the unstable-beats warning switch on when a recording gets noisy, and does accuracy
   fall at the same time? MIT-BIH Noise Stress Test Database: record 118 (a DS1 hold-out patient, never trained on)
   with electrode-motion noise added at 24, 18, 12, 6, 0 and -6 dB SNR.
2. More AF patients for the rhythm model. The AF detector is retrained with the 23 recordings of the MIT-BIH Atrial
   Fibrillation Database added to the DS1 patients, and the joint rhythm-and-beat model is tried again with the
   AF probability of a detector trained on the AF database only (so no DS1 patient is seen twice).
3. Real-time version. Every feature uses past beats only, plus the next R peak (one-beat delay); beat patterns are
   re-estimated every 50 beats from the previous 300; the fusion gate uses the V fraction of the beats seen so far.

Nothing here changes the main pipeline. Thresholds and weights are chosen on DS1 patients only, as in evaluate.py.
Outputs: results/extensions.md, results/extensions.json, results/noise_stress.png
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

from evaluate import cnn_proba, device
from src.af import AFDetector, af_metrics, apply_af_rule
from src.data import (AFDB_DIR, AFDB_RECORDS, DATA_DIR, INCART_DIR, N_LEAD_A_FEATURES, NSTDB_DIR, NSTDB_RECORDS,
                      load_external, load_splits, rhythm_only)
from src.joint import JointModel
from src.metrics import macro_f1, report
from src.model import BeatNet
from src.reliability import FIT_COL, RR_PRE_COL, fusion_gate, recording_signals

RESULTS = Path(__file__).resolve().parent / "results"
NAVY, BLUE, AMBER = "#0A2E5C", "#1565C0", "#F4A100"
A = N_LEAD_A_FEATURES
SETS = ["DS2", "SVDB", "INCART"]


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def load_cnns(tag, dev):
    models = []
    for p in sorted(RESULTS.glob(f"beatnet{tag}_seed*.pt")):
        m = BeatNet(in_ch=2, n_feat=A).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        models.append(m)
    return models


class Pipeline:
    """The main pipeline (lead A): ensemble of logistic regression + CNN seeds, AF rule, fusion gate."""

    def __init__(self, d, tag, dev, weight=None):
        self.dev = dev
        self.lr = lr_model().fit(d["train_f"][:, :A], d["train_y"])
        self.cnns = load_cnns(tag, dev)
        norm = np.load(RESULTS / f"feature_norm{tag}.npz")
        self.mu, self.sd = norm["mu"][:A], norm["sd"][:A]
        self.af = AFDetector().fit(d["train_r"], d["train_af"], d["train_rec"])
        if weight is None:                        # same rule as evaluate.py: N/S/V macro-F1 on the DS1 hold-out
            pl, pc = self.lr.predict_proba(d["val_f"][:, :A]), self.cnn(d["val_x"], d["val_f"])
            grid = np.round(np.arange(0, 1.0001, 0.05), 2)
            weight = float(grid[int(np.argmax([macro_f1(d["val_y"], (w * pl + (1 - w) * pc).argmax(1), ["N", "S", "V"])
                                               for w in grid]))])
        self.w = weight

    def cnn(self, x, f):
        f = ((f[:, :A] - self.mu) / self.sd).astype(np.float32)
        return cnn_proba(self.cnns, np.ascontiguousarray(x[:, :2]), f, self.dev).mean(0)

    def proba(self, x, f):
        return self.w * self.lr.predict_proba(f[:, :A]) + (1 - self.w) * self.cnn(x, f)

    def predict(self, x, f, r, rec, af_pred=None, causal_gate=False):
        p = self.proba(x, f)
        pred = apply_af_rule(p.argmax(1), self.af.predict(r) if af_pred is None else af_pred)
        return (causal_fusion_gate if causal_gate else fusion_gate)(p, pred, rec), p


def causal_fusion_gate(prob, pred, rec, tau=0.01):
    """Real-time fusion gate: F is allowed only once >= tau of the recording's beats so far were predicted V."""
    out = pred.copy()
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        v_frac = np.cumsum(pred[idx] == 2) / np.arange(1, len(idx) + 1)
        block = (out[idx] == 3) & (v_frac < tau)
        out[idx[block]] = prob[idx[block]][:, :3].argmax(1)
    return out


def test_sets(realtime=False):
    d = load_splits(realtime=realtime)
    sets = {"DS2": {k: d[f"test_{k}"] for k in ["x", "f", "r", "y", "rec"]}}
    for db, name in [("svdb", "SVDB"), ("incart", "INCART")]:
        if db == "incart" and not (INCART_DIR / "RECORDS").exists():
            continue
        e = load_external(db, realtime=realtime)
        sets[name] = {k: e[f"ext_{k}"] for k in ["x", "f", "r", "y", "rec"]}
    return d, sets


def summary(rep):
    return {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
            **{f"{c}_{m}": rep["per_class"][c][k] for c in "NSVF" for m, k in [("se", "sensitivity"), ("ppv", "ppv")]}}


# ------------------------------------------------------------------ 1. noise stress test
def noise_test(pipe, d, thresholds):
    if not all((NSTDB_DIR / f"{r}.dat").exists() for r in NSTDB_RECORDS):
        return None
    e = load_external("nstdb")
    rows = []
    clean = d["val_rec"] == 118
    variants = [("clean", {k: d[f"val_{k}"][clean] for k in ["x", "f", "r", "y", "rec", "consist", "n_tmpl"]})]
    for i, r in enumerate(NSTDB_RECORDS):
        m = e["ext_rec"] == i
        variants.append((f"{int(r[4:].replace('_', '-'))} dB", {k: e[f"ext_{k}"][m] for k in
                                                           ["x", "f", "r", "y", "rec", "consist", "n_tmpl"]}))
    for label, v in variants:
        pred, _ = pipe.predict(v["x"], v["f"], v["r"], v["rec"])
        sig = recording_signals(v["f"][:, FIT_COL], v["f"][:, RR_PRE_COL], v["consist"], v["n_tmpl"])
        rep = report(v["y"], pred)
        rows.append({"snr": label, "beats": int(len(v["y"])), "accuracy": rep["accuracy"],
                     "V_se": rep["per_class"]["V"]["sensitivity"], "V_ppv": rep["per_class"]["V"]["ppv"],
                     "unstable": sig["unstable"], "warning": sig["unstable"] > thresholds["unstable"]})
    return rows


# ------------------------------------------------------------------ 2. more AF patients
def afdb_rhythm():
    path = DATA_DIR / "afdb_rhythm.npz"
    if not path.exists():
        if not all((AFDB_DIR / f"{r}.qrs").exists() and (AFDB_DIR / f"{r}.dat").exists() for r in AFDB_RECORDS):
            return None
        parts = {}
        for r in AFDB_RECORDS:
            print(f"  AFDB {r}...", flush=True)
            rr, af = rhythm_only(AFDB_DIR, r, "afdb")
            step = 4                              # every 4th beat: ~10-hour recordings, neighbouring beats overlap
            parts[r] = (rr[::step], af[::step])
        np.savez_compressed(path, r=np.concatenate([p[0] for p in parts.values()]),
                            af=np.concatenate([p[1] for p in parts.values()]),
                            rec=np.concatenate([[k] * len(p[1]) for k, p in parts.items()]))
    z = np.load(path)
    return z["r"], z["af"], z["rec"]


def more_af(pipe, d, sets):
    a = afdb_rhythm()
    if a is None:
        return None
    r_a, af_a, rec_a = a
    out = {"afdb": {"recordings": len(np.unique(rec_a)), "beats_used": int(len(af_a)), "af_beats": int(af_a.sum())},
           "ds1_af_patients": int(len(np.unique(d["train_rec"][d["train_af"]])))}
    both = AFDetector().fit(np.vstack([d["train_r"], r_a]), np.r_[d["train_af"], af_a],
                            np.r_[d["train_rec"].astype(str), rec_a])
    afdb_only = AFDetector().fit(r_a, af_a, rec_a)
    out["detectors"] = {"DS1 only (main pipeline)": pipe.af.summary(), "DS1 + AF database": both.summary()}
    out["af_detection_DS2"] = {"DS1 only (main pipeline)": af_metrics(pipe.af.predict(d["test_r"]), d["test_af"]),
                               "DS1 + AF database": af_metrics(both.predict(d["test_r"]), d["test_af"])}

    # joint model, now with an AF probability from a detector that saw 23 AF patients (none of them in DS1)
    aug = lambda r: np.hstack([r, afdb_only.predict_proba(r)[:, None]])
    joint = JointModel().fit(d["train_f"], aug(d["train_r"]), d["train_y"], d["train_rec"])
    out["joint_model"] = joint.summary()
    lr = pipe.lr
    out["test"] = {}
    for name, s in sets.items():
        p_lr = lr.predict_proba(s["f"][:, :A])
        ens = pipe.proba(s["x"], s["f"])
        rows = {
            "Main pipeline (AF detector: DS1 only)": fusion_gate(ens, apply_af_rule(ens.argmax(1), pipe.af.predict(s["r"])), s["rec"]),
            "Main pipeline (AF detector: DS1 + AF database)": fusion_gate(ens, apply_af_rule(ens.argmax(1), both.predict(s["r"])), s["rec"]),
            "Logistic regression + AF rule (DS1 + AF database) + gate": fusion_gate(p_lr, apply_af_rule(p_lr.argmax(1), both.predict(s["r"])), s["rec"]),
        }
        pj = joint.predict_proba(s["f"], aug(s["r"]))
        rows["Joint model with AF probability + gate"] = fusion_gate(pj, pj.argmax(1), s["rec"])
        out["test"][name] = {k: summary(report(s["y"], v)) for k, v in rows.items()}
    return out


# ------------------------------------------------------------------ 3. real-time version
def realtime(dev, offline):
    if not list(RESULTS.glob("beatnet_realtime_seed*.pt")):
        return None
    d, sets = test_sets(realtime=True)
    pipe = Pipeline(d, "_realtime", dev)
    out = {"ensemble_weight_lr": pipe.w, "cnn_seeds": len(pipe.cnns), "af_detector": pipe.af.summary(),
           "af_detection_DS2": af_metrics(pipe.af.predict(d["test_r"]), d["test_af"]), "test": {}}
    for name, s in sets.items():
        pred, _ = pipe.predict(s["x"], s["f"], s["r"], s["rec"], causal_gate=True)
        out["test"][name] = {"offline (main pipeline)": offline[name], "real-time": summary(report(s["y"], pred))}
    return out


# ------------------------------------------------------------------ main
def main():
    dev = device()
    rel = json.loads((RESULTS / "reliability.json").read_text())
    thresholds = rel["ds1"]["thresholds"]
    w = json.loads((RESULTS / "ensemble.json").read_text())["weight_lr"][""]
    d, sets = test_sets()
    pipe = Pipeline(d, "", dev, weight=w)
    offline = {n: summary(json.loads((RESULTS / f"{n.lower()}_ensemble_af_gate.json").read_text())) for n in sets}

    out = {"noise": noise_test(pipe, d, thresholds)}
    print("noise test done" if out["noise"] else "noise test skipped (run scripts/download_extra.command)", flush=True)
    out["more_af"] = more_af(pipe, d, sets)
    print("AF database part done" if out["more_af"] else "AF database part skipped", flush=True)
    del d, sets
    out["realtime"] = realtime(dev, offline)
    print("real-time part done" if out["realtime"] else "real-time part skipped (run train.py --realtime)", flush=True)
    (RESULTS / "extensions.json").write_text(json.dumps(out, indent=2, default=str))

    md = ["# Extensions", "", "Generated by `extensions.py`. The main pipeline is not changed by anything here.", ""]
    if out["noise"]:
        md += ["## 1. Noise stress test", "",
               f"Record 118 (DS1 hold-out, never trained on) with electrode-motion noise from the MIT-BIH Noise Stress Test "
               f"Database. Noise is added in alternating 2-minute segments after the first 5 minutes. Warning threshold "
               f"(chosen on DS1): {thresholds['unstable']:.0%} unstable beats.", "",
               "| Noise level (SNR) | Beat accuracy | V Se / +P | Unstable beats | Warning |", "|---|---|---|---|---|"]
        for r in out["noise"]:
            md.append(f"| {r['snr']} | {r['accuracy']:.1%} | {r['V_se']:.1%} / {r['V_ppv']:.1%} | {r['unstable']:.1%} | "
                      f"{'⚠ yes' if r['warning'] else 'no'} |")
        fig, ax1 = plt.subplots(figsize=(6.4, 3.4))
        xs = np.arange(len(out["noise"]))
        ax1.plot(xs, [r["accuracy"] for r in out["noise"]], "o-", color=NAVY, label="beat accuracy")
        ax1.set_ylabel("beat accuracy", color=NAVY)
        ax1.set_ylim(0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(xs, [r["unstable"] for r in out["noise"]], "s--", color=AMBER, label="unstable beats")
        ax2.axhline(thresholds["unstable"], color=AMBER, lw=0.8, alpha=0.6)
        ax2.set_ylabel("share of unstable beats", color=AMBER)
        ax1.set_xticks(xs, [r["snr"] for r in out["noise"]])
        ax1.set_xlabel("noise level (signal-to-noise ratio; lower = noisier)")
        ax1.set_title("Record 118 with added electrode-motion noise", fontsize=10)
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / "noise_stress.png", dpi=150)
        plt.close(fig)
    if out["more_af"]:
        a = out["more_af"]
        md += ["", "## 2. Rhythm model with more AF patients", "",
               f"MIT-BIH Atrial Fibrillation Database: {a['afdb']['recordings']} recordings, every 4th beat used "
               f"({a['afdb']['beats_used']:,} beats, {a['afdb']['af_beats']:,} in AF). DS1 has {a['ds1_af_patients']} AF patients.", "",
               "| AF detector | Chosen by leave-patient-out CV | AF Se on DS2 | AF +P on DS2 | Specificity on DS2 |", "|---|---|---|---|---|"]
        for k, v in a["af_detection_DS2"].items():
            det = a["detectors"][k]
            md.append(f"| {k} | {det['chosen'].replace('_', ' ')}, threshold {det['threshold']} | {v['sensitivity']:.1%} | "
                      f"{v['ppv']:.1%} | {v['specificity']:.1%} |")
        md += ["", f"Joint model with the AF probability as an extra input: {a['joint_model']['chosen'].replace('_', ' ')} "
               f"(leave-patient-out macro-F1 on DS1: " +
               ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in a['joint_model']['cv_macro_f1'].items()) + ").", "",
               "| Test set | Model | Accuracy | Macro-F1 | S Se / +P |", "|---|---|---|---|---|"]
        for name, rows in a["test"].items():
            for k, v in rows.items():
                md.append(f"| {name} | {k} | {v['accuracy']:.1%} | {v['macro_f1']:.3f} | {v['S_se']:.1%} / {v['S_ppv']:.1%} |")
    if out["realtime"]:
        rt = out["realtime"]
        md += ["", "## 3. Real-time version", "",
               f"Past beats only, plus the next R peak. {rt['cnn_seeds']} CNN seeds, ensemble weight {rt['ensemble_weight_lr']} "
               f"(chosen on the DS1 hold-out). Real-time AF detector on DS2: Se {rt['af_detection_DS2']['sensitivity']:.1%}, "
               f"+P {rt['af_detection_DS2']['ppv']:.1%}.", "",
               "| Test set | Version | Accuracy | Macro-F1 | N Se / +P | S Se / +P | V Se / +P | F Se / +P |",
               "|---|---|---|---|---|---|---|---|"]
        for name, rows in rt["test"].items():
            for k, v in rows.items():
                md.append(f"| {name} | {k} | {v['accuracy']:.1%} | {v['macro_f1']:.3f} | " +
                          " | ".join(f"{v[c + '_se']:.1%} / {v[c + '_ppv']:.1%}" for c in "NSVF") + " |")
    (RESULTS / "extensions.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
