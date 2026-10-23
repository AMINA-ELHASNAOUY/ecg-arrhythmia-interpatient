"""Recording-level ideas that use context a single beat cannot see.

1. Fusion gate. A fusion (F) beat is a normal beat and a ventricular beat firing at the same moment,
   so it cannot happen in a recording without ventricular ectopy. Beat by beat, false F alarms look
   just like real fusion (on time, slightly unusual shape), so the gate works at recording level:
   if the model itself predicts V for fewer than TAU of a recording's beats, its F predictions are
   replaced by the best of N / S / V. TAU = 1% is fixed in advance, not tuned.

2. Reliability warning, with causes. Computed without labels and without the model:
   - unstable beats (the warning): share of on-time beats that match none of the patient's own
     conduction patterns (correlation < 0.9 with the nearest template). Noise, electrode problems or
     shapes the model has never seen. Threshold chosen on DS1 patients only.
   - frequent early beats (information): share of beats arriving > 15% early (ectopy or irregular rhythm).
   - several conduction patterns (information): the patient switches between beat shapes (src/data.py).
   The first version of the flag, the median correlation with a single median beat ("consistency"),
   is kept for comparison.
"""
import numpy as np

TAU = 0.01                  # fusion gate: minimum predicted V fraction for F to be allowed
FIT_COL = 7                 # feature column: lead A correlation between a beat and its nearest template
RR_PRE_COL = 0              # feature column: log(RR before the beat / mean of the 10 previous RRs)
ONTIME = 0.10               # |log RR ratio| below this = on time
UNMATCHED = 0.90            # an on-time beat with correlation < 0.9 to every template is "unstable"
EARLY = np.log(0.85)        # a beat arriving > 15% early
ACC_TARGET = 0.90           # a recording "fails" when beat accuracy is below this


def fusion_gate(prob, pred, rec, tau=TAU):
    out = pred.copy()
    for r in np.unique(rec):
        m = rec == r
        if (pred[m] == 2).mean() < tau:
            mf = m & (out == 3)
            out[mf] = prob[mf][:, :3].argmax(1)
    return out


def recording_signals(fit, rr_pre, consistency, n_patterns):
    """Label-free reliability signals of ONE recording (per-beat arrays of that recording)."""
    ontime = np.abs(rr_pre) < ONTIME
    pool = fit[ontime] if ontime.sum() >= 10 else fit
    return {"unstable": float((pool < UNMATCHED).mean()),
            "early": float((rr_pre < EARLY).mean()),
            "consistency": float(np.median(consistency)),
            "patterns": int(n_patterns[0])}


def per_recording(y, rec, pred, fit, rr_pre, consistency, n_patterns, af_pred=None):
    rows = []
    for r in np.unique(rec):
        m = rec == r
        row = {"record": str(r), "beats": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
               "pred_F_fraction": float((pred[m] == 3).mean())}
        row.update(recording_signals(fit[m], rr_pre[m], consistency[m], n_patterns[m]))
        for c_i, c in [(1, "S"), (2, "V")]:
            row[f"true_{c}_burden"] = float((y[m] == c_i).mean())
            row[f"pred_{c}_burden"] = float((pred[m] == c_i).mean())
        if af_pred is not None:
            row["af_burden"] = float(af_pred[m].mean())
        rows.append(row)
    return rows


def choose_threshold(rows, key, higher_is_worse, grid):
    """Threshold on one signal maximising Youden's J for 'accuracy < ACC_TARGET' (DS1 recordings only)."""
    v = np.array([r[key] for r in rows])
    bad = np.array([r["accuracy"] for r in rows]) < ACC_TARGET
    flag = (lambda t: v > t) if higher_is_worse else (lambda t: v < t)
    scores = {float(t): float(flag(t)[bad].mean() - flag(t)[~bad].mean()) for t in grid}
    best = max(scores, key=scores.get)
    return best, scores


def causes(row, thresholds):
    out = []
    if row["unstable"] > thresholds["unstable"]:
        out.append("unstable beats")
    if row["early"] > thresholds["early"]:
        out.append("frequent early beats")
    if row["patterns"] > 1:
        out.append(f"{row['patterns']} beat patterns")
    return out


def burden_category(b):
    """Ectopic burden category: <1%, 1-10%, >=10% of beats (the 10% line is commonly used for PVC burden)."""
    return int(np.digitize(b, [0.01, 0.10]))
