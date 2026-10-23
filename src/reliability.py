"""Two recording-level ideas that use context a single beat cannot see.

1. Fusion gate. A fusion (F) beat is a normal beat and a ventricular beat firing at the same moment,
   so it cannot happen in a recording without ventricular ectopy. Beat by beat, false F alarms look
   just like real fusion (on time, slightly unusual shape), so the gate works at recording level:
   if the model itself predicts V for fewer than TAU of a recording's beats, its F predictions are
   replaced by the best of N / S / V. TAU = 1% is fixed in advance, not tuned.

2. Reliability flag. Before trusting a recording's result, check how consistent its beats are:
   the median correlation between each beat and the patient's own median beat (lead A). It needs no
   labels and no model. The threshold is chosen on DS1 patients only (out-of-fold predictions).
"""
import numpy as np

TAU = 0.01                  # fusion gate: minimum predicted V fraction for F to be allowed
CONSISTENCY_COL = 7         # feature column: lead A correlation between beat and patient template
ACC_TARGET = 0.90           # a recording "fails" when beat accuracy is below this


def fusion_gate(prob, pred, rec, tau=TAU):
    out = pred.copy()
    for r in np.unique(rec):
        m = rec == r
        if (pred[m] == 2).mean() < tau:
            mf = m & (out == 3)
            out[mf] = prob[mf][:, :3].argmax(1)
    return out


def per_recording(y, rec, pred, consistency, af_pred=None):
    """consistency: per-beat lead A correlation with the patient template (feature column CONSISTENCY_COL)."""
    rows = []
    for r in np.unique(rec):
        m = rec == r
        row = {"record": str(r), "beats": int(m.sum()),
               "accuracy": float((pred[m] == y[m]).mean()),
               "consistency": float(np.median(consistency[m])),
               "pred_F_fraction": float((pred[m] == 3).mean())}
        for c_i, c in [(1, "S"), (2, "V")]:
            row[f"true_{c}_burden"] = float((y[m] == c_i).mean())
            row[f"pred_{c}_burden"] = float((pred[m] == c_i).mean())
        if af_pred is not None:
            row["af_burden"] = float(af_pred[m].mean())
        rows.append(row)
    return rows


def choose_threshold(rows, grid=np.round(np.arange(0.80, 0.9951, 0.005), 3)):
    """Threshold on consistency maximising Youden's J for 'accuracy < ACC_TARGET' (DS1 recordings only)."""
    c = np.array([r["consistency"] for r in rows])
    bad = np.array([r["accuracy"] for r in rows]) < ACC_TARGET
    scores = {float(t): float((c < t)[bad].mean() - (c < t)[~bad].mean()) for t in grid}
    best = max(scores, key=scores.get)
    return best, scores


def burden_category(b):
    """Ectopic burden category: <1%, 1-10%, >=10% of beats (the 10% line is commonly used for PVC burden)."""
    return int(np.digitize(b, [0.01, 0.10]))
