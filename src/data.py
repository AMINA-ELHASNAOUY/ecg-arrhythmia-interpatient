"""Beat extraction for MIT-BIH (inter-patient DS1/DS2 split) and the MIT-BIH Supraventricular
Arrhythmia Database (external test set).

Every beat is described *relative to its own patient*, because what generalises to unseen
people is "how different is this beat from this person's usual beat / rhythm", not the raw shape:

  x        (2, 256)  channel 0: the beat (0.5-40 Hz band-pass, z-normalised)
                     channel 1: the beat minus the patient's template (median beat of the record)
  f        (11,)     7 local RR features (log ratios, no global normalisation) +
                     4 template features (correlation / RMS distance to the template, whole beat and QRS)

Two ways to locate beats:
  peaks="annotated"  R peaks from the expert annotations (standard in the literature)
  peaks="detected"   R peaks found automatically (wfdb XQRS); labels are taken from the nearest
                     annotation within 150 ms, so the pipeline runs as it would on a new recording

Labels follow the AAMI EC57 grouping: N, S, V, F (Q beats dropped, paced records excluded).
The template is computed from the unlabeled signal only.
"""
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from wfdb import processing

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MITDB_DIR = DATA_DIR / "mitdb"
SVDB_DIR = DATA_DIR / "svdb"

FS = 360                         # everything is resampled to 360 Hz
BEFORE, AFTER = 90, 166          # 256 samples per beat, R peak at index 90
QRS = slice(70, 115)             # ~ +-60 ms around the R peak
MATCH_TOL = int(0.15 * FS)       # 150 ms, AAMI beat-matching window
CLASSES = ["N", "S", "V", "F"]
RR_NAMES = ["pre/local10", "post/local10", "pre/post", "pre/prev_pre",
            "local10/local50", "pre/local50", "post/local50"]
TMPL_NAMES = ["corr_beat", "corr_qrs", "rms_beat", "rms_qrs"]

AAMI = {
    **{s: 0 for s in "NLRej"},   # N: normal, bundle branch blocks, escape beats
    **{s: 1 for s in "AaJS"},    # S: supraventricular ectopic
    **{s: 2 for s in "VE"},      # V: ventricular ectopic
    "F": 3,                      # F: fusion
}
BEAT_SYMBOLS = set("NLRejAaJSVEF/fQ")

# de Chazal et al. (2004) inter-patient split. Paced records 102, 104, 107, 217 are excluded.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
       205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]
# DS1 patients held out for early stopping / ensemble weighting (never trained on, never DS2).
VAL = [108, 118, 207, 223]


# ---------------------------------------------------------------- signal
def _bandpass(x):
    b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def read_record(db_dir, rec):
    """Band-passed lead at 360 Hz + beat annotations (positions at 360 Hz, symbols)."""
    path = str(Path(db_dir) / str(rec))
    sig = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    lead = sig.sig_name.index("MLII") if "MLII" in sig.sig_name else 0
    raw = np.nan_to_num(sig.p_signal[:, lead])
    samples = np.asarray(ann.sample, dtype=np.float64)
    if sig.fs != FS:
        ratio = Fraction(FS, int(round(sig.fs))).limit_denominator(1000)
        raw = resample_poly(raw, ratio.numerator, ratio.denominator)
        samples = samples * FS / sig.fs
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    return _bandpass(raw), np.round(samples[keep]).astype(int), [ann.symbol[i] for i in keep]


def detect_peaks(x):
    """Automatic R-peak detection: XQRS, then snap each detection to the local extremum."""
    det = processing.xqrs_detect(sig=x, fs=FS, verbose=False)
    det = processing.correct_peaks(x, det, search_radius=int(0.05 * FS),
                                   smooth_window_size=int(0.15 * FS), peak_dir="compare")
    det = np.unique(det[(det >= 0) & (det < len(x))])
    return det


def match_peaks(det, ref):
    """Greedy one-to-one matching of detections to reference beats within 150 ms.
    Returns, for each detection, the index of its reference beat or -1."""
    owner = np.full(len(det), -1)
    if len(det) == 0:
        return owner
    used = np.zeros(len(det), bool)
    for j, r in enumerate(ref):
        k = np.searchsorted(det, r)
        best, best_d = -1, MATCH_TOL + 1
        for c in (k - 1, k):
            if 0 <= c < len(det) and not used[c] and abs(det[c] - r) < best_d:
                best, best_d = c, abs(det[c] - r)
        if best >= 0:
            used[best] = True
            owner[best] = j
    return owner


# ---------------------------------------------------------------- features
def _corr_rows(a, t):
    a = a - a.mean(1, keepdims=True)
    t = t - t.mean()
    return (a @ t) / (np.linalg.norm(a, axis=1) * np.linalg.norm(t) + 1e-8)


def beats_from_peaks(x, peaks, labels):
    """peaks: all beat positions of the record (defines the rhythm);
    labels: AAMI label per peak, or -1 for beats that are not classified (Q, unmatched)."""
    rr = np.diff(peaks) / FS
    windows, rrf, ys, used = [], [], [], []
    for k in range(1, len(peaks) - 1):          # need a previous and a next beat
        p = peaks[k]
        if p - BEFORE < 0 or p + AFTER > len(x):
            continue
        w = x[p - BEFORE:p + AFTER]
        windows.append((w - w.mean()) / (w.std() + 1e-8))
        pre, post = max(rr[k - 1], 1e-3), max(rr[k], 1e-3)
        prev = max(rr[k - 2], 1e-3) if k >= 2 else pre
        l10 = rr[max(0, k - 10):k].mean()
        l50 = rr[max(0, k - 25):k + 25].mean()
        rrf.append([pre / l10, post / l10, pre / post, pre / prev, l10 / l50, pre / l50, post / l50])
        ys.append(labels[k])
        used.append(k)

    windows = np.asarray(windows, np.float32)
    template = np.median(windows, axis=0)            # all beats of the record, labels not used
    diff = windows - template
    tmpl = np.stack([
        _corr_rows(windows, template),
        _corr_rows(windows[:, QRS], template[QRS]),
        np.sqrt((diff ** 2).mean(1)),
        np.sqrt((diff[:, QRS] ** 2).mean(1)),
    ], 1)
    rrf = np.log(np.clip(np.asarray(rrf), 1e-3, None))
    ys = np.asarray(ys)
    keep = ys >= 0
    return (np.stack([windows, diff], 1)[keep].astype(np.float16),
            np.hstack([rrf, tmpl])[keep].astype(np.float32),
            ys[keep].astype(np.int64))


def load_record(db_dir, rec, peaks="annotated"):
    x, ref, syms = read_record(db_dir, rec)
    ref_labels = np.array([AAMI.get(s, -1) for s in syms])
    if peaks == "annotated":
        out = beats_from_peaks(x, ref, ref_labels)
        stats = None
    else:
        det = detect_peaks(x)
        owner = match_peaks(det, ref)
        labels = np.where(owner >= 0, ref_labels[np.maximum(owner, 0)], -1)
        out = beats_from_peaks(x, det, labels)
        tp = int((owner >= 0).sum())
        stats = {"record": int(rec) if str(rec).isdigit() else rec, "reference_beats": int(len(ref)),
                 "detected": int(len(det)), "tp": tp, "fn": int(len(ref) - tp), "fp": int(len(det) - tp)}
    return out, stats


# ---------------------------------------------------------------- datasets
def _stack(parts, recs, prefix):
    return {
        f"{prefix}_x": np.concatenate([p[0] for p in parts]),
        f"{prefix}_f": np.concatenate([p[1] for p in parts]),
        f"{prefix}_y": np.concatenate([p[2] for p in parts]),
        f"{prefix}_rec": np.concatenate([np.full(len(p[2]), int(r)) for p, r in zip(parts, recs)]),
    }


def build_mitdb(peaks="annotated"):
    out, stats = {}, []
    for name, recs in [("train", [r for r in DS1 if r not in VAL]), ("val", VAL), ("test", DS2)]:
        res = [load_record(MITDB_DIR, r, peaks) for r in recs]
        out.update(_stack([r[0] for r in res], recs, name))
        stats += [r[1] for r in res if r[1]]
    np.savez_compressed(DATA_DIR / f"mitdb_{peaks}.npz", **out)
    return out, stats


def svdb_records():
    return [r.strip() for r in (SVDB_DIR / "RECORDS").read_text().split() if r.strip()]


def build_svdb(peaks="annotated"):
    recs = svdb_records()
    res = [load_record(SVDB_DIR, r, peaks) for r in recs]
    out = _stack([r[0] for r in res], recs, "ext")
    np.savez_compressed(DATA_DIR / f"svdb_{peaks}.npz", **out)
    return out, [r[1] for r in res if r[1]]


def _load(path, builder, peaks):
    if not path.exists():
        print(f"Extracting beats -> {path.name} (first run only)...")
        builder(peaks)
    d = dict(np.load(path))
    for k in list(d):
        if k.endswith("_x"):
            d[k] = d[k].astype(np.float32)
    return d


def load_splits(peaks="annotated"):
    return _load(DATA_DIR / f"mitdb_{peaks}.npz", build_mitdb, peaks)


def load_external(peaks="annotated"):
    return _load(DATA_DIR / f"svdb_{peaks}.npz", build_svdb, peaks)


if __name__ == "__main__":
    import sys
    peaks = sys.argv[1] if len(sys.argv) > 1 else "annotated"
    d, _ = build_mitdb(peaks)
    for s in ["train", "val", "test"]:
        counts = np.bincount(d[f"{s}_y"], minlength=4)
        print(f"{s:5s} beats={len(d[f'{s}_y']):6d}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
