"""MIT-BIH beat extraction with an inter-patient (de Chazal DS1/DS2) split.

Every beat is described *relative to its own patient*, because what generalises to unseen
people is "how different is this beat from this person's usual beat / rhythm", not the raw shape:

  x        (2, 256)  channel 0: the beat (lead MLII, 0.5-40 Hz band-pass, z-normalised)
                     channel 1: the beat minus the patient's template (median beat of the record)
  feats    (11,)     7 local RR features (log ratios, no global normalisation) +
                     4 template features (correlation / RMS distance to the template, whole beat and QRS)

Labels follow the AAMI EC57 grouping: N, S, V, F (Q beats dropped, paced records excluded).
The template is computed from the unlabeled signal only (no annotations of beat classes are used).
"""
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MITDB_DIR = DATA_DIR / "mitdb"
CACHE = DATA_DIR / "beats_v2.npz"

FS = 360
BEFORE, AFTER = 90, 166          # 256 samples per beat, R peak at index 90
QRS = slice(70, 115)             # ~ +-60 ms around the R peak
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
# DS1 patients held out for early stopping (never trained on, never DS2).
VAL = [108, 118, 207, 223]


def _bandpass(x):
    b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def _corr_rows(a, t):
    a = a - a.mean(1, keepdims=True)
    t = t - t.mean()
    return (a @ t) / (np.linalg.norm(a, axis=1) * np.linalg.norm(t) + 1e-8)


def load_record(rec):
    path = str(MITDB_DIR / str(rec))
    sig = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    lead = sig.sig_name.index("MLII") if "MLII" in sig.sig_name else 0
    x = _bandpass(sig.p_signal[:, lead])

    idx = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    peaks = np.asarray(ann.sample)[idx]
    syms = [ann.symbol[i] for i in idx]
    rr = np.diff(peaks) / FS

    beats, rrf, labels = [], [], []
    for k in range(1, len(peaks) - 1):          # need a previous and a next beat
        if syms[k] not in AAMI:
            continue
        p = peaks[k]
        if p - BEFORE < 0 or p + AFTER > len(x):
            continue
        w = x[p - BEFORE:p + AFTER]
        beats.append((w - w.mean()) / (w.std() + 1e-8))
        pre, post = rr[k - 1], rr[k]
        prev = rr[k - 2] if k >= 2 else pre
        l10 = rr[max(0, k - 10):k].mean()
        l50 = rr[max(0, k - 25):k + 25].mean()
        rrf.append([pre / l10, post / l10, pre / post, pre / prev, l10 / l50, pre / l50, post / l50])
        labels.append(AAMI[syms[k]])

    beats = np.asarray(beats, np.float32)
    rrf = np.log(np.clip(np.asarray(rrf, np.float32), 1e-3, None))
    template = np.median(beats, axis=0)
    diff = beats - template
    tmpl = np.stack([
        _corr_rows(beats, template),
        _corr_rows(beats[:, QRS], template[QRS]),
        np.sqrt((diff ** 2).mean(1)),
        np.sqrt((diff[:, QRS] ** 2).mean(1)),
    ], 1).astype(np.float32)

    x2 = np.stack([beats, diff.astype(np.float32)], 1)
    return x2, np.hstack([rrf, tmpl]), np.asarray(labels, np.int64)


def build_cache():
    out = {}
    for name, recs in [("train", [r for r in DS1 if r not in VAL]), ("val", VAL), ("test", DS2)]:
        parts = [load_record(r) for r in recs]
        out[f"{name}_x"] = np.concatenate([p[0] for p in parts])
        out[f"{name}_f"] = np.concatenate([p[1] for p in parts])
        out[f"{name}_y"] = np.concatenate([p[2] for p in parts])
        out[f"{name}_rec"] = np.concatenate([np.full(len(p[2]), r) for p, r in zip(parts, recs)])
    np.savez_compressed(CACHE, **out)
    return out


def load_splits():
    if not CACHE.exists():
        print("Extracting beats from MIT-BIH (first run only)...")
        return build_cache()
    return dict(np.load(CACHE))


if __name__ == "__main__":
    d = build_cache()
    for s in ["train", "val", "test"]:
        counts = np.bincount(d[f"{s}_y"], minlength=4)
        print(f"{s:5s} beats={len(d[f'{s}_y']):6d}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
