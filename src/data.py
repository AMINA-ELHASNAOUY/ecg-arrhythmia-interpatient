"""Beat extraction for MIT-BIH (inter-patient DS1/DS2 split) and two external databases:
the MIT-BIH Supraventricular Arrhythmia Database (SVDB) and the St Petersburg INCART 12-lead database.

Every beat is described *relative to its own patient*, because what generalises to unseen
people is "how different is this beat from this person's usual beat / rhythm", not the raw shape.
Two leads are used (lead A = MLII on MIT-BIH, ECG1 on SVDB, II on INCART; lead B = the second channel / V1).

  x      (4, 256)  lead A beat, lead A beat - patient template, lead B beat, lead B beat - template
  f      (15,)     7 local RR features (log ratios) + 4 template features for lead A + 4 for lead B
  r      (8,)      rhythm context for the AF detector: 6 RR-irregularity features over +-15 beats
                   + 2 P-wave consistency features (atrial fibrillation has no regular P wave)
  af     bool      beat lies in an atrial fibrillation / flutter episode (MIT-BIH rhythm annotations)

Beats can be located from the expert annotations (peaks="annotated") or automatically with XQRS
(peaks="detected"; labels from the nearest annotation within 150 ms).
Labels follow AAMI EC57: N, S, V, F (Q beats dropped, paced records excluded).
Templates and rhythm features use the unlabeled signal only.
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
INCART_DIR = DATA_DIR / "incart"

FS = 360                         # everything is resampled to 360 Hz
BEFORE, AFTER = 90, 166          # 256 samples per beat, R peak at index 90
QRS = slice(70, 115)             # ~ +-60 ms around the R peak
PWAVE = slice(18, 50)            # ~ -200 .. -110 ms before the R peak
MATCH_TOL = int(0.15 * FS)       # 150 ms, AAMI beat-matching window
RHYTHM_W = 15                    # beats on each side for rhythm context
CLASSES = ["N", "S", "V", "F"]
RR_NAMES = ["pre/local10", "post/local10", "pre/post", "pre/prev_pre",
            "local10/local50", "pre/local50", "post/local50"]
TMPL_NAMES = ["corr_beat", "corr_qrs", "rms_beat", "rms_qrs"]
FEATURE_NAMES = RR_NAMES + [f"A_{n}" for n in TMPL_NAMES] + [f"B_{n}" for n in TMPL_NAMES]
RHYTHM_NAMES = ["rr_cv", "rr_nrmssd", "frac_big_jumps", "med_jump", "rr_autocorr1", "rr_autocorr2",
                "p_consecutive_corr", "p_local_template_corr"]
N_LEAD_A_FEATURES = 11           # RR + lead A template: the single-lead feature set

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

LEADS = {"mitdb": ("MLII", None), "svdb": ("ECG1", "ECG2"), "incart": ("II", "V1")}


# ---------------------------------------------------------------- signal
def _bandpass(x):
    b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def read_record(db_dir, rec, db="mitdb"):
    """Two band-passed leads at 360 Hz, beat positions (360 Hz), symbols, AF flag per beat."""
    path = str(Path(db_dir) / str(rec))
    sig = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    name_a, name_b = LEADS[db]
    ia = sig.sig_name.index(name_a) if name_a in sig.sig_name else 0
    ib = sig.sig_name.index(name_b) if name_b in sig.sig_name else (1 if ia == 0 else 0)
    leads = []
    for i in (ia, ib):
        raw = np.nan_to_num(sig.p_signal[:, i])
        if sig.fs != FS:
            ratio = Fraction(FS, int(round(sig.fs))).limit_denominator(1000)
            raw = resample_poly(raw, ratio.numerator, ratio.denominator)
        leads.append(_bandpass(raw))
    samples = np.asarray(ann.sample, dtype=np.float64) * FS / sig.fs

    rhythm, keep, af = "(N", [], []
    for i, (s, note) in enumerate(zip(ann.symbol, ann.aux_note)):
        if note and note.startswith("("):
            rhythm = note.strip("\x00").strip()
        if s in BEAT_SYMBOLS:
            keep.append(i)
            af.append(rhythm in ("(AFIB", "(AFL"))
    return (leads[0], leads[1], np.round(samples[keep]).astype(int),
            [ann.symbol[i] for i in keep], np.asarray(af, bool))


def detect_peaks(x):
    """Automatic R-peak detection: XQRS, then snap each detection to the local extremum."""
    det = processing.xqrs_detect(sig=x, fs=FS, verbose=False)
    det = processing.correct_peaks(x, det, search_radius=int(0.05 * FS),
                                   smooth_window_size=int(0.15 * FS), peak_dir="compare")
    return np.unique(det[(det >= 0) & (det < len(x))])


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


def _znorm_rows(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)


def _windows(x, peaks):
    w = np.stack([x[p - BEFORE:p + AFTER] for p in peaks]).astype(np.float32)
    return _znorm_rows(w)


def _template_features(w):
    template = np.median(w, axis=0)                  # all beats of the record, labels not used
    diff = w - template
    feats = np.stack([
        _corr_rows(w, template),
        _corr_rows(w[:, QRS], template[QRS]),
        np.sqrt((diff ** 2).mean(1)),
        np.sqrt((diff[:, QRS] ** 2).mean(1)),
    ], 1)
    return diff, feats


def _autocorr(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / ((x * x).sum() + 1e-9)) if len(x) > lag else 0.0


def _smooth(v, k=RHYTHM_W):
    pad = np.pad(v, k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")


def _pwave_features(w):
    """P-wave consistency: similarity of each beat's P-wave segment to the previous beat's and to the
    local median P segment, averaged over +-15 beats. Regular sinus/atrial rhythm -> high; AF -> low."""
    p = _znorm_rows(w[:, PWAVE])
    consec = np.r_[0.0, (p[1:] * p[:-1]).mean(1)]
    consec[0] = consec[1] if len(consec) > 1 else 0.0
    local = np.empty(len(p))
    for i in range(len(p)):
        med = np.median(p[max(0, i - RHYTHM_W):i + RHYTHM_W + 1], 0)
        med = (med - med.mean()) / (med.std() + 1e-8)
        local[i] = (p[i] * med).mean()
    return np.stack([_smooth(consec), _smooth(local)], 1)


def beats_from_peaks(xa, xb, peaks, labels, af):
    """peaks: all beat positions of the record (defines the rhythm);
    labels: AAMI label per peak, or -1 for beats that are not classified (Q, unmatched)."""
    rr = np.diff(peaks) / FS
    ks = [k for k in range(1, len(peaks) - 1)
          if peaks[k] - BEFORE >= 0 and peaks[k] + AFTER <= min(len(xa), len(xb))]
    pk = peaks[ks]
    wa, wb = _windows(xa, pk), _windows(xb, pk)
    da, ta = _template_features(wa)
    db, tb = _template_features(wb)

    rrf, rhy = [], []
    for k in ks:
        pre, post = max(rr[k - 1], 1e-3), max(rr[k], 1e-3)
        prev = max(rr[k - 2], 1e-3) if k >= 2 else pre
        l10 = rr[max(0, k - 10):k].mean()
        l50 = rr[max(0, k - 25):k + 25].mean()
        rrf.append([pre / l10, post / l10, pre / post, pre / prev, l10 / l50, pre / l50, post / l50])
        seg = rr[max(0, k - RHYTHM_W):k + RHYTHM_W]
        m = seg.mean()
        jumps = np.diff(seg)
        rhy.append([seg.std() / m, np.sqrt((jumps ** 2).mean()) / m, (np.abs(jumps) / m > 0.1).mean(),
                    np.median(np.abs(jumps)) / np.median(seg), _autocorr(seg, 1), _autocorr(seg, 2)])
    rrf = np.log(np.clip(np.asarray(rrf), 1e-3, None))
    rhythm = np.hstack([np.asarray(rhy), _pwave_features(wa)])

    y = np.asarray([labels[k] for k in ks])
    keep = y >= 0
    x = np.stack([wa, da, wb, db], 1)
    f = np.hstack([rrf, ta, tb])
    return {"x": x[keep].astype(np.float16), "f": f[keep].astype(np.float32),
            "r": rhythm[keep].astype(np.float32), "y": y[keep].astype(np.int64),
            "af": np.asarray([af[k] for k in ks])[keep]}


def load_record(db_dir, rec, peaks="annotated", db="mitdb"):
    xa, xb, ref, syms, ref_af = read_record(db_dir, rec, db)
    ref_labels = np.array([AAMI.get(s, -1) for s in syms])
    if peaks == "annotated":
        return beats_from_peaks(xa, xb, ref, ref_labels, ref_af), None
    det = detect_peaks(xa)
    owner = match_peaks(det, ref)
    labels = np.where(owner >= 0, ref_labels[np.maximum(owner, 0)], -1)
    af = np.where(owner >= 0, ref_af[np.maximum(owner, 0)], False)
    out = beats_from_peaks(xa, xb, det, labels, af)
    tp = int((owner >= 0).sum())
    stats = {"record": int(rec) if str(rec).isdigit() else rec, "reference_beats": int(len(ref)),
             "detected": int(len(det)), "tp": tp, "fn": int(len(ref) - tp), "fp": int(len(det) - tp)}
    return out, stats


# ---------------------------------------------------------------- datasets
def _stack(parts, recs, prefix):
    out = {f"{prefix}_{k}": np.concatenate([p[k] for p in parts]) for k in ["x", "f", "r", "y", "af"]}
    out[f"{prefix}_rec"] = np.concatenate([np.full(len(p["y"]), i) for i, p in enumerate(parts)])
    out[f"{prefix}_recname"] = np.array([str(r) for r in recs])
    return out


def build_mitdb(peaks="annotated"):
    out, stats = {}, []
    for name, recs in [("train", [r for r in DS1 if r not in VAL]), ("val", VAL), ("test", DS2)]:
        res = [load_record(MITDB_DIR, r, peaks, "mitdb") for r in recs]
        part = _stack([r[0] for r in res], recs, name)
        part[f"{name}_rec"] = np.asarray(recs)[part[f"{name}_rec"]]     # real record numbers
        out.update(part)
        stats += [r[1] for r in res if r[1]]
    np.savez_compressed(DATA_DIR / f"mitdb_{peaks}.npz", **out)
    return out, stats


def records(db_dir):
    return [r.strip() for r in (Path(db_dir) / "RECORDS").read_text().split() if r.strip()]


def build_external(db, peaks="annotated"):
    db_dir = {"svdb": SVDB_DIR, "incart": INCART_DIR}[db]
    recs = records(db_dir)
    res = [load_record(db_dir, r, peaks, db) for r in recs]
    out = _stack([r[0] for r in res], recs, "ext")
    np.savez_compressed(DATA_DIR / f"{db}_{peaks}.npz", **out)
    return out, [r[1] for r in res if r[1]]


def _load(path, builder):
    if not path.exists():
        print(f"Extracting beats -> {path.name} (first run only)...")
        builder()
    d = dict(np.load(path))
    for k in list(d):
        if k.endswith("_x"):
            d[k] = d[k].astype(np.float32)
    return d


def load_splits(peaks="annotated"):
    return _load(DATA_DIR / f"mitdb_{peaks}.npz", lambda: build_mitdb(peaks))


def load_external(db="svdb", peaks="annotated"):
    return _load(DATA_DIR / f"{db}_{peaks}.npz", lambda: build_external(db, peaks))


if __name__ == "__main__":
    import sys
    peaks = sys.argv[1] if len(sys.argv) > 1 else "annotated"
    d, _ = build_mitdb(peaks)
    for s in ["train", "val", "test"]:
        counts = np.bincount(d[f"{s}_y"], minlength=4)
        print(f"{s:5s} beats={len(d[f'{s}_y']):6d}  AF beats={int(d[f'{s}_af'].sum()):5d}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
