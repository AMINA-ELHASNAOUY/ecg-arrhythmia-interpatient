"""Beat extraction for MIT-BIH (inter-patient DS1/DS2 split) and two external databases:
the MIT-BIH Supraventricular Arrhythmia Database (SVDB) and the St Petersburg INCART 12-lead database.

Every beat is described *relative to its own patient*, because what generalises to unseen
people is "how different is this beat from this person's usual beat / rhythm", not the raw shape.
Two leads are used (lead A = MLII on MIT-BIH, ECG1 on SVDB, II on INCART; lead B = the second channel / V1).

  x      (4, 256)  lead A beat, lead A beat - nearest patient template, lead B beat, lead B beat - template
  f      (15,)     7 local RR features (log ratios) + 4 template features for lead A + 4 for lead B
  r      (8,)      rhythm context for the AF detector: 6 RR-irregularity features over +-15 beats
                   + 2 P-wave consistency features (atrial fibrillation has no regular P wave)
  af     bool      beat lies in an atrial fibrillation / flutter episode (MIT-BIH rhythm annotations)
  tmpl   int       which of the patient's templates (conduction patterns, lead A) the beat is closest to
  n_tmpl int       number of conduction patterns found in the beat's recording (lead A)
  consist float    correlation of the beat with the patient's single median beat (lead A), for the reliability flag

Beats can be located from the expert annotations (peaks="annotated") or automatically with XQRS
(peaks="detected"; labels from the nearest annotation within 150 ms).
Labels follow AAMI EC57: N, S, V, F (Q beats dropped, paced records excluded).
Templates and rhythm features use the unlabeled signal only.

Templates: a patient can have more than one normal conduction pattern (e.g. intermittent bundle branch block).
With a single median beat, the minority pattern looks "on time but different", which is exactly what a fusion
beat looks like. So the on-time beats of each recording are clustered into up to MAX_TEMPLATES patterns, and each
beat is compared with its nearest pattern. Premature beats (most S and V beats) never create a pattern, so
frequent ectopy cannot turn into a "normal" template.
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
NSTDB_DIR = DATA_DIR / "nstdb"          # MIT-BIH Noise Stress Test Database (extensions.py)
AFDB_DIR = DATA_DIR / "afdb"            # MIT-BIH Atrial Fibrillation Database (extensions.py)
NSTDB_RECORDS = [f"118e{snr}" for snr in ("24", "18", "12", "06", "00", "_6")]   # 118: DS1 hold-out, never trained on
AFDB_RECORDS = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121", "05261", "06426",
                "06453", "06995", "07162", "07859", "07879", "07910", "08215", "08219", "08378", "08405", "08434", "08455"]

FS = 360                         # everything is resampled to 360 Hz
BEFORE, AFTER = 90, 166          # 256 samples per beat, R peak at index 90
QRS = slice(70, 115)             # ~ +-60 ms around the R peak
PWAVE = slice(18, 50)            # ~ -200 .. -110 ms before the R peak
MATCH_TOL = int(0.15 * FS)       # 150 ms, AAMI beat-matching window
RHYTHM_W = 15                    # beats on each side for rhythm context
MAX_TEMPLATES = 3                # conduction patterns per patient (1 = a single median beat, the first version)
MIN_PATTERN = 0.10               # a conduction pattern must cover >= 10% of the recording's beats
ONTIME = 0.10                    # |log(RR / local RR)| below this = the beat is not premature or late
MIN_RUN_FRACTION = 0.5           # >= 50% of a pattern's beats must sit in runs of >= 3 consecutive beats
                                 # (MIT-BIH: conduction changes 98-100%, V beats at most 38%)
FEATURE_VERSION = "v2"           # v2: multi-template features; cached beat files are rebuilt when this changes
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

LEADS = {"mitdb": ("MLII", None), "svdb": ("ECG1", "ECG2"), "incart": ("II", "V1"), "nstdb": ("MLII", None)}


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


def _corr_matrix(a, t):
    """Correlation of every row of a with every row of t -> (len(a), len(t))."""
    a = _znorm_rows(a)
    t = _znorm_rows(t)
    return a @ t.T / a.shape[1]


def _run_fraction(mask, min_len=3):
    """Fraction of the True values that sit in runs of >= min_len consecutive True values."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    runs = edges[1::2] - edges[::2]
    return float(runs[runs >= min_len].sum() / max(runs.sum(), 1))


def find_templates(w, ontime, max_templates=MAX_TEMPLATES):
    """Conduction patterns of one recording (labels not used).

    Starts from the median beat. On-time beats that match no template (whole-beat corr < 0.9) seed a new
    pattern; patterns are refined like k-means (median of their on-time members) and kept only if they cover
    >= MIN_PATTERN of all beats, most of their members are on time, they come in runs of consecutive beats
    (a conduction change lasts for a while; ectopic beats are scattered), and they differ clearly from the
    other patterns (corr < 0.95). Whole beats are compared, because conduction changes such as bundle branch block
    show most in the ST-T segment."""
    n = len(w)
    templates = [np.median(w, 0)]
    if max_templates == 1 or ontime.sum() < 2 * MIN_PATTERN * n:
        return np.stack(templates)
    for _ in range(max_templates - 1):
        c = _corr_matrix(w[ontime], np.stack(templates)).max(1)
        seed = w[ontime][c < 0.9]
        if len(seed) < MIN_PATTERN * n:
            break
        cand = templates + [np.median(seed, 0)]
        for _ in range(5):
            a = _corr_matrix(w, np.stack(cand)).argmax(1)
            cand = [np.median(w[(a == k) & ontime], 0) if ((a == k) & ontime).sum() > 0 else cand[k]
                    for k in range(len(cand))]
        a = _corr_matrix(w, np.stack(cand)).argmax(1)
        keep = [k for k in range(len(cand))
                if ((a == k) & ontime).sum() >= MIN_PATTERN * n and ontime[a == k].mean() >= 0.6
                and _run_fraction(a == k) >= MIN_RUN_FRACTION]
        merged = []                                  # drop near-duplicate patterns
        for t in [cand[k] for k in keep]:
            if all(_corr_matrix(t[None], u[None])[0, 0] < 0.95 for u in merged):
                merged.append(t)
        if len(merged) <= len(templates):
            break
        templates = merged
    return np.stack(templates)


def _template_features(w, ontime, max_templates=MAX_TEMPLATES):
    templates = find_templates(w, ontime, max_templates)
    nearest = _corr_matrix(w, templates).argmax(1)
    template = templates[nearest]                    # each beat's own conduction pattern
    diff = w - template
    feats = np.stack([
        ((_znorm_rows(w) * _znorm_rows(template)).mean(1)),
        ((_znorm_rows(w[:, QRS]) * _znorm_rows(template[:, QRS])).mean(1)),
        np.sqrt((diff ** 2).mean(1)),
        np.sqrt((diff[:, QRS] ** 2).mean(1)),
    ], 1)
    return diff, feats, nearest, len(templates)


def _autocorr(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / ((x * x).sum() + 1e-9)) if len(x) > lag else 0.0


def _smooth(v, k=RHYTHM_W, causal=False):
    if causal:                                       # trailing mean over the current and 2k previous beats
        c = np.cumsum(np.r_[0.0, v])
        idx = np.arange(1, len(v) + 1)
        lo = np.maximum(0, idx - (2 * k + 1))
        return (c[idx] - c[lo]) / (idx - lo)
    pad = np.pad(v, k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")


def _pwave_features(w, causal=False):
    """P-wave consistency: similarity of each beat's P-wave segment to the previous beat's and to the
    local median P segment, averaged over +-15 beats (causal: the previous 30 beats). Regular sinus/atrial
    rhythm -> high; AF -> low."""
    p = _znorm_rows(w[:, PWAVE])
    consec = np.r_[0.0, (p[1:] * p[:-1]).mean(1)]
    consec[0] = consec[1] if len(consec) > 1 else 0.0
    local = np.empty(len(p))
    for i in range(len(p)):
        lo, hi = (max(0, i - 2 * RHYTHM_W), i + 1) if causal else (max(0, i - RHYTHM_W), i + RHYTHM_W + 1)
        med = np.median(p[lo:hi], 0)
        med = (med - med.mean()) / (med.std() + 1e-8)
        local[i] = (p[i] * med).mean()
    return np.stack([_smooth(consec, causal=causal), _smooth(local, causal=causal)], 1)


def _rr_and_rhythm(rr, ks, causal=False):
    """RR features (7, log ratios) and RR-irregularity features (6) for the beats ks.
    Offline: context on both sides. Causal (real-time): past beats plus the next R peak (one-beat delay)."""
    rrf, rhy = [], []
    for k in ks:
        pre, post = max(rr[k - 1], 1e-3), max(rr[k], 1e-3)
        prev = max(rr[k - 2], 1e-3) if k >= 2 else pre
        l10 = rr[max(0, k - 10):k].mean()
        l50 = rr[max(0, k - 50):k].mean() if causal else rr[max(0, k - 25):k + 25].mean()
        rrf.append([pre / l10, post / l10, pre / post, pre / prev, l10 / l50, pre / l50, post / l50])
        seg = rr[max(0, k - 2 * RHYTHM_W):k + 1] if causal else rr[max(0, k - RHYTHM_W):k + RHYTHM_W]
        m = seg.mean()
        jumps = np.diff(seg)
        rhy.append([seg.std() / m, np.sqrt((jumps ** 2).mean()) / m, (np.abs(jumps) / m > 0.1).mean(),
                    np.median(np.abs(jumps)) / np.median(seg), _autocorr(seg, 1), _autocorr(seg, 2)])
    return np.log(np.clip(np.asarray(rrf), 1e-3, None)), np.asarray(rhy)


REFRESH, HISTORY = 50, 300       # real-time templates: re-estimated every 50 beats from the previous 300 beats


def _template_features_causal(w, ontime):
    """Real-time version of _template_features: each block of 50 beats is compared with the beat patterns
    found in the previous 300 beats. The first block uses its own 50 beats as a warm-up (about 40 s of look-ahead,
    once, at the start of a recording). Returns also the correlation with a past-only single median beat."""
    n = len(w)
    diff = np.empty_like(w)
    feats = np.empty((n, 4), np.float32)
    nearest = np.zeros(n, np.int64)
    consist = np.empty(n, np.float32)
    n_max = 1
    for c in range(0, n, REFRESH):
        lo, hi = (0, min(n, REFRESH)) if c == 0 else (max(0, c - HISTORY), c)
        templates = find_templates(w[lo:hi], ontime[lo:hi])
        n_max = max(n_max, len(templates))
        blk = slice(c, min(n, c + REFRESH))
        wb = w[blk]
        a = _corr_matrix(wb, templates).argmax(1)
        t = templates[a]
        d = wb - t
        diff[blk] = d
        nearest[blk] = a
        feats[blk] = np.stack([(_znorm_rows(wb) * _znorm_rows(t)).mean(1),
                               (_znorm_rows(wb[:, QRS]) * _znorm_rows(t[:, QRS])).mean(1),
                               np.sqrt((d ** 2).mean(1)), np.sqrt((d[:, QRS] ** 2).mean(1))], 1)
        consist[blk] = _corr_rows(wb, np.median(w[lo:hi], 0))
    return diff, feats, nearest, n_max, consist


def beats_from_peaks(xa, xb, peaks, labels, af, realtime=False):
    """peaks: all beat positions of the record (defines the rhythm);
    labels: AAMI label per peak, or -1 for beats that are not classified (Q, unmatched).
    realtime=True: every feature uses past beats only, plus the next R peak (one-beat delay)."""
    rr = np.diff(peaks) / FS
    ks = [k for k in range(1, len(peaks) - 1)
          if peaks[k] - BEFORE >= 0 and peaks[k] + AFTER <= min(len(xa), len(xb))]
    pk = peaks[ks]
    wa, wb = _windows(xa, pk), _windows(xb, pk)
    pre_rr = np.array([max(rr[k - 1], 1e-3) / rr[max(0, k - 10):k].mean() for k in ks])
    ontime = np.abs(np.log(pre_rr)) < ONTIME
    if realtime:
        da, ta, tmpl, n_tmpl, consist = _template_features_causal(wa, ontime)
        db, tb, _, _, _ = _template_features_causal(wb, ontime)
    else:
        da, ta, tmpl, n_tmpl = _template_features(wa, ontime)
        db, tb, _, _ = _template_features(wb, ontime)
        consist = _corr_rows(wa, np.median(wa, 0))
    rrf, rhy = _rr_and_rhythm(rr, ks, realtime)
    rhythm = np.hstack([rhy, _pwave_features(wa, realtime)])

    y = np.asarray([labels[k] for k in ks])
    keep = y >= 0
    x = np.stack([wa, da, wb, db], 1)
    f = np.hstack([rrf, ta, tb])
    return {"x": x[keep].astype(np.float16), "f": f[keep].astype(np.float32),
            "r": rhythm[keep].astype(np.float32), "y": y[keep].astype(np.int64),
            "af": np.asarray([af[k] for k in ks])[keep],
            "tmpl": tmpl[keep].astype(np.int8), "n_tmpl": np.full(int(keep.sum()), n_tmpl, np.int8),
            "consist": consist[keep].astype(np.float32)}


def rhythm_only(db_dir, rec, db):
    """Rhythm features and AF flags only (no beat classes), from the database's own QRS annotations ('qrs').
    Used for the MIT-BIH Atrial Fibrillation Database, which has rhythm labels but no beat labels."""
    path = str(Path(db_dir) / str(rec))
    sig = wfdb.rdrecord(path, channels=[0])
    raw = np.nan_to_num(sig.p_signal[:, 0])
    ratio = Fraction(FS, int(round(sig.fs))).limit_denominator(1000)
    xa = _bandpass(resample_poly(raw, ratio.numerator, ratio.denominator))
    qrs = wfdb.rdann(path, "qrs")
    peaks = np.round(np.asarray(qrs.sample, np.float64) * FS / sig.fs).astype(int)
    rann = wfdb.rdann(path, "atr")
    starts = np.round(np.asarray(rann.sample, np.float64) * FS / sig.fs).astype(int)
    notes = [n.strip("\x00").strip() for n in rann.aux_note]
    rr = np.diff(peaks) / FS
    ks = [k for k in range(1, len(peaks) - 1) if peaks[k] - BEFORE >= 0 and peaks[k] + AFTER <= len(xa)]
    pk = peaks[ks]
    idx = np.searchsorted(starts, pk, side="right") - 1
    af = np.array([i >= 0 and notes[i] in ("(AFIB", "(AFL") for i in idx])
    _, rhy = _rr_and_rhythm(rr, ks)
    r = np.hstack([rhy, _pwave_features(_windows(xa, pk))])
    return r.astype(np.float32), af


def load_record(db_dir, rec, peaks="annotated", db="mitdb", realtime=False):
    xa, xb, ref, syms, ref_af = read_record(db_dir, rec, db)
    ref_labels = np.array([AAMI.get(s, -1) for s in syms])
    if peaks == "annotated":
        return beats_from_peaks(xa, xb, ref, ref_labels, ref_af, realtime), None
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
    out = {f"{prefix}_{k}": np.concatenate([p[k] for p in parts]) for k in ["x", "f", "r", "y", "af", "tmpl", "n_tmpl", "consist"]}
    out[f"{prefix}_rec"] = np.concatenate([np.full(len(p["y"]), i) for i, p in enumerate(parts)])
    out[f"{prefix}_recname"] = np.array([str(r) for r in recs])
    return out


def build_mitdb(peaks="annotated", realtime=False):
    out, stats = {}, []
    for name, recs in [("train", [r for r in DS1 if r not in VAL]), ("val", VAL), ("test", DS2)]:
        res = [load_record(MITDB_DIR, r, peaks, "mitdb", realtime) for r in recs]
        part = _stack([r[0] for r in res], recs, name)
        part[f"{name}_rec"] = np.asarray(recs)[part[f"{name}_rec"]]     # real record numbers
        out.update(part)
        stats += [r[1] for r in res if r[1]]
    np.savez_compressed(beats_path("mitdb", peaks, realtime), **out)
    return out, stats


def records(db_dir):
    return [r.strip() for r in (Path(db_dir) / "RECORDS").read_text().split() if r.strip()]


def build_external(db, peaks="annotated", realtime=False):
    db_dir = {"svdb": SVDB_DIR, "incart": INCART_DIR, "nstdb": NSTDB_DIR}[db]
    recs = records(db_dir) if db != "nstdb" else NSTDB_RECORDS
    res = [load_record(db_dir, r, peaks, db, realtime) for r in recs]
    out = _stack([r[0] for r in res], recs, "ext")
    np.savez_compressed(beats_path(db, peaks, realtime), **out)
    return out, [r[1] for r in res if r[1]]


def beats_path(db, peaks="annotated", realtime=False):
    return DATA_DIR / f"{db}_{peaks}_{FEATURE_VERSION}{'_realtime' if realtime else ''}.npz"


def _load(path, builder):
    if not path.exists():
        print(f"Extracting beats -> {path.name} (first run only)...")
        builder()
    d = dict(np.load(path))
    for k in list(d):
        if k.endswith("_x"):
            d[k] = d[k].astype(np.float32)
    return d


def load_splits(peaks="annotated", realtime=False):
    return _load(beats_path("mitdb", peaks, realtime), lambda: build_mitdb(peaks, realtime))


def load_external(db="svdb", peaks="annotated", realtime=False):
    return _load(beats_path(db, peaks, realtime), lambda: build_external(db, peaks, realtime))


if __name__ == "__main__":
    import sys
    peaks = sys.argv[1] if len(sys.argv) > 1 else "annotated"
    d, _ = build_mitdb(peaks)
    for s in ["train", "val", "test"]:
        counts = np.bincount(d[f"{s}_y"], minlength=4)
        print(f"{s:5s} beats={len(d[f'{s}_y']):6d}  AF beats={int(d[f'{s}_af'].sum()):5d}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
