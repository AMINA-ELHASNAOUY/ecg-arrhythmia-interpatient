"""Fully automatic pipeline: detect R peaks with XQRS instead of using the expert annotations.

1. QRS detector quality against the annotations (AAMI 150 ms matching window).
2. The feature model (logistic regression) trained and tested on automatically detected beats.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DS1, DS2, build_mitdb, load_splits
from src.metrics import report, save_report

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    _, stats = build_mitdb("detected")
    tp = sum(s["tp"] for s in stats)
    fn = sum(s["fn"] for s in stats)
    fp = sum(s["fp"] for s in stats)
    summary = {
        "records": len(stats),
        "reference_beats": tp + fn,
        "sensitivity": round(tp / (tp + fn), 5),
        "ppv": round(tp / (tp + fp), 5),
        "worst_records": sorted(stats, key=lambda s: s["fn"] + s["fp"], reverse=True)[:5],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detector.json").write_text(json.dumps({"summary": summary, "per_record": stats}, indent=2))
    print(f"QRS detector on {len(stats)} records ({len(DS1)} DS1 + {len(DS2)} DS2): "
          f"Se {summary['sensitivity']:.4f}  +P {summary['ppv']:.4f}  (FN {fn}, FP {fp})")

    d = load_splits("detected")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(np.concatenate([d["train_f"], d["val_f"]]), np.concatenate([d["train_y"], d["val_y"]]))
    rep = report(d["test_y"], clf.predict(d["test_f"]))
    rep["note"] = "beats located by XQRS; missed beats are not classified (see detector.json)"
    save_report(rep, RESULTS, "logistic_regression_detected")
    print(f"Logistic regression on detected beats  DS2 accuracy {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}")
    for c, v in rep["per_class"].items():
        print(f"  {c}: Se {v['sensitivity']:.3f}  +P {v['ppv']:.3f}  (n={v['support']})")


if __name__ == "__main__":
    main()
