"""Feature baselines on the same split: logistic regression and gradient boosting
on the 11 patient-relative features (7 RR + 4 template). No waveform input.
"""
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.data import load_splits
from src.metrics import report, save_report

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    d = load_splits()
    X = np.concatenate([d["train_f"], d["val_f"]])
    y = np.concatenate([d["train_y"], d["val_y"]])
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=0),
    }
    for name, clf in models.items():
        clf.fit(X, y)
        rep = report(d["test_y"], clf.predict(d["test_f"]))
        save_report(rep, RESULTS, name)
        print(f"{name}  DS2 accuracy {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}")
        for c, v in rep["per_class"].items():
            print(f"  {c}: Se {v['sensitivity']:.3f}  +P {v['ppv']:.3f}  (n={v['support']})")


if __name__ == "__main__":
    main()
