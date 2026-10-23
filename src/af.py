"""Atrial fibrillation (AF) detector and the AF-aware rule.

Why: during AF the rhythm is irregular, so normal beats arrive "early" and look like S beats to an
RR-based classifier. By definition there are no supraventricular ectopic beats inside AF, so beats
predicted as S while the AF detector is on are relabelled N.

The detector works on rhythm context only (RR irregularity over +-15 beats and P-wave consistency).
Model family and threshold are chosen with leave-patient-out cross-validation on the DS1 training
patients. Test data are never used for these choices.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATES = {
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200, class_weight="balanced", random_state=0),
}
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def _f1(pred, true):
    tp = (pred & true).sum()
    return 2 * tp / (pred.sum() + true.sum() + 1e-9)


class AFDetector:
    def fit(self, r, af, groups):
        self.cv = {}
        for name, make in CANDIDATES.items():
            oof = np.zeros(len(r))
            for tr, te in GroupKFold(5).split(r, af, groups):
                oof[te] = make().fit(r[tr], af[tr]).predict_proba(r[te])[:, 1]
            scores = {float(t): float(_f1(oof > t, af)) for t in THRESHOLDS}
            best_t = max(scores, key=scores.get)
            self.cv[name] = {"threshold": best_t, "cv_f1": round(scores[best_t], 4)}
        self.name = max(self.cv, key=lambda n: self.cv[n]["cv_f1"])
        self.threshold = self.cv[self.name]["threshold"]
        self.model = CANDIDATES[self.name]().fit(r, af)
        return self

    def predict_proba(self, r):
        return self.model.predict_proba(r)[:, 1]

    def predict(self, r):
        return self.predict_proba(r) > self.threshold

    def summary(self):
        return {"chosen": self.name, "threshold": self.threshold, "cross_validation": self.cv}


def apply_af_rule(pred, af_pred):
    """S predictions inside detected AF become N."""
    out = pred.copy()
    out[(out == 1) & af_pred] = 0
    return out


def af_metrics(pred, true):
    tp = int((pred & true).sum())
    return {"af_beats": int(true.sum()), "sensitivity": round(tp / max(true.sum(), 1), 4),
            "ppv": round(tp / max(pred.sum(), 1), 4),
            "specificity": round(float(((~pred) & (~true)).sum() / max((~true).sum(), 1)), 4)}
