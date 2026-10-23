"""Joint rhythm-and-beat model: one classifier sees the beat features AND the rhythm context
(the same 8 features the AF detector uses), instead of a post-hoc AF rule.

The model family is chosen by leave-patient-out cross-validation on the 18 DS1 training patients
(4-class macro-F1), among a logistic regression, a logistic regression with rhythm x RR interactions
(so a linear model can learn "an early beat in an irregular rhythm is not S") and gradient boosting.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import N_LEAD_A_FEATURES
from .metrics import macro_f1


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced"))


CANDIDATES = {
    "logistic_regression": (_lr, False),
    "logistic_regression_interactions": (_lr, True),
    "gradient_boosting": (lambda: HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                                 random_state=0), False),
}


def joint_features(f, r, interactions=False):
    f = f[:, :N_LEAD_A_FEATURES]
    x = [f, r]
    if interactions:                 # RR features x rhythm irregularity (rr_cv, frac_big_jumps)
        x += [f[:, :7] * r[:, [0]], f[:, :7] * r[:, [2]]]
    return np.hstack(x)


def _proba(model, x):
    p = np.zeros((len(x), 4))
    p[:, model.classes_] = model.predict_proba(x)
    return p


class JointModel:
    def fit(self, f, r, y, groups):
        self.cv = {}
        for name, (make, inter) in CANDIDATES.items():
            x = joint_features(f, r, inter)
            oof = np.zeros((len(y), 4))
            for tr, te in GroupKFold(6).split(x, y, groups):
                oof[te] = _proba(make().fit(x[tr], y[tr]), x[te])
            self.cv[name] = round(macro_f1(y, oof.argmax(1)), 4)
        self.name = max(self.cv, key=self.cv.get)
        make, self.inter = CANDIDATES[self.name]
        self.model = make().fit(joint_features(f, r, self.inter), y)
        return self

    def predict_proba(self, f, r):
        return _proba(self.model, joint_features(f, r, self.inter))

    def summary(self):
        return {"chosen": self.name, "cv_macro_f1": self.cv}
