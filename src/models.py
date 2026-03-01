from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def get_model(name: str = "logreg", calibrate: bool = True, seed: int = 42):
    """
    Returns a classifier that supports predict_proba.

    Models:
      - logreg: fast, stable baseline
      - rf: nonlinear baseline

    If calibrate=True, wraps model with isotonic calibration (better probability quality).
    """
    name = name.lower().strip()

    if name == "logreg":
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )
    elif name == "rf":
        base = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    if not calibrate:
        return base

    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def fit_predict_p_green(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]
