"""Default classifier definitions used for synthetic data evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_default_classifiers(random_state: int = 42) -> dict[str, Any]:
    """Return a dictionary of default sklearn-compatible classifiers.

    All classifiers support ``fit``, ``predict``, and ``predict_proba``.

    Parameters
    ----------
    random_state : int, default 42
        Seed for reproducibility where applicable.

    Returns
    -------
    dict[str, estimator]
        Mapping of short model names to unfitted estimators.

    Notes
    -----
    Optional boosting models (LightGBM, XGBoost) are included only when the
    corresponding package is installed.  This keeps the core dependency list
    small while allowing power users to leverage gradient boosting.
    """
    models: dict[str, Any] = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=random_state)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "RF": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    # ── optional: LightGBM ──────────────────────────────────────────────
    try:
        from lightgbm import LGBMClassifier

        models["LGB"] = LGBMClassifier(
            n_estimators=100, random_state=random_state, verbose=-1
        )
    except ImportError:  # pragma: no cover
        pass

    # ── optional: XGBoost ───────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier

        models["XGB"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    except ImportError:  # pragma: no cover
        pass

    return models
