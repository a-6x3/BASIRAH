"""
InsiderGuard engine: model training/loading, scoring, context-aware priority,
and per-feature explanations. Shared by the offline build script and the app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

FEATURES = ["logon_count", "off_hours_logons", "usb_events", "http_events"]

# A day is "unusual for this user" when a feature exceeds the user's own
# 95th percentile (over their previous 60 active days) by more than a margin.
TOL_ABS = np.array([3.0, 1.0, 1.0, 20.0])   # minimum margin per feature
TOL_REL = np.array([0.5, 0.0, 0.0, 0.5])    # plus a share of the user's p95

BASELINE_WINDOW = 60
BASELINE_MIN_DAYS = 7


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def train_bundle(X: pd.DataFrame) -> dict:
    """Same model as the original pipeline (3_train_model.py)."""
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(X[FEATURES].fillna(0))
    raw = model.decision_function(X[FEATURES].fillna(0))
    return {
        "model": model,
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "sklearn_version": sklearn.__version__,
    }


def load_or_train(model_path: Path, train_df: pd.DataFrame) -> tuple[dict, bool]:
    """Load the saved model; retrain from the shipped table if the saved one
    was built with a different scikit-learn version. Returns (bundle, retrained)."""
    try:
        bundle = joblib.load(model_path)
        if bundle.get("sklearn_version") == sklearn.__version__:
            return bundle, False
    except Exception:
        pass
    return train_bundle(train_df), True


def risk_from_raw(raw: np.ndarray, bundle: dict) -> np.ndarray:
    span = bundle["raw_max"] - bundle["raw_min"]
    return np.clip((bundle["raw_max"] - raw) / span * 100.0, 0.0, 100.0)


def score_matrix(bundle: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X: (n, 4) in FEATURES order -> (risk 0-100, flagged bool)."""
    X = np.asarray(X, dtype=float).reshape(-1, len(FEATURES))
    df = pd.DataFrame(X, columns=FEATURES)
    raw = bundle["model"].decision_function(df)
    flagged = bundle["model"].predict(df) == -1
    return risk_from_raw(raw, bundle), flagged


# --------------------------------------------------------------------------
# Context-aware layer (the user's own baseline)
# --------------------------------------------------------------------------
def unusual_flags(x: np.ndarray, p95: np.ndarray) -> np.ndarray:
    """Boolean per feature: is today's value unusual for THIS user?
    x, p95: shape (..., 4). NaN baseline -> False (handled by the caller)."""
    margin = np.maximum(TOL_ABS, TOL_REL * p95)
    return x > (p95 + margin)


def priority(flagged: bool, any_unusual: bool, has_baseline: bool = True) -> str:
    if not flagged:
        return ""
    if not has_baseline:
        return "HIGH"          # no history yet: stay conservative
    return "HIGH" if any_unusual else "ROUTINE"


def contributions(bundle: dict, x: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Model-faithful explanation by ablation: how many risk points disappear
    if each feature is replaced by the user's usual (median) value."""
    x = np.asarray(x, float)
    neutral = np.asarray(neutral, float)
    rows = [x] + [np.where(np.arange(len(x)) == i, neutral, x) for i in range(len(x))]
    risk, _ = score_matrix(bundle, np.vstack(rows))
    return np.maximum(risk[0] - risk[1:], 0.0)


def tier(risk: float) -> str:
    if risk >= 80:
        return "CRITICAL"
    if risk >= 50:
        return "ELEVATED"
    return "LOW"


def build_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-user baseline from the PREVIOUS days only (no peeking)."""
    df = df.sort_values(["user", "date_only"]).reset_index(drop=True)
    g = df.groupby("user", sort=False)
    for f in FEATURES:
        prev = g[f].shift(1)
        r = prev.groupby(df["user"], sort=False).rolling(
            BASELINE_WINDOW, min_periods=BASELINE_MIN_DAYS)
        df[f"b_med_{f}"] = r.median().reset_index(level=0, drop=True).astype("float32")
        df[f"b_p95_{f}"] = r.quantile(0.95).reset_index(level=0, drop=True).astype("float32")
    return df


def baseline_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    med = df[[f"b_med_{f}" for f in FEATURES]].to_numpy(dtype=float)
    p95 = df[[f"b_p95_{f}" for f in FEATURES]].to_numpy(dtype=float)
    return med, p95


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
