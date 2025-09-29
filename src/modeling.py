import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Tuple, Iterator, List
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from xgboost import XGBClassifier

from .config import Config

# --- Walk-forward splitting ---
def forward_splits(n: int, n_train: int, n_test: int, gap: int) -> Iterator[Tuple[slice, slice]]:
    """Yields (train_slice, test_slice) for a walk-forward scheme."""
    i = 0
    while i + n_train + gap + n_test <= n:
        tr = slice(i, i + n_train)
        te = slice(i + n_train + gap, i + n_train + gap + n_test)
        yield tr, te
        i += n_test

# --- Metrics ---
def evaluate_probs(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Return a dict of robust metrics for probabilistic outputs."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "mcc": matthews_corrcoef(y_true, yhat),
    }

def best_threshold(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Grid-search the decision threshold to maximize MCC."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    grid = np.linspace(0.05, 0.95, 37)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        v = matthews_corrcoef(y_true, (p >= t).astype(int))
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v

# --- Pipeline and Model Training ---
def make_pipeline(config: Config, feature_names: List[str], scale_pos_weight: float = 1.0) -> Pipeline:
    """Creates a scikit-learn pipeline with preprocessing and an XGBoost classifier."""
    xgb_params = config.xgb_params.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight

    base = XGBClassifier(**xgb_params)

    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_names)],
        remainder="drop"
    )

    clf = CalibratedClassifierCV(estimator=base, cv=3, method=config.calib_method)
    return Pipeline([("prep", pre), ("clf", clf)])

def run_walk_forward(X: pd.DataFrame, y: pd.Series, config: Config) -> Dict[str, Any]:
    """
    Runs the main walk-forward validation loop.
    Optionally performs hyperparameter tuning if config.run_tuning is True.
    """
    if config.run_tuning:
        print("Hyperparameter tuning is not yet fully implemented in this refactoring.")

    all_probs = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    fold_metrics_list = []

    n_total = len(y)
    n_train = int(n_total * config.train_size)
    n_test = int(n_total * config.test_size)

    feature_names = X.columns.tolist()

    print(f"Starting walk-forward validation with {n_train} train samples and {n_test} test samples per fold.")
    for fold_id, (tr, te) in enumerate(forward_splits(n_total, n_train, n_test, config.gap), 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        pos = int(y_tr.sum())
        neg = int(len(y_tr) - pos)
        spw = (neg / pos) if pos > 0 else 1.0

        model = make_pipeline(config, feature_names, scale_pos_weight=spw)
        model.fit(X_tr, y_tr)

        p = model.predict_proba(X_te)[:, 1]
        all_probs[te] = p

        metrics = evaluate_probs(y_te.values, p, thr=0.5)
        metrics['fold'] = fold_id
        fold_metrics_list.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metrics_list)

    # OOF results
    mask = ~np.isnan(all_probs)
    y_oof = y.values[mask]
    p_oof = all_probs[mask]

    t_best, m_best = best_threshold(y_oof, p_oof)

    oof_metrics_0_5 = evaluate_probs(y_oof, p_oof, thr=0.5)
    oof_metrics_best = evaluate_probs(y_oof, p_oof, thr=t_best)

    # Add the date index to the OOF predictions
    oof_dates = X.index[mask]
    oof_predictions = pd.DataFrame({"y_true": y_oof, "p_hat": p_oof}, index=oof_dates)

    results = {
        "oof_predictions": oof_predictions,
        "fold_metrics": fold_metrics_df,
        "oof_metrics_@0.5": oof_metrics_0_5,
        "oof_metrics_@best_thr": oof_metrics_best,
        "best_threshold": t_best,
    }

    print("Walk-forward validation complete.")
    return results