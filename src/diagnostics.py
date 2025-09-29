import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from typing import Dict, Any, List

from .config import Config
from .modeling import make_pipeline

def plot_reliability_diagram(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 12):
    """Plots a reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, p_hat, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Reliability Diagram (OOF)")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_permutation_importance(X: pd.DataFrame, y: pd.Series, feature_names: List[str], config: Config):
    """Calculates and plots permutation feature importance."""
    from sklearn.model_selection import train_test_split

    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.25, shuffle=False)

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    model = make_pipeline(config, feature_names, scale_pos_weight=spw)
    model.fit(X_tr, y_tr)

    pi = permutation_importance(
        model, X_ho, y_ho,
        n_repeats=10, scoring="roc_auc", random_state=config.random_seed, n_jobs=-1
    )

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['feature'], imp_df['importance_mean'], xerr=imp_df['importance_std'])
    plt.title("Permutation Feature Importance (ROC-AUC on Holdout)")
    plt.gca().invert_yaxis()
    plt.show()

    return imp_df

def plot_equity_curves(backtest_results: Dict[str, Any], config: Config):
    """Plots the equity curves from backtest results."""
    plt.figure(figsize=(10, 6))
    for name, result in backtest_results.items():
        plt.plot(result['equity_curve'][config.date_col], result['equity_curve']['equity'], label=name)

    plt.title("Strategy Equity Curves (OOF, with costs)")
    plt.xlabel(config.date_col)
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def generate_summary_report(
    config: Config,
    prices: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    model_results: Dict[str, Any],
    backtest_results: Dict[str, Any]
):
    """Generates and prints a summary report of the entire process."""

    summary = {
        "Data": {
            "Ticker": config.ticker,
            "Date Range": f"{config.start_date} to {config.end_date}",
            "Raw Rows": len(prices),
            "Model Rows": len(y),
            "Features": X.shape[1],
        },
        "Labeling": {
            "Method": config.label_method,
            "Positive Rate": f"{y.mean():.4f}",
        },
        "Walk-Forward Validation": {
            "Folds": len(model_results['fold_metrics']),
            "Train Size": f"{config.train_size:.0%}",
            "Test Size": f"{config.test_size:.0%}",
            "Gap": f"{config.gap} days",
        },
        "OOF Metrics (@0.5)": model_results['oof_metrics_@0.5'],
        "OOF Metrics (@best)": model_results['oof_metrics_@best_thr'],
        "Best Threshold": f"{model_results['best_threshold']:.3f}",
        "Backtest (Long-Only)": backtest_results['long_only']['stats'],
        "Backtest (Long-Short)": backtest_results['long_short']['stats'],
    }

    print("--- FRACTAL MODEL SUMMARY REPORT ---")
    for section, data in summary.items():
        print(f"\n## {section}")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {data}")
    print("\n------------------------------------")