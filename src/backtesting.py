import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .config import Config

def performance_stats(equity_curve: pd.Series, freq: int = 252) -> Dict[str, float]:
    """Return simple CAGR, Sharpe (daily->annual), and max drawdown."""
    rets = np.log(equity_curve).diff().dropna()
    if rets.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}

    ann_ret = float(np.exp(rets.mean() * freq) - 1.0)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(freq))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan

    peak = equity_curve.cummax()
    dd = (equity_curve / peak - 1.0)
    mdd = float(dd.min())

    return {"CAGR": ann_ret, "Sharpe": sharpe, "MaxDD": mdd}

def backtest_from_oof(
    oof_df: pd.DataFrame,
    prices: pd.DataFrame,
    t_best: float,
    config: Config,
    mode: str = "long_only"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs a backtest from out-of-fold predictions.
    """
    # Ensure the date column is in the correct format for merging
    prices[config.date_col] = pd.to_datetime(prices[config.date_col])
    oof_df[config.date_col] = pd.to_datetime(oof_df[config.date_col])

    df = pd.merge(oof_df, prices[[config.date_col, 'logret']], on=config.date_col, how='left')
    df = df.sort_values(config.date_col).reset_index(drop=True)

    df['r1'] = df['logret'].shift(-1)
    df = df.dropna(subset=['r1']).reset_index(drop=True)

    if mode == "long_only":
        df['pos'] = (df['p_hat'] >= t_best).astype(int)
    elif mode == "long_short":
        df['pos'] = 0
        df.loc[df['p_hat'] >= t_best, 'pos'] = 1
        df.loc[df['p_hat'] <= (1 - t_best), 'pos'] = -1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df['pos_shift'] = df['pos'].shift(1).fillna(0)
    df['turnover'] = (df['pos_shift'] - df['pos_shift'].shift(1)).abs().fillna(df['pos_shift'].abs())

    tc = (config.tc_bps / 1e4)
    df['ret_strategy'] = df['pos_shift'] * df['r1'] - df['turnover'] * tc

    df['equity'] = np.exp(df['ret_strategy'].cumsum())
    stats = performance_stats(df['equity'])

    return df, stats

def run_backtests(
    oof_predictions: pd.DataFrame,
    prices: pd.DataFrame,
    best_threshold: float,
    config: Config
) -> Dict[str, Any]:
    """
    Runs both long-only and long-short backtests.
    """
    prices_with_logret = prices.copy()
    prices_with_logret['logret'] = np.log(prices_with_logret['Close']).diff()

    bt_long, st_long = backtest_from_oof(
        oof_predictions, prices_with_logret, best_threshold, config, mode="long_only"
    )
    bt_ls, st_ls = backtest_from_oof(
        oof_predictions, prices_with_logret, best_threshold, config, mode="long_short"
    )

    print("Backtesting complete.")
    return {
        "long_only": {"equity_curve": bt_long, "stats": st_long},
        "long_short": {"equity_curve": bt_ls, "stats": st_ls},
    }