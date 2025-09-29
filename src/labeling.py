import numpy as np
import pandas as pd
from .config import Config

def compute_labels_horizon(df: pd.DataFrame, horizon: int, dead_zone: float) -> pd.Series:
    """
    Binary label from future log return over a fixed horizon.
      y = 1 if future logret > +dead_zone
      y = 0 if future logret < -dead_zone
      y = NaN otherwise (ignored)
    """
    px = df['Close'].astype(float)
    fwd = np.log(px).shift(-horizon) - np.log(px)
    y = pd.Series(np.nan, index=px.index)
    y[fwd > dead_zone] = 1
    y[fwd < -dead_zone] = 0
    return y.rename('target')

def compute_labels_triple_barrier(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_h: int,
    vol_ewm_span: int
) -> pd.Series:
    """
    Triple-barrier labels.
    """
    px = df['Close'].astype(float).values
    idx = df.index
    r = pd.Series(np.log(df['Close']).diff(), index=idx)
    vol = r.ewm(span=vol_ewm_span).std().values

    y = np.full(len(px), np.nan)
    n = len(px)
    for t in range(n):
        if t >= n - 1:
            break
        p0 = px[t]
        # dynamic barriers
        up = p0 * np.exp(pt_mult * (vol[t] if not np.isnan(vol[t]) else 0.0))
        dn = p0 * np.exp(-sl_mult * (vol[t] if not np.isnan(vol[t]) else 0.0))
        last = min(t + max_h, n - 1)

        label = np.nan
        for u in range(t + 1, last + 1):
            pu = px[u]
            if pu >= up:
                label = 1
                break
            if pu <= dn:
                label = 0
                break
        if np.isnan(label):
            label = 1 if px[last] > p0 else 0
        y[t] = label

    return pd.Series(y, index=idx, name='target')

def create_labels(features_df: pd.DataFrame, config: Config) -> pd.Series:
    """
    Dispatcher for creating labels based on the method specified in the config.
    """
    if config.label_method == 'horizon':
        print("Creating labels using fixed horizon method.")
        return compute_labels_horizon(
            features_df,
            horizon=config.horizon,
            dead_zone=config.dead_zone
        )
    elif config.label_method == 'triple_barrier':
        print("Creating labels using triple-barrier method.")
        return compute_labels_triple_barrier(
            features_df,
            pt_mult=config.tb_pt,
            sl_mult=config.tb_sl,
            max_h=config.tb_max_h,
            vol_ewm_span=config.vol_ewm_span
        )
    else:
        raise ValueError(f"Unknown labeling method: {config.label_method}")