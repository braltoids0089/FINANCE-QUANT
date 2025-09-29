import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Iterable, Tuple
from .config import Config

# --- Basic Helpers ---
def logret(x: pd.Series) -> pd.Series:
    """Log returns of a price series."""
    return np.log(x).diff()

def realized_vol(r: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility over a rolling window."""
    return r.rolling(window).std() * np.sqrt(252)

def bipower_variation(r: pd.Series, window: int = 20) -> pd.Series:
    """Bipower variation (annualized) as a robust volatility proxy."""
    lam = np.pi / 2
    a = r.abs()
    prod = a * a.shift(1)
    return lam * prod.rolling(window).mean() * 252

def hl_range(df: pd.DataFrame) -> pd.Series:
    """High–low range normalized by Close."""
    return (df['High'] - df['Low']) / df['Close']

def rolling_acf_absr(series: pd.Series, lags: Iterable[int], window: int) -> pd.DataFrame:
    """Rolling autocorrelation of absolute returns (volatility clustering proxy)."""
    out = pd.DataFrame(index=series.index)
    x = series.values
    for lag in lags:
        vals = []
        for i in range(len(x)):
            if i < window:
                vals.append(np.nan)
                continue
            w = np.abs(x[i-window:i])
            y = w[lag:]
            z = w[:-lag]
            if y.std(ddof=1) == 0 or z.std(ddof=1) == 0:
                vals.append(np.nan)
            else:
                vals.append(np.corrcoef(y, z)[0, 1])
        out[f'acf_absr_lag{lag}_{window}'] = vals
    return out

# --- Hurst Exponent via R/S ---
def _rs_block(seg: np.ndarray) -> float:
    y = seg - seg.mean()
    z = np.cumsum(y)
    R = z.max() - z.min()
    S = y.std(ddof=1)
    return np.log(R / S) if S > 0 else np.nan

def rolling_hurst_rs(series: pd.Series, window: int, scales: Tuple[int, ...]) -> pd.Series:
    """Rolling Hurst estimate using multi-scale R/S regression."""
    out = []
    s = series.values.astype(float)
    for i in range(len(s)):
        if i < window:
            out.append(np.nan)
            continue
        xw = s[i-window:i]
        vals, ns = [], []
        for n in scales:
            if n >= window:
                break
            chunks = len(xw) // n
            if chunks < 4:
                continue
            rs_vals = []
            for k in range(chunks):
                seg = xw[k*n:(k+1)*n]
                rs = _rs_block(seg)
                if not np.isnan(rs):
                    rs_vals.append(np.exp(rs))
            if len(rs_vals) >= 3:
                vals.append(np.log(np.mean(rs_vals)))
                ns.append(np.log(n))
        if len(ns) >= 3:
            X = sm.add_constant(ns)
            beta = sm.OLS(vals, X).fit().params
            H = beta[1]
            out.append(H)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, name=f"H_RS_{window}")

# --- DFA Alpha ---
def dfa_alpha(series: pd.Series, window: int, order: int, scales: Tuple[int, ...]) -> pd.Series:
    """Rolling DFA alpha using multi-scale regression on log F(n) vs log n."""
    out = []
    x = series.values.astype(float)
    for i in range(len(x)):
        if i < window:
            out.append(np.nan)
            continue
        w = x[i-window:i]
        y = np.cumsum(w - w.mean())
        Fn, ln_n = [], []
        for n in scales:
            if n >= window:
                break
            chunks = len(y) // n
            if chunks < 4:
                continue
            errs = []
            for k in range(chunks):
                seg = y[k*n:(k+1)*n]
                t = np.arange(n)
                coeff = np.polyfit(t, seg, order)
                trend = np.polyval(coeff, t)
                errs.append(np.sqrt(np.mean((seg - trend) ** 2)))
            Fn.append(np.mean(errs))
            ln_n.append(np.log(n))
        if len(ln_n) >= 3:
            X = sm.add_constant(ln_n)
            alpha = sm.OLS(np.log(Fn), X).fit().params[1]
            out.append(alpha)
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index, name=f"DFA_{window}")

# --- Wavelet Energy ---
def wavelet_energy_features(series: pd.Series, window: int, wavelet: str, levels: int) -> pd.DataFrame:
    """Energy share of detail coefficients across levels (normalized) over a rolling window."""
    try:
        import pywt
    except ImportError:
        raise ImportError("pywavelets not installed. Please run: pip install PyWavelets")

    idx = series.index
    cols = {f'wl_Ed{j}_{window}': [] for j in range(1, levels + 1)}
    x = series.values
    for i in range(len(x)):
        if i < window:
            for j in range(1, levels + 1):
                cols[f'wl_Ed{j}_{window}'].append(np.nan)
            continue
        w = x[i-window:i]
        coeffs = pywt.wavedec(w, wavelet, level=levels)
        details = coeffs[1:]
        en = [float(np.sum(d**2)) for d in details[::-1]]
        tot = sum(en)
        for j, e in enumerate(en, start=1):
            cols[f'wl_Ed{j}_{window}'].append((e / tot) if tot > 0 else np.nan)
    return pd.DataFrame(cols, index=idx)

# --- Main Feature Creation Function ---
def create_features(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Orchestrates the creation of all features.
    """
    prices = prices.copy()
    prices['logret'] = logret(prices['Close'])

    feature_blocks = [
        prices[['Date', 'Close', 'logret']],
        hl_range(prices).rename('hl_range'),
        realized_vol(prices['logret'], 20).rename('rv20'),
        realized_vol(prices['logret'], 60).rename('rv60'),
        bipower_variation(prices['logret'], 20).rename('bv20'),
        bipower_variation(prices['logret'], 60).rename('bv60'),
        rolling_hurst_rs(prices['logret'].fillna(0.0), window=config.win_short, scales=config.hurst_scales),
        rolling_hurst_rs(prices['logret'].fillna(0.0), window=config.win_long, scales=config.hurst_scales),
        dfa_alpha(prices['logret'].fillna(0.0), window=config.win_short, order=1, scales=config.dfa_scales),
        dfa_alpha(prices['logret'].fillna(0.0), window=config.win_long, order=1, scales=config.dfa_scales),
        rolling_acf_absr(prices['logret'].fillna(0.0), lags=config.acf_lags, window=config.win_short)
    ]

    if config.use_wavelets:
        wl_features = wavelet_energy_features(
            prices['logret'].fillna(0.0), window=config.win_long,
            wavelet=config.wavelet, levels=config.wavelet_lvl
        )
        feature_blocks.append(wl_features)

    features_df = pd.concat(feature_blocks, axis=1)
    print("Feature engineering complete.")
    return features_df