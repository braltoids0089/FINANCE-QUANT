import pandas as pd
import numpy as np
from typing import Any
from .config import Config

def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Handle both shapes from yfinance:
    - Single-level columns: ['Open','High','Low','Close','Adj Close','Volume']
    - MultiIndex columns:
        (a) top level = fields, second level = tickers
        (b) top level = tickers, second level = fields
    Returns a DataFrame with single-level columns for one ticker.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df  # already flat

    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    if ticker in lvl0:
        sub = df[ticker]
    elif ticker in lvl1:
        sub = df.xs(ticker, axis=1, level=1)
    else:
        uniq0, uniq1 = sorted(set(lvl0)), sorted(set(lvl1))
        if len(uniq0) == 1 and uniq0[0] not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            sub = df.xs(uniq0[0], axis=1, level=0)
        elif len(uniq1) == 1:
            sub = df.xs(uniq1[0], axis=1, level=1)
        else:
            raise ValueError(
                f"Ticker '{ticker}' not found in yfinance columns. "
                f"Level0={uniq0[:6]}..., Level1={uniq1[:6]}..."
            )
    sub.columns = [str(c) for c in sub.columns]
    return sub

def load_price_data(config: Config) -> pd.DataFrame:
    """
    Loads price data using yfinance or from a CSV file based on the config.
    """
    if config.use_yfinance:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Please run: pip install yfinance")

        df = yf.download(
            config.ticker, start=config.start_date, end=config.end_date,
            auto_adjust=False, progress=False, group_by='ticker'
        )
        if df.empty:
            raise ValueError("Downloaded data is empty. Check ticker or date range.")

        df = _flatten_yf_columns(df, config.ticker)

        cmap = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'adj close': 'Adj Close', 'volume': 'Volume'
        }
        df = df.rename(columns={c: cmap.get(c.lower(), c) for c in df.columns})

        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[keep].copy()

        df.index = pd.to_datetime(df.index)
        df.index.name = config.date_col
        df = df.sort_index()
        df = df.loc[~df.index.duplicated(keep='first')]

        for c in keep:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Close']).reset_index()

    else:
        df = pd.read_csv(config.csv_path)
        df.columns = [c.strip() for c in df.columns]
        ren = {}
        for c in df.columns:
            cl = c.lower()
            if cl == 'date': ren[c] = config.date_col
            elif cl == 'open': ren[c] = 'Open'
            elif cl == 'high': ren[c] = 'High'
            elif cl == 'low': ren[c] = 'Low'
            elif cl == 'close': ren[c] = 'Close'
            elif cl == 'volume': ren[c] = 'Volume'
        df = df.rename(columns=ren)
        df[config.date_col] = pd.to_datetime(df[config.date_col])
        df = df.sort_values(config.date_col).drop_duplicates(config.date_col).reset_index(drop=True)

    print(f"Loaded {len(df)} rows for {config.ticker} [{config.start_date} → {config.end_date}].")
    return df