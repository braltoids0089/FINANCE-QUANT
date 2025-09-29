from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

@dataclass
class Config:
    # General
    random_seed: int = 42
    date_col: str = 'Date'

    # Data Loading
    use_yfinance: bool = True
    ticker: str = "SPY"
    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    csv_path: str = 'your_prices.csv'  # For CSV loading

    # Feature Engineering
    win_short: int = 256
    win_long: int = 512
    use_wavelets: bool = False
    wavelet: str = 'db2'
    wavelet_lvl: int = 4
    acf_lags: Tuple[int, ...] = (1, 2, 3, 4, 5)
    hurst_scales: Tuple[int, ...] = tuple(range(16, 129, 8))
    dfa_scales: Tuple[int, ...] = tuple(range(16, 129, 8))


    # Labeling
    label_method: str = 'horizon'  # 'horizon' or 'triple_barrier'
    horizon: int = 20
    dead_zone: float = 0.0002
    tb_pt: float = 2.0
    tb_sl: float = 2.0
    tb_max_h: int = 50
    vol_ewm_span: int = 50

    # Walk-Forward Validation
    train_size: float = 0.55
    test_size: float = 0.15
    gap: int = 5

    # Modeling
    calib_method: str = "sigmoid"
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    })

    # Hyperparameter Tuning (Optuna)
    run_tuning: bool = False
    tuning_metric: str = "mcc"
    n_trials: int = 25
    study_name: str = "xgb_fractal_walkforward"

    # Backtesting
    tc_bps: int = 5

# Create a default config instance
default_config = Config()