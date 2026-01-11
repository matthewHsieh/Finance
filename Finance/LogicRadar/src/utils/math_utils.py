import pandas as pd
import numpy as np

def compute_max_lag_correlation(target_series: pd.Series, driver_series: pd.Series, max_lookback: int = 60) -> tuple:
    """
    Computes the maximum correlation between target and driver with shifts.
    Positive Lag means Driver leads Target.
    
    Returns: (best_lag, max_correlation)
    """
    best_lag = 0
    max_corr = 0
    
    # We test lags: Driver(t - lag) vs Target(t)
    # This checks if past driver values predict current target.
    # Lags: 0, 5, 10, ... 60
    lags = [0, 5, 10, 20, 40, 60]
    
    for lag in lags:
        # Shift driver forward by 'lag' days to align 'lag' days ago with today
        shifted_driver = driver_series.shift(lag)
        
        # Validation: Avoid RuntimeWarning for constant/zero-variance data (e.g. Monthly Macro in short window)
        if shifted_driver.std() == 0 or target_series.std() == 0:
            corr = 0.0
        else:
            # Correlation on common available data
            corr = target_series.corr(shifted_driver)
        
        if pd.isna(corr):
            continue
            
        if abs(corr) > abs(max_corr):
            max_corr = corr
            best_lag = lag
            
    return best_lag, max_corr

def calculate_rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculates rolling correlation.
    """
    return series_a.rolling(window=window).corr(series_b)

def calculate_z_score(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculates rolling Z-Score.
    """
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std
