"""Reusable outlier utilities (IQR, Z-score, Winsorization)."""
import pandas as pd
import numpy as np

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = (s < lower) | (s > upper)
    return out.fillna(False)

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(False, index=series.index)
    z = (s - mu) / sigma
    return (z.abs() > threshold).fillna(False)

def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = pd.to_numeric(series, errors="coerce")
    try:
        lower_f = float(lower); upper_f = float(upper)
    except Exception:
        lower_f, upper_f = 0.05, 0.95
    if not (0 <= lower_f <= 1 and 0 <= upper_f <= 1) or lower_f >= upper_f:
        return s
    q_low = s.quantile(lower_f)
    q_high = s.quantile(upper_f)
    return s.clip(lower=q_low, upper=q_high)
