"""Utilities for computing fundamental quality scores."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

QUALITY_METRIC_ALIASES = {
    "roe": ("roe", "return_on_equity", "roe_ttm", "returnOnEquityTTM"),
    "free_cash_flow": (
        "free_cash_flow",
        "fcf",
        "freeCashFlow",
        "freeCashFlowTTM",
        "free_cash_flow_ttm",
        "fcf_ttm",
    ),
    "net_debt_ebitda": (
        "net_debt_ebitda",
        "netDebtEbitda",
        "netDebtToEBITDA",
        "netDebtEBITDA",
    ),
    "interest_coverage": (
        "interest_coverage",
        "interestCoverage",
        "interestCoverageTTM",
    ),
    "margin_trend": ("margin_trend_5y", "marginTrend5Y"),
}


def _coerce_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return ``series`` converted to ``float`` with the provided ``index``."""

    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.reindex(index).astype(float)


def _extract_metric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return a numeric series for the first present column in ``candidates``."""

    for column in candidates:
        if column in df.columns:
            return _coerce_numeric(df[column], df.index)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _zscore(series: pd.Series) -> pd.Series:
    """Compute the population z-score of ``series`` with NaN-safe handling."""

    if series.empty:
        return pd.Series(dtype=float)

    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)

    mean = valid.mean()
    std = valid.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)

    z = (values - mean) / std
    return z.fillna(0.0)


def compute_quality_scores(df: pd.DataFrame) -> pd.Series:
    """Return the quality factor score for each row in ``df``.

    The implementation awards one point for each quality signal and normalises
    the sum to the 0-1 range before applying a z-score across the entire
    universe. Missing inputs default to a score contribution of zero so that the
    final result does not devolve into ``NaN`` values when data points are
    unavailable.
    """

    if df.empty:
        return pd.Series(dtype=float)

    roe = _extract_metric(df, QUALITY_METRIC_ALIASES["roe"])
    free_cash_flow = _extract_metric(df, QUALITY_METRIC_ALIASES["free_cash_flow"])
    net_debt_ebitda = _extract_metric(df, QUALITY_METRIC_ALIASES["net_debt_ebitda"])
    interest_coverage = _extract_metric(df, QUALITY_METRIC_ALIASES["interest_coverage"])
    margin_trend = _extract_metric(df, QUALITY_METRIC_ALIASES["margin_trend"])

    signals = [
        (roe > 10).fillna(False).astype(float),
        (free_cash_flow > 0).fillna(False).astype(float),
        (net_debt_ebitda < 2.5).fillna(False).astype(float),
        (interest_coverage > 5).fillna(False).astype(float),
        (margin_trend > 0).fillna(False).astype(float),
    ]

    raw_score = sum(signals)
    normalised = raw_score / len(signals)
    return _zscore(normalised)


__all__ = ["compute_quality_scores"]
