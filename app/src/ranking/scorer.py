"""Helpers to compute composite ranking scores for securities.

This module provides a light-weight interface for turning the various
fundamental and technical inputs into a single composite score that can be
used for ranking.  The functions favour robustness over mathematical
perfection – they guard against ``NaN`` values, empty inputs and divisions by
zero so the pipeline can operate on incomplete market data without crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return ``values`` as a float ``Series`` aligned to ``index``."""

    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)

    if not isinstance(values, pd.Series):
        coerced = pd.Series(values, index=index)
    else:
        coerced = values.reindex(index)

    return pd.to_numeric(coerced, errors="coerce").astype(float)


def zscore(series: pd.Series | None) -> pd.Series:
    """Return the population *z*-score of ``series``.

    When the input is empty, constant or contains only ``NaN`` values the
    result defaults to ``0`` to avoid spurious ``NaN`` results.
    """

    if series is None:
        return pd.Series(dtype=float)

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.empty:
        return pd.Series(dtype=float, index=series.index)

    mean = numeric.mean(skipna=True)
    std = numeric.std(ddof=0, skipna=True)

    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index, dtype=float)

    z = (numeric - mean) / std
    return z.fillna(0.0).astype(float)


def momentum_penalty(
    price: pd.Series | None,
    sma100: pd.Series | None,
    sma200: pd.Series | None,
    drawdown_1m: pd.Series | None,
) -> pd.Series:
    """Return a momentum penalty derived from moving averages and drawdown.

    The penalty is constructed so that securities trading below their moving
    averages or exhibiting a material one-month drawdown receive negative
    adjustments.  Missing data leads to a neutral (zero) penalty and divisions
    by zero are explicitly avoided.
    """

    index = None
    for candidate in (price, sma100, sma200, drawdown_1m):
        if isinstance(candidate, pd.Series):
            index = candidate.index
            break

    if index is None:
        index = pd.Index([])

    price_series = _coerce_series(price, index)
    sma100_series = _coerce_series(sma100, index)
    sma200_series = _coerce_series(sma200, index)
    drawdown_series = _coerce_series(drawdown_1m, index).fillna(0.0)

    drawdown_series = drawdown_series.clip(upper=0.0, lower=-1.0)

    def _below_ratio(avg: pd.Series) -> pd.Series:
        ratio = pd.Series(0.0, index=index, dtype=float)
        mask = avg.gt(0) & price_series.notna()
        safe_avg = avg.where(mask)
        ratio.loc[mask] = (price_series.loc[mask] - safe_avg.loc[mask]) / safe_avg.loc[mask]
        ratio = ratio.clip(upper=0.0, lower=-1.0)
        return ratio.fillna(0.0)

    below_sma100 = _below_ratio(sma100_series)
    below_sma200 = _below_ratio(sma200_series)

    penalty = 0.4 * below_sma100 + 0.4 * below_sma200 + 0.2 * drawdown_series
    return penalty.fillna(0.0).astype(float)


def score(securities: pd.DataFrame) -> pd.DataFrame:
    """Return the composite score per security.

    Parameters
    ----------
    securities:
        A dataframe that should contain ``dcf_upside``, ``multiple_score``,
        ``quality``, ``price``, ``sma100``, ``sma200`` and ``drawdown_1m``
        columns.  Missing inputs are treated as neutral (zero) contributions.
    """

    if securities is None or securities.empty:
        return pd.DataFrame({"composite_score": pd.Series(dtype=float)})

    df = securities.copy()
    index = df.index

    dcf = _coerce_series(df.get("dcf_upside"), index)
    quality = _coerce_series(df.get("quality"), index)
    multiple = _coerce_series(df.get("multiple_score"), index).fillna(0.0)

    penalty = momentum_penalty(
        df.get("price"),
        df.get("sma100"),
        df.get("sma200"),
        df.get("drawdown_1m"),
    )

    composite = 0.45 * zscore(dcf) + 0.35 * multiple + 0.20 * zscore(quality) + penalty
    composite = composite.reindex(index).fillna(0.0)

    return pd.DataFrame({"composite_score": composite.astype(float)}, index=index)


__all__ = ["momentum_penalty", "score", "zscore"]
