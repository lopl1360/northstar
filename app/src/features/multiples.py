"""Valuation multiple feature calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MULTIPLE_METRIC_ALIASES = {
    "pe": ("pe_ttm", "pe", "priceEarnings", "priceEarningsRatio"),
    "ev_ebit": ("ev_ebit", "evebit", "enterpriseValueToEbit", "evToEbit"),
    "ps": ("ps_ttm", "ps", "priceSales", "priceToSalesRatio"),
    "fcf_yield": ("fcf_yield", "freeCashFlowYield", "fcfYield"),
}


def _coerce_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.reindex(index).astype(float)


def _extract_metric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return _coerce_numeric(df[column], df.index)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _groupwise_zscore(values: pd.Series, groups: pd.Series | None) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)

    if groups is None:
        groups = pd.Series("Universe", index=values.index)
    else:
        groups = groups.reindex(values.index).fillna("Universe")

    values = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"group": groups, "value": values})

    def _compute(group: pd.DataFrame) -> pd.Series:
        valid = group["value"].dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=group.index)

        mean = valid.mean()
        std = valid.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=group.index)

        z = (group["value"] - mean) / std
        return z.fillna(0.0)

    result = frame.groupby("group", sort=False, group_keys=False).apply(_compute)
    return result.reindex(values.index).astype(float)


def compute_multiple_scores(df: pd.DataFrame) -> pd.Series:
    """Return the composite valuation multiple score per row of ``df``."""

    if df.empty:
        return pd.Series(dtype=float)

    sector = df["sector"] if "sector" in df.columns else None

    metrics = {
        name: _extract_metric(df, aliases) for name, aliases in MULTIPLE_METRIC_ALIASES.items()
    }

    z_scores: dict[str, pd.Series] = {}
    for name, series in metrics.items():
        z = _groupwise_zscore(series, sector)
        if name in {"pe", "ev_ebit", "ps"}:
            z = -z
        z_scores[name] = z

    score_frame = pd.DataFrame(z_scores)
    composite = score_frame.mean(axis=1, skipna=True)
    return composite.fillna(0.0)


__all__ = ["compute_multiple_scores"]
