"""Portfolio selection heuristics.

This module implements the lightweight portfolio selection logic used by the
pipeline.  The heuristics closely follow the specification in order to produce
two to three balanced ideas per day while remaining tolerant of missing
fundamental or technical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_MAX_PICKS = 3
MIN_TARGET_RETURN = 0.12
STOP_LOSS_LOWER = -0.18
STOP_LOSS_UPPER = -0.12
SECTOR_EXCEPTION_MARGIN = 0.4


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for the selection heuristics."""

    max_picks: int = DEFAULT_MAX_PICKS
    min_target: float = MIN_TARGET_RETURN
    stop_bounds: tuple[float, float] = (STOP_LOSS_LOWER, STOP_LOSS_UPPER)
    sector_exception_margin: float = SECTOR_EXCEPTION_MARGIN


def _coerce_float(value: object) -> float:
    """Return ``value`` coerced to ``float`` or ``NaN`` if not numeric."""

    if value is None:
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(number):
        return np.nan
    return number


def multiples_reversion_price(row: pd.Series) -> float:
    """Return the price implied by a reversion to the sector median EV/EBIT.

    The calculation is intentionally conservative: when either the current
    multiple or the sector median is missing/invalid the function falls back to
    ``NaN`` which allows the caller to rely on alternative valuation anchors.
    """

    price = _coerce_float(row.get("price"))
    current_multiple = _coerce_float(row.get("ev_ebit"))
    sector_median = _coerce_float(row.get("sector_median_ev_ebit"))

    if not np.isfinite(price) or price <= 0:
        return np.nan
    if not np.isfinite(current_multiple) or current_multiple <= 0:
        return np.nan
    if not np.isfinite(sector_median) or sector_median <= 0:
        return np.nan

    implied_price = price * (sector_median / current_multiple)
    if not np.isfinite(implied_price) or implied_price <= 0:
        return np.nan
    return float(implied_price)


def swing_low_support(row: pd.Series) -> float:
    """Return the most relevant swing-low support level available in ``row``."""

    candidates: Sequence[str] = (
        "swing_low_support",
        "recent_swing_low",
        "swing_low",
        "support_level",
        "support",
    )

    for column in candidates:
        value = row.get(column)
        support = _coerce_float(value)
        if np.isfinite(support) and support > 0:
            return support
    return np.nan


def _compute_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` enriched with target and stop calculations."""

    df = frame.copy()

    price = pd.to_numeric(df.get("price"), errors="coerce")
    dcf_upside = pd.to_numeric(df.get("dcf_upside"), errors="coerce")
    atr14 = pd.to_numeric(df.get("ATR14"), errors="coerce")

    dcf_component = price * (1 + 0.6 * dcf_upside.clip(lower=0.0, upper=0.5).fillna(0.0))

    multiples_price = df.apply(multiples_reversion_price, axis=1)
    df["multiples_reversion_price"] = multiples_price

    target = dcf_component.where(multiples_price.isna(), np.minimum(dcf_component, multiples_price))
    df["target_price"] = target

    support_levels = df.apply(swing_low_support, axis=1)

    stop_candidates = pd.DataFrame({
        "atr_stop": price - 2 * atr14,
        "support_stop": support_levels,
        "fixed_stop": price * 0.85,
    })
    stop_loss = stop_candidates.max(axis=1, skipna=True)

    df["stop_loss"] = stop_loss

    with np.errstate(divide="ignore", invalid="ignore"):
        target_return = (target - price) / price
        stop_return = (stop_loss - price) / price

    df["target_return"] = target_return.replace([np.inf, -np.inf], np.nan)
    df["stop_return"] = stop_return.replace([np.inf, -np.inf], np.nan)

    return df


def compute_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper returning ``frame`` with target and stop columns."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["target_price", "stop_loss", "target_return", "stop_return"])

    return _compute_targets(frame)


def _passes_entry_filters(row: pd.Series, config: SelectionConfig) -> bool:
    """Return ``True`` if ``row`` satisfies the entry criteria."""

    price = _coerce_float(row.get("price"))
    target_return = _coerce_float(row.get("target_return"))
    stop_return = _coerce_float(row.get("stop_return"))

    if not np.isfinite(price) or price <= 0:
        return False
    if not np.isfinite(target_return) or target_return < config.min_target:
        return False
    lower, upper = config.stop_bounds
    if not np.isfinite(stop_return) or stop_return < lower or stop_return > upper:
        return False
    return True


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` sorted by composite score, upside and risk."""

    sortable = df.copy()
    sortable["_composite_sort"] = pd.to_numeric(
        sortable.get("composite_score"), errors="coerce"
    ).fillna(-np.inf)
    sortable["_target_sort"] = pd.to_numeric(
        sortable.get("target_return"), errors="coerce"
    ).fillna(-np.inf)
    sortable["_stop_sort"] = pd.to_numeric(sortable.get("stop_return"), errors="coerce").fillna(np.inf)

    sorted_df = sortable.sort_values(
        by=["_composite_sort", "_target_sort", "_stop_sort"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return sorted_df.drop(columns=["_composite_sort", "_target_sort", "_stop_sort"])


def select(
    securities: pd.DataFrame | None,
    *,
    config: SelectionConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return the selected picks and accompanying metadata.

    The result is a tuple ``(picks, notes_json)`` where ``picks`` contains up to
    three rows of the input ``securities`` augmented with ``target_price`` and
    ``stop_loss`` columns, and ``notes_json`` documents the filters that were
    applied during selection.
    """

    if securities is None or securities.empty:
        empty = pd.DataFrame(columns=["target_price", "stop_loss", "target_return", "stop_return"])
        return empty, {
            "filters": {
                "min_target_return": MIN_TARGET_RETURN,
                "stop_bounds": [STOP_LOSS_LOWER, STOP_LOSS_UPPER],
                "max_picks": DEFAULT_MAX_PICKS,
                "sector_exception_margin": SECTOR_EXCEPTION_MARGIN,
            },
            "universe_size": 0,
            "candidates_considered": 0,
            "selected": [],
        }

    if config is None:
        config = SelectionConfig()

    enriched = _compute_targets(securities)

    filtered = enriched[enriched.apply(_passes_entry_filters, axis=1, config=config)]
    if filtered.empty:
        notes = {
            "filters": {
                "min_target_return": config.min_target,
                "stop_bounds": list(config.stop_bounds),
                "max_picks": config.max_picks,
                "sector_exception_margin": config.sector_exception_margin,
            },
            "universe_size": int(len(securities)),
            "candidates_considered": 0,
            "selected": [],
        }
        return filtered, notes

    sorted_candidates = _sort_candidates(filtered)

    selections: list = []
    sector_scores: dict[str, float] = {}
    sector_counts: dict[str, int] = {}

    for idx, row in sorted_candidates.iterrows():
        if len(selections) >= config.max_picks:
            break

        sector = str(row.get("sector") or "Unknown")
        score = _coerce_float(row.get("composite_score"))
        best_score = sector_scores.get(sector, -np.inf)
        count = sector_counts.get(sector, 0)

        if count >= 1:
            if np.isfinite(score) and score > best_score + config.sector_exception_margin:
                selections.append(idx)
                sector_counts[sector] = count + 1
                sector_scores[sector] = max(best_score, score)
            continue

        selections.append(idx)
        sector_counts[sector] = count + 1
        if np.isfinite(score):
            sector_scores[sector] = max(best_score, score)

    picks = sorted_candidates.loc[selections]

    notes_json = {
        "filters": {
            "min_target_return": config.min_target,
            "stop_bounds": list(config.stop_bounds),
            "max_picks": config.max_picks,
            "sector_exception_margin": config.sector_exception_margin,
        },
        "universe_size": int(len(securities)),
        "candidates_considered": int(len(sorted_candidates)),
        "selected": picks.get("symbol", pd.Series(dtype=str)).tolist(),
        "sector_allocation": (
            picks.get("sector", pd.Series(dtype=str)).fillna("Unknown").value_counts().to_dict()
        ),
    }

    return picks, notes_json


__all__ = [
    "SelectionConfig",
    "compute_targets",
    "multiples_reversion_price",
    "select",
    "swing_low_support",
]
