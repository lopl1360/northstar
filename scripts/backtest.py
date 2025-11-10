#!/usr/bin/env python
"""Backtest harness for historical valuation snapshots.

This script loads archived ``valuations`` and ``daily_metrics`` snapshots,
selects the top ideas for each day, and evaluates how the picks would have
performed under different target multipliers and stop thresholds.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

DATE_PATTERNS: Sequence[str] = ("%Y%m%d", "%Y-%m-%d", "%Y_%m_%d")
DEFAULT_MULTIPLIERS: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7)
DEFAULT_STOP_LEVELS: Mapping[str, float] = {"12%": 0.12, "15%": 0.15, "18%": 0.18}


@dataclass
class Snapshot:
    """A normalized snapshot of a daily export."""

    path: Path
    frame: pd.DataFrame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the daily valuation snapshots")
    parser.add_argument(
        "--valuations-path",
        type=Path,
        default=Path("valuations"),
        help="Directory (or file) containing valuation snapshots.",
    )
    parser.add_argument(
        "--daily-metrics-path",
        type=Path,
        default=Path("daily_metrics"),
        help="Directory (or file) containing daily metrics snapshots.",
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=3,
        help="Maximum number of picks per day to evaluate.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Evaluation window in trading days (default: 20).",
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="*",
        default=list(DEFAULT_MULTIPLIERS),
        help="Target multipliers to test (default: 0.5 0.55 0.6 0.65 0.7).",
    )
    parser.add_argument(
        "--stop-band",
        type=float,
        default=0.15,
        help="Stop-loss percentage used to simulate exits (default: 0.15).",
    )
    return parser.parse_args(argv)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def _parse_date_from_string(value: str) -> date | None:
    stripped = value.strip()
    for pattern in DATE_PATTERNS:
        try:
            return datetime.strptime(stripped, pattern).date()
        except ValueError:
            continue
    digits = "".join(ch for ch in stripped if ch.isdigit())
    if len(digits) >= 8:
        for pattern in DATE_PATTERNS:
            try:
                return datetime.strptime(digits[:8], pattern).date()
            except ValueError:
                continue
    return None


def _iter_snapshot_files(base: Path, dataset_name: str) -> Iterator[Snapshot]:
    if base.is_file():
        frame = _read_table(base)
        yield Snapshot(path=base, frame=frame)
        return

    direct_files = sorted(p for p in base.iterdir() if p.suffix in {".csv", ".parquet", ".pq"})
    if direct_files:
        for path in direct_files:
            yield Snapshot(path=path, frame=_read_table(path))
        return

    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        candidate = None
        specific = child / f"{dataset_name}.csv"
        if specific.exists():
            candidate = specific
        else:
            matches = list(child.glob(f"{dataset_name}.*"))
            if matches:
                candidate = matches[0]
        if candidate and candidate.exists():
            yield Snapshot(path=candidate, frame=_read_table(candidate))


def _normalise_snapshot(snapshot: Snapshot, dataset_name: str) -> pd.DataFrame:
    frame = snapshot.frame.copy()
    if "symbol" not in frame.columns:
        raise ValueError(f"Missing 'symbol' column in {snapshot.path}")

    if "ds" in frame.columns:
        frame["ds"] = pd.to_datetime(frame["ds"]).dt.date
    else:
        inferred = None
        for candidate in (snapshot.path.stem, snapshot.path.parent.name):
            inferred = _parse_date_from_string(candidate)
            if inferred:
                break
        if not inferred:
            raise ValueError(f"Unable to infer snapshot date for {snapshot.path}")
        frame["ds"] = inferred

    frame["symbol"] = frame["symbol"].astype(str)
    frame.columns = [col.strip() for col in frame.columns]
    frame["dataset"] = dataset_name
    return frame


def load_history(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    frames: list[pd.DataFrame] = []
    for snapshot in _iter_snapshot_files(path, dataset_name):
        frames.append(_normalise_snapshot(snapshot, dataset_name))

    if not frames:
        raise ValueError(f"No snapshot files found in {path}")

    combined = pd.concat(frames, ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"]).dt.date
    combined = combined.drop_duplicates(subset=["symbol", "ds"], keep="last")
    return combined


def compute_target_price(row: Mapping[str, object], multiplier: float) -> float:
    price = float(row.get("price", float("nan")))
    dcf_upside = row.get("dcf_upside")
    multiples_price = row.get("multiples_reversion_price")

    price = float(price) if math.isfinite(price) else float("nan")
    dcf = float(dcf_upside) if pd.notna(dcf_upside) else float("nan")
    multiples = float(multiples_price) if pd.notna(multiples_price) else float("nan")

    dcf_component = float("nan")
    if math.isfinite(price) and price > 0 and math.isfinite(dcf):
        clipped = min(max(dcf, 0.0), 0.5)
        dcf_component = price * (1.0 + multiplier * clipped)

    if math.isfinite(dcf_component) and math.isfinite(multiples) and multiples > 0:
        return float(min(dcf_component, multiples))
    if math.isfinite(multiples) and multiples > 0:
        return float(multiples)
    return float(dcf_component)


def _quantiles(values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return [float("nan") for _ in quantiles]
    return [float(np.quantile(arr, q)) for q in quantiles]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    valuations = load_history(args.valuations_path, "valuations")
    metrics = load_history(args.daily_metrics_path, "daily_metrics")

    valuations = valuations.rename(columns={"fv": "fv_dcf"})
    if "multiples_reversion_price" not in valuations.columns and "multiples_reversion_price" in metrics.columns:
        valuations = valuations.merge(
            metrics[["symbol", "ds", "multiples_reversion_price"]],
            on=["symbol", "ds"],
            how="left",
            suffixes=("", "_metrics"),
        )

    merged = valuations.merge(
        metrics[[
            "symbol",
            "ds",
            "price",
        ]].drop_duplicates(subset=["symbol", "ds"]),
        on=["symbol", "ds"],
        how="inner",
        suffixes=("", "_metrics"),
    )

    merged["price"] = pd.to_numeric(merged.get("price"), errors="coerce")
    merged["composite_score"] = pd.to_numeric(merged.get("composite_score"), errors="coerce")
    merged["dcf_upside"] = pd.to_numeric(merged.get("dcf_upside"), errors="coerce")

    merged = merged[merged["price"].notna() & (merged["price"] > 0)]
    merged = merged[merged["composite_score"].notna()]

    all_metrics = metrics[["symbol", "ds", "price"]].copy()
    all_metrics["price"] = pd.to_numeric(all_metrics["price"], errors="coerce")
    all_metrics = all_metrics.dropna(subset=["price"])
    all_metrics.sort_values(["symbol", "ds"], inplace=True)

    grouped_metrics: dict[str, pd.DataFrame] = {}
    for symbol, group in all_metrics.groupby("symbol"):
        grouped_metrics[symbol] = group.reset_index(drop=True)

    unique_dates = sorted(set(merged["ds"]))
    total_days = 0
    picks_processed = 0

    mae_values: list[float] = []
    mfe_values: list[float] = []
    ret10_values: list[float] = []
    ret20_values: list[float] = []
    stop_band_hits = {name: 0 for name in DEFAULT_STOP_LEVELS}

    multiplier_stats: dict[float, dict[str, list | int | float]] = {}
    for multiplier in args.multipliers:
        multiplier_stats[multiplier] = {
            "count": 0,
            "target_hits": 0,
            "stop_hits": 0,
            "holds": 0,
            "realized_returns": [],
            "target_returns": [],
            "days_to_target": [],
        }

    stop_mid = float(args.stop_band)
    stop_mid = abs(stop_mid)

    for current_date in unique_dates:
        day_universe = merged[merged["ds"] == current_date]
        if day_universe.empty:
            continue
        total_days += 1

        sorted_day = day_universe.sort_values("composite_score", ascending=False)
        picks = sorted_day.head(args.max_picks)
        if picks.empty:
            continue

        processed_for_day = 0

        for _, pick in picks.iterrows():
            symbol = pick["symbol"]
            entry_price = float(pick.get("price", float("nan")))
            if not math.isfinite(entry_price) or entry_price <= 0:
                continue

            history = grouped_metrics.get(symbol)
            if history is None:
                continue

            mask = history["ds"] == current_date
            if not mask.any():
                continue
            start_idx = int(np.where(mask)[0][0])
            window = int(args.window)
            history_slice = history.iloc[start_idx : start_idx + window + 1]
            if history_slice.empty:
                continue

            processed_for_day += 1
            prices = history_slice["price"].to_numpy(dtype=float)
            returns = (prices - entry_price) / entry_price

            mae = float(np.nanmin(returns))
            mfe = float(np.nanmax(returns))
            mae_values.append(mae)
            mfe_values.append(mfe)

            if len(returns) > 10:
                ret10_values.append(float(returns[10]))
            else:
                ret10_values.append(float("nan"))
            if len(returns) > 20:
                ret20_values.append(float(returns[20]))
            else:
                ret20_values.append(float("nan"))

            for label, threshold in DEFAULT_STOP_LEVELS.items():
                if mae <= -abs(threshold):
                    stop_band_hits[label] += 1

            for multiplier in args.multipliers:
                stats = multiplier_stats[multiplier]
                target_price = compute_target_price(pick, multiplier)
                if not math.isfinite(target_price) or target_price <= 0:
                    continue

                target_return = (target_price - entry_price) / entry_price
                stats["count"] += 1
                stats["target_returns"].append(target_return)

                target_hit_day = None
                stop_hit_day = None
                stop_price = entry_price * (1 - stop_mid)

                for idx in range(1, len(prices)):
                    price = prices[idx]
                    if math.isfinite(price):
                        if target_hit_day is None and price >= target_price:
                            target_hit_day = idx
                        if stop_hit_day is None and price <= stop_price:
                            stop_hit_day = idx
                    if target_hit_day is not None and stop_hit_day is not None:
                        break

                if target_hit_day is not None and (
                    stop_hit_day is None or target_hit_day <= stop_hit_day
                ):
                    stats["target_hits"] += 1
                    stats["realized_returns"].append(target_return)
                    stats["days_to_target"].append(target_hit_day)
                elif stop_hit_day is not None:
                    stats["stop_hits"] += 1
                    stats["realized_returns"].append(-stop_mid)
                    stats["days_to_target"].append(stop_hit_day)
                else:
                    stats["holds"] += 1
                    final_return = float(returns[-1])
                    stats["realized_returns"].append(final_return)
                    stats["days_to_target"].append(len(prices) - 1)

        picks_processed += processed_for_day

    if picks_processed == 0:
        print("No picks were generated from the provided snapshots.")
        return 0

    mae_clean = [v for v in mae_values if math.isfinite(v)]
    mfe_clean = [v for v in mfe_values if math.isfinite(v)]
    ret10_clean = [v for v in ret10_values if math.isfinite(v)]
    ret20_clean = [v for v in ret20_values if math.isfinite(v)]

    print("=== Backtest Summary ===")
    print(f"Days processed: {total_days}")
    print(f"Total picks: {picks_processed} (avg {picks_processed / max(total_days, 1):.2f} per day)")

    if ret10_clean:
        print(
            f"10-day returns: mean {statistics.mean(ret10_clean):.3f}, median {statistics.median(ret10_clean):.3f}, n={len(ret10_clean)}"
        )
    else:
        print("10-day returns: insufficient data")

    if ret20_clean:
        print(
            f"20-day returns: mean {statistics.mean(ret20_clean):.3f}, median {statistics.median(ret20_clean):.3f}, n={len(ret20_clean)}"
        )
    else:
        print("20-day returns: insufficient data")

    if mae_clean:
        q25, q50, q75 = _quantiles(mae_clean, [0.25, 0.5, 0.75])
        print(
            "Max adverse excursion quantiles: "
            f"25% {q25:.3f}, median {q50:.3f}, 75% {q75:.3f}"
        )
    else:
        print("Max adverse excursion: insufficient data")

    for label, threshold in DEFAULT_STOP_LEVELS.items():
        rate = stop_band_hits[label] / picks_processed
        print(f"MAE <= -{label}: {rate:.1%}")

    print("\nTarget multiplier evaluation (stop {:.0%}):".format(stop_mid))
    best_multiplier = None
    best_expectation = -float("inf")

    for multiplier in sorted(multiplier_stats):
        stats = multiplier_stats[multiplier]
        count = stats["count"]
        if count == 0:
            print(f"  {multiplier:.2f}: no valid targets")
            continue
        realized = [r for r in stats["realized_returns"] if math.isfinite(r)]
        expectation = statistics.mean(realized) if realized else float("nan")
        hit_rate = stats["target_hits"] / count
        stop_rate = stats["stop_hits"] / count
        hold_rate = stats["holds"] / count
        median_days = (
            statistics.median(stats["days_to_target"])
            if stats["days_to_target"]
            else float("nan")
        )
        avg_target = (
            statistics.mean(stats["target_returns"])
            if stats["target_returns"]
            else float("nan")
        )
        print(
            f"  {multiplier:.2f}: hit {hit_rate:.1%}, stop {stop_rate:.1%}, hold {hold_rate:.1%}, "
            f"avg target {avg_target:.3f}, expected {expectation:.3f}, median days {median_days:.1f}"
        )
        if math.isfinite(expectation) and expectation > best_expectation:
            best_expectation = expectation
            best_multiplier = multiplier

    if best_multiplier is not None:
        print(
            f"\nSuggested target multiplier: {best_multiplier:.2f} "
            f"(highest expected return {best_expectation:.3f})"
        )
    else:
        print("\nUnable to suggest a target multiplier (insufficient data).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
