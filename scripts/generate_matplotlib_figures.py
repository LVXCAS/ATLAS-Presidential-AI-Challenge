#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_dt(value: str) -> datetime:
    v = str(value).strip()
    if not v:
        raise ValueError("Empty datetime")
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return datetime.fromisoformat(v[:10])


def _read_candles(path: Path) -> Tuple[List[datetime], List[float], List[float], List[float]]:
    dates: List[datetime] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            date = row.get("date") or row.get("timestamp")
            if not date:
                continue
            try:
                dates.append(_parse_dt(date))
                highs.append(float(row.get("high", 0)))
                lows.append(float(row.get("low", 0)))
                closes.append(float(row.get("close", 0)))
            except ValueError:
                continue
    return dates, highs, lows, closes


def _read_risk_scores(path: Path) -> Tuple[List[datetime], List[float], List[bool]]:
    payload = json.loads(path.read_text())
    windows = payload.get("windows") or []
    if not windows:
        return [], [], []
    steps = windows[0].get("steps") or []

    dates: List[datetime] = []
    scores: List[float] = []
    stress_flags: List[bool] = []
    for step in steps:
        time = step.get("time")
        if not time:
            continue
        quant = step.get("quant_team") or {}
        score = quant.get("aggregated_score")
        if score is None:
            continue
        try:
            dates.append(_parse_dt(time))
            scores.append(float(score))
            stress_flags.append(bool(step.get("is_stress")))
        except ValueError:
            continue
    return dates, scores, stress_flags


def _compress_stress_ranges(dates: List[datetime], flags: List[bool]) -> List[Tuple[datetime, datetime]]:
    ranges: List[Tuple[datetime, datetime]] = []
    start = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = dates[idx]
        if not flag and start is not None:
            ranges.append((start, dates[idx - 1]))
            start = None
    if start is not None:
        ranges.append((start, dates[-1]))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matplotlib figures for the frontend.")
    parser.add_argument(
        "--price-csv",
        default=str(_repo_root() / "data" / "equities" / "spy.csv"),
        help="Input OHLCV CSV (default: data/equities/spy.csv).",
    )
    parser.add_argument(
        "--eval-json",
        default=str(_repo_root() / "submission" / "evaluation_results.json"),
        help="Input evaluation JSON (default: submission/evaluation_results.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "frontend" / "public" / "figures"),
        help="Output directory for PNGs (default: frontend/public/figures).",
    )
    parser.add_argument("--max-rows", type=int, default=180, help="Max candles for plots.")
    args = parser.parse_args()

    root = _repo_root()
    mpl_cache = root / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    price_csv = Path(args.price_csv)
    eval_json = Path(args.eval_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates, highs, lows, closes = _read_candles(price_csv)
    if args.max_rows and len(dates) > args.max_rows:
        dates = dates[-args.max_rows :]
        highs = highs[-args.max_rows :]
        lows = lows[-args.max_rows :]
        closes = closes[-args.max_rows :]

    if dates:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.fill_between(dates, lows, highs, color="#c15f2b", alpha=0.12, label="Daily range")
        ax.plot(dates, closes, color="#3b6b5b", linewidth=1.6, label="Close")
        ax.set_title("SPY closing price (cached)")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "spy_close.png", dpi=160)
        plt.close(fig)

    if eval_json.exists():
        risk_dates, scores, stress_flags = _read_risk_scores(eval_json)
        if risk_dates:
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.plot(risk_dates, scores, color="#1a1a1a", linewidth=1.4, label="Risk score")
            ax.axhline(0.25, color="#3b6b5b", linestyle="--", linewidth=1.0, label="GREENLIGHT max")
            ax.axhline(0.36, color="#c15f2b", linestyle="--", linewidth=1.0, label="STAND_DOWN min")

            for start, end in _compress_stress_ranges(risk_dates, stress_flags):
                ax.axvspan(start, end, color="#c15f2b", alpha=0.08)

            ax.set_title("ATLAS risk score timeline (cached)")
            ax.set_ylabel("Risk score (0-1)")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "risk_scores.png", dpi=160)
            plt.close(fig)

    print(f"Wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
