#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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


def _read_candles(path: Path) -> Tuple[List[datetime], List[float], List[float], List[float], List[float]]:
    dates: List[datetime] = []
    opens: List[float] = []
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
                opens.append(float(row.get("open", 0)))
                highs.append(float(row.get("high", 0)))
                lows.append(float(row.get("low", 0)))
                closes.append(float(row.get("close", 0)))
            except ValueError:
                continue
    return dates, opens, highs, lows, closes


def _read_risk_scores(path: Path) -> Tuple[List[datetime], List[float], List[bool], List[str]]:
    payload = json.loads(path.read_text())
    windows = payload.get("windows") or []
    if not windows:
        return [], [], [], []
    steps = windows[0].get("steps") or []

    dates: List[datetime] = []
    scores: List[float] = []
    stress_flags: List[bool] = []
    labels: List[str] = []
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
            labels.append(str(quant.get("label", "")))
        except ValueError:
            continue
    return dates, scores, stress_flags, labels


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


def _compress_label_ranges(
    dates: List[datetime], labels: List[str]
) -> List[Tuple[str, datetime, datetime]]:
    ranges: List[Tuple[str, datetime, datetime]] = []
    if not dates or not labels:
        return ranges
    current = labels[0]
    start = dates[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            ranges.append((current, start, dates[idx - 1]))
            current = label
            start = dates[idx]
    ranges.append((current, start, dates[-1]))
    return ranges


def _read_agent_scores(path: Path) -> Tuple[List[datetime], Dict[str, List[float]]]:
    payload = json.loads(path.read_text())
    windows = payload.get("windows") or []
    if not windows:
        return [], {}
    steps = windows[0].get("steps") or []

    dates: List[datetime] = []
    agent_scores: Dict[str, List[float]] = {}
    nan = float("nan")

    for step in steps:
        time = step.get("time")
        if not time:
            continue
        quant = step.get("quant_team") or {}
        agents = quant.get("agents") or {}

        try:
            dates.append(_parse_dt(time))
        except ValueError:
            continue

        # Extend existing series with NaN for this step
        for name in agent_scores:
            agent_scores[name].append(nan)

        idx = len(dates) - 1
        for name, meta in agents.items():
            if name not in agent_scores:
                agent_scores[name] = [nan] * idx
                agent_scores[name].append(nan)
            val = meta.get("score")
            agent_scores[name][idx] = float(val) if val is not None else nan

    return dates, agent_scores


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
    from matplotlib.patches import Patch, Rectangle

    price_csv = Path(args.price_csv)
    eval_json = Path(args.eval_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates, opens, highs, lows, closes = _read_candles(price_csv)
    if args.max_rows and len(dates) > args.max_rows:
        dates = dates[-args.max_rows :]
        opens = opens[-args.max_rows :]
        highs = highs[-args.max_rows :]
        lows = lows[-args.max_rows :]
        closes = closes[-args.max_rows :]

    if dates:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = mdates.date2num(dates)
        width = 0.6
        up = "#3b6b5b"
        down = "#c15f2b"

        for idx, x_i in enumerate(x):
            open_v = opens[idx]
            close_v = closes[idx]
            high_v = highs[idx]
            low_v = lows[idx]
            color = up if close_v >= open_v else down

            ax.vlines(x_i, low_v, high_v, color=color, linewidth=1.0)
            rect_y = min(open_v, close_v)
            rect_h = abs(close_v - open_v)
            if rect_h == 0:
                rect_h = max(0.02, 0.001 * max(1.0, open_v))
            ax.add_patch(
                Rectangle(
                    (x_i - width / 2, rect_y),
                    width,
                    rect_h,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.85,
                )
            )

        ax.set_title("SPY candlestick chart (cached)")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        ax.legend(
            handles=[
                Patch(facecolor=up, edgecolor=up, label="Up day"),
                Patch(facecolor=down, edgecolor=down, label="Down day"),
            ],
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(output_dir / "spy_close.png", dpi=160)
        plt.close(fig)

    if eval_json.exists():
        risk_dates, scores, stress_flags, labels = _read_risk_scores(eval_json)
        if risk_dates:
            fig, (ax, ax2) = plt.subplots(
                2,
                1,
                figsize=(9, 4.6),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            ax.plot(risk_dates, scores, color="#1a1a1a", linewidth=1.4, label="Risk score")
            ax.axhline(0.25, color="#3b6b5b", linestyle="--", linewidth=1.0, label="GREENLIGHT max")
            ax.axhline(0.36, color="#c15f2b", linestyle="--", linewidth=1.0, label="STAND_DOWN min")

            for start, end in _compress_stress_ranges(risk_dates, stress_flags):
                ax.axvspan(start, end, color="#c15f2b", alpha=0.08)

            ax.set_title("ATLAS risk score timeline (cached)")
            ax.set_ylabel("Risk score (0-1)")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(frameon=False, fontsize=8, loc="upper left")

            posture_colors = {
                "GREENLIGHT": "#3b6b5b",
                "WATCH": "#d49a6a",
                "STAND_DOWN": "#c15f2b",
            }
            for label, start, end in _compress_label_ranges(risk_dates, labels):
                color = posture_colors.get(label, "#999999")
                ax2.axvspan(start, end, color=color, alpha=0.3)

            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            ax2.set_ylabel("Posture")
            ax2.grid(False)
            ax2.legend(
                handles=[
                    Patch(facecolor=posture_colors["GREENLIGHT"], edgecolor=posture_colors["GREENLIGHT"], label="GREENLIGHT"),
                    Patch(facecolor=posture_colors["WATCH"], edgecolor=posture_colors["WATCH"], label="WATCH"),
                    Patch(facecolor=posture_colors["STAND_DOWN"], edgecolor=posture_colors["STAND_DOWN"], label="STAND_DOWN"),
                ],
                frameon=False,
                fontsize=7,
                loc="upper left",
                ncol=3,
            )

            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(output_dir / "risk_scores.png", dpi=160)
            plt.close(fig)

        agent_dates, agent_scores = _read_agent_scores(eval_json)
        if agent_dates and agent_scores:
            # Pick top agents by score variance (most dynamic).
            variability: List[Tuple[str, float]] = []
            for name, series in agent_scores.items():
                values = [v for v in series if not math.isnan(v)]
                if len(values) < 2:
                    continue
                avg = sum(values) / len(values)
                var = sum((v - avg) ** 2 for v in values) / len(values)
                variability.append((name, var))
            top = [name for name, _ in sorted(variability, key=lambda x: x[1], reverse=True)[:5]]

            if top:
                fig, ax = plt.subplots(figsize=(9, 3.8))
                palette = ["#3b6b5b", "#c15f2b", "#5c6f91", "#d2a650", "#7a5c8e"]

                for idx, name in enumerate(top):
                    ax.plot(
                        agent_dates,
                        agent_scores[name],
                        linewidth=1.2,
                        color=palette[idx % len(palette)],
                        label=name,
                    )

                ax.set_title("Top agent risk lenses (cached)")
                ax.set_ylabel("Agent score (0-1)")
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle="--", alpha=0.25)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                fig.autofmt_xdate()
                ax.legend(frameon=False, fontsize=7, ncol=2)
                fig.tight_layout()
                fig.savefig(output_dir / "agent_scores.png", dpi=160)
                plt.close(fig)

    print(f"Wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
