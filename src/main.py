from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from agents import assess_agents
from aggregator import aggregate
from explain import build_explanation


def load_csv(path: Path) -> Tuple[List[float], List[float]]:
    prices: List[float] = []
    volumes: List[float] = []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                price = float(row.get("close", ""))
                volume = float(row.get("volume", "0") or 0.0)
            except ValueError:
                continue
            prices.append(price)
            volumes.append(volume)

    return prices, volumes


def synthetic_scenario(name: str) -> Tuple[List[float], List[float]]:
    scenarios = {
        "calm": [
            100.0, 100.05, 100.1, 100.08, 100.12,
            100.15, 100.18, 100.2, 100.22, 100.25,
            100.28, 100.3, 100.32, 100.35, 100.38,
            100.4, 100.42, 100.45, 100.47, 100.5,
        ],
        "transition": [
            100.0, 100.2, 100.35, 100.5, 100.7,
            100.85, 101.0, 100.9, 100.8, 100.7,
            100.6, 100.4, 100.2, 100.05, 99.9,
            99.8, 99.75, 99.7, 99.65, 99.6,
        ],
        "crisis": [
            100.0, 102.0, 98.0, 103.0, 97.0,
            104.0, 95.0, 101.0, 94.0, 100.0,
            96.0, 105.0, 92.0, 107.0, 90.0,
            99.0, 93.0, 108.0, 91.0, 97.0,
        ],
    }
    prices = scenarios.get(name, [])
    volumes = [1000000.0 for _ in prices]
    return prices, volumes


def run_atlas(prices: List[float], volumes: List[float]) -> dict:
    signals = assess_agents(prices, volumes)
    result = aggregate(signals)
    explanation = build_explanation(
        result["posture"],
        result.get("risk_flags", []),
        result.get("agent_signals", {}),
    )
    result["explanation"] = explanation
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS Codex execution demo (agentic reasoning).")
    parser.add_argument("--input", help="Path to scenario CSV (date,open,high,low,close,volume).")
    parser.add_argument("--scenario", choices=["calm", "transition", "crisis"], help="Use a synthetic scenario.")
    args = parser.parse_args()

    if args.scenario:
        prices, volumes = synthetic_scenario(args.scenario)
    else:
        if not args.input:
            parser.error("Provide --input or --scenario.")
        prices, volumes = load_csv(Path(args.input))

    if not prices:
        raise SystemExit("No usable price data found.")

    result = run_atlas(prices, volumes)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
