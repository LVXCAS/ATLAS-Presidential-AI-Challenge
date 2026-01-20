#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_candles(path: Path) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = [col for col in REQUIRED_COLUMNS if col not in headers]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        for row in reader:
            if not row:
                continue
            date = (row.get("date") or row.get("timestamp") or "").strip()
            if not date:
                continue
            try:
                open_v = float(row.get("open", ""))
                high_v = float(row.get("high", ""))
                low_v = float(row.get("low", ""))
                close_v = float(row.get("close", ""))
            except ValueError:
                continue
            rows.append({"date": date, "open": open_v, "high": high_v, "low": low_v, "close": close_v})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export cached CSV candles for the frontend.")
    parser.add_argument(
        "--input",
        default=str(_repo_root() / "data" / "equities" / "spy.csv"),
        help="Input CSV path (default: data/equities/spy.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(_repo_root() / "frontend" / "src" / "data" / "candles_spy.json"),
        help="Output JSON path (default: frontend/src/data/candles_spy.json).",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label for the frontend.")
    parser.add_argument("--source", default="cached_csv", help="Source label for the frontend.")
    parser.add_argument("--max-rows", type=int, default=80, help="Number of candles to keep (tail).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    candles = _read_candles(input_path)
    if not candles:
        raise SystemExit("No candles found in input file.")

    if args.max_rows and len(candles) > args.max_rows:
        candles = candles[-args.max_rows :]

    payload = {
        "symbol": args.symbol,
        "source": args.source,
        "candles": candles,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path} ({len(candles)} candles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
