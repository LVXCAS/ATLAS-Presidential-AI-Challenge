#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip().lower()
    if not v:
        return True
    if v in {"***", "****", "*****"}:
        return True
    tokens = ("your_", "your-", "changeme", "replace", "example", "password_here", "token_here", "key_here")
    return any(token in v for token in tokens)


def _load_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in env:
            if _is_placeholder(env.get(key)) and not _is_placeholder(value):
                env[key] = value
        else:
            env[key] = value
    return env


def _get_env(name: str, env: Dict[str, str]) -> Optional[str]:
    return os.environ.get(name) or env.get(name)


def _run(cmd: List[str]) -> int:
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh demo data across providers.")
    parser.add_argument("--providers", nargs="*", default=None, help="Subset of providers to use.")
    parser.add_argument("--equities", nargs="*", default=None, help="Equity symbols to fetch.")
    parser.add_argument("--fx", nargs="*", default=None, help="FX pairs to fetch.")
    parser.add_argument("--macro", nargs="*", default=None, help="FRED series IDs to fetch.")
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--sleep", type=float, default=12.0, help="Seconds to sleep between calls.")
    parser.add_argument("--enable-live", action="store_true", help="Required to allow API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only.")
    args = parser.parse_args()

    root = _repo_root()
    dotenv = _load_dotenv(root / ".env")
    live_enabled = args.enable_live or _truthy(_get_env("USE_LIVE_DATA", dotenv))
    if not live_enabled:
        print(
            "Live data access is disabled. Re-run with --enable-live or set USE_LIVE_DATA=true in .env.",
            file=sys.stderr,
        )
        return 1

    providers = args.providers or ["alpha_vantage", "polygon", "alpaca", "fred"]
    equities = args.equities or ["SPY", "AAPL", "MSFT"]
    fx_pairs = args.fx or ["EURUSD", "GBPUSD"]
    macro_series = args.macro or ["DGS10", "CPIAUCSL"]

    cache_script = root / "scripts" / "cache_data.py"
    base_cmd = [sys.executable, str(cache_script)]
    if args.enable_live:
        base_cmd.append("--enable-live")
    if args.start_date:
        base_cmd.extend(["--start-date", args.start_date])
    if args.end_date:
        base_cmd.extend(["--end-date", args.end_date])
    if args.max_rows:
        base_cmd.extend(["--max-rows", str(args.max_rows)])
    if args.sleep is not None:
        base_cmd.extend(["--sleep", str(args.sleep)])
    if args.dry_run:
        base_cmd.append("--dry-run")

    status = 0
    if "alpha_vantage" in providers and _get_env("ALPHA_VANTAGE_API_KEY", dotenv):
        status |= _run(base_cmd + ["--provider", "alpha_vantage", "--asset-class", "equities", "--symbols", *equities])
        status |= _run(base_cmd + ["--provider", "alpha_vantage", "--asset-class", "fx", "--symbols", *fx_pairs])
    if "polygon" in providers and _get_env("POLYGON_API_KEY", dotenv):
        status |= _run(base_cmd + ["--provider", "polygon", "--asset-class", "equities", "--symbols", *equities])
        status |= _run(base_cmd + ["--provider", "polygon", "--asset-class", "fx", "--symbols", *fx_pairs])
    if "alpaca" in providers and _get_env("ALPACA_API_KEY", dotenv) and _get_env("ALPACA_SECRET_KEY", dotenv):
        status |= _run(base_cmd + ["--provider", "alpaca", "--asset-class", "equities", "--symbols", *equities])
    if "fred" in providers and _get_env("FRED_API_KEY", dotenv):
        status |= _run(base_cmd + ["--provider", "fred", "--asset-class", "macro", "--symbols", *macro_series])

    if status != 0:
        print("One or more provider refreshes failed. See output above.", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
