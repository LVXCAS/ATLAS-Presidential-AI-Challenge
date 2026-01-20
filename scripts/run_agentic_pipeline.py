#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ATLAS agentic pipeline end-to-end.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached data first.")
    parser.add_argument("--providers", nargs="*", default=None, help="Providers to refresh.")
    parser.add_argument("--enable-live", action="store_true", help="Required for refresh.")
    parser.add_argument("--asset-class", default="equities", choices=["equities", "fx"])
    parser.add_argument("--symbol", default="SPY", help="Symbol for cached demo/eval.")
    parser.add_argument("--data-dir", default=None, help="Root data directory (default: repo/data).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on cached rows.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip quant_team_eval.py.")
    parser.add_argument("--skip-demo", action="store_true", help="Skip quant_team_demo.py.")
    args = parser.parse_args()

    root = _repo_root()
    data_dir = args.data_dir or str(root / "data")

    status = 0
    if args.refresh:
        refresh_script = root / "scripts" / "refresh_demo_data.py"
        cmd = [sys.executable, str(refresh_script)]
        if args.providers:
            cmd.extend(["--providers", *args.providers])
        if args.enable_live:
            cmd.append("--enable-live")
        status |= _run(cmd)

    if not args.skip_eval:
        eval_script = root / "Agents" / "ATLAS_HYBRID" / "quant_team_eval.py"
        cmd = [
            sys.executable,
            str(eval_script),
            "--data-source",
            "cached",
            "--asset-class",
            args.asset_class,
            "--symbol",
            args.symbol,
            "--data-dir",
            data_dir,
        ]
        if args.max_rows:
            cmd.extend(["--max-rows", str(args.max_rows)])
        status |= _run(cmd)

    if not args.skip_demo:
        demo_script = root / "Agents" / "ATLAS_HYBRID" / "quant_team_demo.py"
        cmd = [
            sys.executable,
            str(demo_script),
            "--data-source",
            "cached",
            "--asset-class",
            args.asset_class,
            "--symbol",
            args.symbol,
            "--data-dir",
            data_dir,
        ]
        if args.max_rows:
            cmd.extend(["--max-rows", str(args.max_rows)])
        status |= _run(cmd)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
