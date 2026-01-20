#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
POLYGON_BASE = "https://api.polygon.io"
ALPACA_DATA_BASE_DEFAULT = "https://data.alpaca.markets"
FRED_BASE = "https://api.stlouisfed.org/fred"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_symbol(symbol: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (symbol or "")).strip("_")


def _parse_fx_pair(symbol: str) -> Tuple[str, str]:
    compact = "".join(ch for ch in symbol if ch.isalnum()).upper()
    if len(compact) != 6:
        raise ValueError(f"FX symbol must look like EURUSD or EUR/USD, got: {symbol}")
    return compact[:3], compact[3:]


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


def _parse_date(value: str) -> Optional[date]:
    if not value:
        return None
    value = str(value).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        pass
    if "T" in value:
        value = value.split("T", 1)[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _default_date_range(days: int = 365) -> Tuple[date, date]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    return start, end


def _ensure_date_range(
    start_date: Optional[date],
    end_date: Optional[date],
    days: int = 365,
) -> Tuple[date, date]:
    default_start, default_end = _default_date_range(days)
    return start_date or default_start, end_date or default_end


def _timestamp_to_date(value) -> Optional[str]:
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e14:
            ts = ts / 1e9
        elif ts > 1e11:
            ts = ts / 1e3
        return datetime.utcfromtimestamp(ts).date().isoformat()
    return None


def _coerce_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_url(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _decode_alpha_vantage_csv(raw: str) -> List[Dict[str, str]]:
    if raw.lstrip().startswith("{"):
        payload = json.loads(raw)
        message = payload.get("Note") or payload.get("Error Message") or payload.get("Information")
        raise RuntimeError(message or "Alpha Vantage returned JSON instead of CSV.")
    reader = csv.DictReader(io.StringIO(raw))
    return [row for row in reader if row]


def _normalize_rows(
    rows: Iterable[Dict[str, str]],
    asset_class: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for row in rows:
        raw_date = row.get("timestamp") or row.get("date")
        parsed_date = _parse_date(raw_date)
        if not parsed_date:
            continue
        if start_date and parsed_date < start_date:
            continue
        if end_date and parsed_date > end_date:
            continue

        open_v = _coerce_float(row.get("open"))
        high_v = _coerce_float(row.get("high"))
        low_v = _coerce_float(row.get("low"))
        close_v = _coerce_float(row.get("close"))
        if None in (open_v, high_v, low_v, close_v):
            continue

        volume_value = row.get("volume") if asset_class == "equities" else "0"
        volume_v = _coerce_float(volume_value)
        if volume_v is None:
            volume_v = 0.0

        cleaned.append(
            {
                "date": parsed_date.isoformat(),
                "open": f"{open_v:.6f}",
                "high": f"{high_v:.6f}",
                "low": f"{low_v:.6f}",
                "close": f"{close_v:.6f}",
                "volume": f"{volume_v:.6f}",
            }
        )
    cleaned.sort(key=lambda row: row["date"])
    return cleaned


def _apply_max_rows(rows: List[Dict[str, str]], max_rows: Optional[int]) -> List[Dict[str, str]]:
    if max_rows and max_rows > 0 and len(rows) > max_rows:
        return rows[-max_rows:]
    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _alpha_vantage_equity_url(symbol: str, api_key: str, outputsize: str) -> str:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
        "apikey": api_key,
    }
    return f"{ALPHA_VANTAGE_BASE}?{urllib.parse.urlencode(params)}"


def _alpha_vantage_fx_url(pair: Tuple[str, str], api_key: str, outputsize: str) -> str:
    params = {
        "function": "FX_DAILY",
        "from_symbol": pair[0],
        "to_symbol": pair[1],
        "outputsize": outputsize,
        "datatype": "csv",
        "apikey": api_key,
    }
    return f"{ALPHA_VANTAGE_BASE}?{urllib.parse.urlencode(params)}"


def _download_alpha_vantage(
    symbol: str,
    asset_class: str,
    api_key: str,
    outputsize: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_rows: Optional[int],
) -> List[Dict[str, str]]:
    if asset_class == "fx":
        pair = _parse_fx_pair(symbol)
        url = _alpha_vantage_fx_url(pair, api_key, outputsize)
    else:
        url = _alpha_vantage_equity_url(symbol, api_key, outputsize)

    raw = _fetch_url(url)
    rows = _decode_alpha_vantage_csv(raw)
    normalized = _normalize_rows(rows, asset_class, start_date, end_date)
    return _apply_max_rows(normalized, max_rows)


def _download_polygon(
    symbol: str,
    asset_class: str,
    api_key: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_rows: Optional[int],
) -> List[Dict[str, str]]:
    if asset_class not in {"equities", "fx"}:
        raise ValueError("Polygon supports equities and fx only.")

    start_date, end_date = _ensure_date_range(start_date, end_date)

    if asset_class == "fx":
        pair = _parse_fx_pair(symbol)
        ticker = f"C:{pair[0]}{pair[1]}"
    else:
        ticker = symbol.upper()

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date.isoformat()}/{end_date.isoformat()}?{urllib.parse.urlencode(params)}"
    )

    raw = _fetch_url(url)
    payload = json.loads(raw)
    if payload.get("status") == "ERROR":
        raise RuntimeError(payload.get("error") or "Polygon request failed.")

    results = payload.get("results") or []
    rows: List[Dict[str, str]] = []
    for bar in results:
        date_str = _timestamp_to_date(bar.get("t"))
        if not date_str:
            continue
        rows.append(
            {
                "date": date_str,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v", 0),
            }
        )

    normalized = _normalize_rows(rows, asset_class, start_date, end_date)
    return _apply_max_rows(normalized, max_rows)


def _download_alpaca(
    symbol: str,
    api_key: str,
    secret_key: str,
    base_url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_rows: Optional[int],
) -> List[Dict[str, str]]:
    start_date, end_date = _ensure_date_range(start_date, end_date)

    params = {
        "timeframe": "1Day",
        "adjustment": "raw",
        "limit": "10000",
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }
    url = f"{base_url.rstrip('/')}/v2/stocks/{symbol.upper()}/bars?{urllib.parse.urlencode(params)}"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    raw = _fetch_url(url, headers=headers)
    payload = json.loads(raw)
    bars = payload.get("bars") or []

    rows: List[Dict[str, str]] = []
    for bar in bars:
        timestamp = bar.get("t")
        date_str = _timestamp_to_date(timestamp)
        if not date_str:
            parsed = _parse_date(timestamp) if isinstance(timestamp, str) else None
            date_str = parsed.isoformat() if parsed else None
        if not date_str:
            continue
        rows.append(
            {
                "date": date_str,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v", 0),
            }
        )

    normalized = _normalize_rows(rows, "equities", start_date, end_date)
    return _apply_max_rows(normalized, max_rows)


def _download_fred(
    series_id: str,
    api_key: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_rows: Optional[int],
) -> List[Dict[str, str]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date.isoformat()
    if end_date:
        params["observation_end"] = end_date.isoformat()

    url = f"{FRED_BASE}/series/observations?{urllib.parse.urlencode(params)}"
    raw = _fetch_url(url)
    payload = json.loads(raw)
    observations = payload.get("observations") or []

    rows: List[Dict[str, str]] = []
    for obs in observations:
        value = _coerce_float(obs.get("value"))
        if value is None:
            continue
        rows.append(
            {
                "date": obs.get("date"),
                "open": value,
                "high": value,
                "low": value,
                "close": value,
                "volume": 0.0,
            }
        )

    normalized = _normalize_rows(rows, "macro", start_date, end_date)
    return _apply_max_rows(normalized, max_rows)


def _summarize(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "no rows"
    return f"{len(rows)} rows ({rows[0]['date']} -> {rows[-1]['date']})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache historical market data from public APIs.")
    parser.add_argument("--provider", default="alpha_vantage", choices=["alpha_vantage", "polygon", "alpaca", "fred"])
    parser.add_argument("--asset-class", default="equities", choices=["equities", "fx", "macro"])
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols or FX pairs to fetch.")
    parser.add_argument("--outputsize", default="full", choices=["full", "compact"])
    parser.add_argument("--start-date", default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--data-dir", default=None, help="Root data directory (default: repo/data).")
    parser.add_argument("--sleep", type=float, default=12.0, help="Seconds to sleep between calls.")
    parser.add_argument("--enable-live", action="store_true", help="Required to allow API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests only.")
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

    api_key = None
    secret_key = None
    base_url = None
    provider = args.provider

    if provider == "alpha_vantage":
        api_key = _get_env("ALPHA_VANTAGE_API_KEY", dotenv)
        if not api_key:
            print("Missing ALPHA_VANTAGE_API_KEY in environment or .env.", file=sys.stderr)
            return 2
        if args.asset_class == "macro":
            print("Alpha Vantage does not support macro in this script.", file=sys.stderr)
            return 2
    elif provider == "polygon":
        api_key = _get_env("POLYGON_API_KEY", dotenv)
        if not api_key:
            print("Missing POLYGON_API_KEY in environment or .env.", file=sys.stderr)
            return 2
        if args.asset_class == "macro":
            print("Polygon does not support macro in this script.", file=sys.stderr)
            return 2
    elif provider == "alpaca":
        api_key = _get_env("ALPACA_API_KEY", dotenv)
        secret_key = _get_env("ALPACA_SECRET_KEY", dotenv)
        base_url = _get_env("ALPACA_DATA_URL", dotenv) or ALPACA_DATA_BASE_DEFAULT
        if not api_key or not secret_key:
            print("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment or .env.", file=sys.stderr)
            return 2
        if args.asset_class != "equities":
            print("Alpaca data support is equities only in this script.", file=sys.stderr)
            return 2
    elif provider == "fred":
        api_key = _get_env("FRED_API_KEY", dotenv)
        if not api_key:
            print("Missing FRED_API_KEY in environment or .env.", file=sys.stderr)
            return 2
        if args.asset_class != "macro":
            print("FRED downloads require --asset-class macro.", file=sys.stderr)
            return 2
    else:
        print(f"Unsupported provider: {provider}", file=sys.stderr)
        return 2

    start_date = _parse_date(args.start_date) if args.start_date else None
    end_date = _parse_date(args.end_date) if args.end_date else None
    data_root = Path(args.data_dir) if args.data_dir else (root / "data")

    for index, symbol in enumerate(args.symbols):
        normalized = _normalize_symbol(symbol)
        if args.asset_class == "fx":
            pair = _parse_fx_pair(symbol)
            normalized = f"{pair[0].lower()}_{pair[1].lower()}"

        output_path = data_root / args.asset_class / f"{normalized}.csv"
        if args.dry_run:
            print(f"[dry-run] {args.provider} {args.asset_class} {symbol} -> {output_path}")
            continue

        try:
            if provider == "alpha_vantage":
                rows = _download_alpha_vantage(
                    symbol=symbol,
                    asset_class=args.asset_class,
                    api_key=api_key,
                    outputsize=args.outputsize,
                    start_date=start_date,
                    end_date=end_date,
                    max_rows=args.max_rows,
                )
            elif provider == "polygon":
                rows = _download_polygon(
                    symbol=symbol,
                    asset_class=args.asset_class,
                    api_key=api_key,
                    start_date=start_date,
                    end_date=end_date,
                    max_rows=args.max_rows,
                )
            elif provider == "alpaca":
                rows = _download_alpaca(
                    symbol=symbol,
                    api_key=api_key,
                    secret_key=secret_key,
                    base_url=base_url,
                    start_date=start_date,
                    end_date=end_date,
                    max_rows=args.max_rows,
                )
            elif provider == "fred":
                rows = _download_fred(
                    series_id=symbol,
                    api_key=api_key,
                    start_date=start_date,
                    end_date=end_date,
                    max_rows=args.max_rows,
                )
            else:
                raise RuntimeError(f"Unsupported provider: {provider}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Failed to fetch {symbol}: {exc}", file=sys.stderr)
            continue

        if not rows:
            print(f"No rows returned for {symbol}", file=sys.stderr)
            continue

        _write_csv(output_path, rows)
        print(f"Wrote {output_path} ({_summarize(rows)})")

        if index < len(args.symbols) - 1 and args.sleep:
            time.sleep(max(args.sleep, 0.0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
