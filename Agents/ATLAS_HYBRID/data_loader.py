from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from simulation_config import SIMULATION_ONLY, USE_LIVE_DATA

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


class DataLoaderError(Exception):
    pass


class LiveDataDisabledError(DataLoaderError):
    pass


class PandasUnavailableError(DataLoaderError):
    pass


class MissingDataError(DataLoaderError):
    pass


class SchemaError(DataLoaderError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_symbol(symbol: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in (symbol or "")).strip("_")


def _resolve_csv_path(data_dir: Path, asset_class: str, symbol: str) -> Path:
    normalized = _normalize_symbol(symbol)
    return data_dir / asset_class / f"{normalized}.csv"


def _find_csv_path(data_dir: Path, asset_class: str, symbol: str) -> Path | None:
    folder = data_dir / asset_class
    normalized = _normalize_symbol(symbol)
    compact = normalized.replace("_", "")

    candidates = [
        folder / f"{normalized}.csv",
        folder / f"{compact}.csv",
        folder / f"{symbol}.csv",
        folder / f"{symbol.lower()}.csv",
        folder / f"{symbol.upper()}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    if not folder.exists():
        return None

    # Case-insensitive fallback on filename stem.
    for path in folder.glob("*.csv"):
        stem = path.stem.lower()
        if stem == normalized or stem == compact:
            return path

    return None


def _normalize_columns(df) -> None:
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        for alt in ("timestamp", "time", "datetime"):
            if alt in df.columns:
                df.rename(columns={alt: "date"}, inplace=True)
                break


def load_cached_csv(
    symbol: str,
    asset_class: str,
    data_dir: Optional[str | Path] = None,
    max_rows: Optional[int] = None,
):
    """
    Load a cached OHLCV CSV into a deterministic pandas DataFrame.
    """
    if pd is None:
        raise PandasUnavailableError("pandas is required to load cached CSVs.")

    if asset_class not in {"fx", "equities"}:
        raise DataLoaderError(f"Unsupported asset class: {asset_class}")

    root = Path(data_dir) if data_dir else (_repo_root() / "data")
    path = _find_csv_path(root, asset_class, symbol) or _resolve_csv_path(root, asset_class, symbol)
    if not path.exists():
        raise MissingDataError(f"Cached CSV not found: {path}")

    df = pd.read_csv(path)
    _normalize_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"CSV missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df["date"] = df["date"].dt.tz_convert(None)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if max_rows:
        df = df.tail(int(max_rows))
    df = df.reset_index(drop=True)
    return df


def load_live_data(*_args, **_kwargs):
    """
    Live data is intentionally disabled in this repository.
    """
    if SIMULATION_ONLY or not USE_LIVE_DATA:
        raise LiveDataDisabledError("Live data access is disabled for safety and reproducibility.")
    raise LiveDataDisabledError("Live data access is not implemented in this repository.")
