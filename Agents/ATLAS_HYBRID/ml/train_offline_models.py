#!/usr/bin/env python3

from __future__ import annotations

"""Offline ML training script (deterministic).

Trains two small linear ridge models on cached OHLCV CSVs:
1) `realized_volatility_5`: forecast near-term realized volatility
2) `max_drawdown_10`: forecast near-term drawdown risk

Determinism guarantees:
- local cached CSVs only (no network)
- per-symbol time split (first 70% train, last 30% test)
- closed-form ridge regression (no stochastic optimization)
- scaler + weights are rounded before saving, and metrics are computed using
  the saved (rounded) parameters.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HYBRID_DIR = Path(__file__).resolve().parents[1]
if str(HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_DIR))

from quant_team_utils import make_market_data  # noqa: E402
from ml.feature_extraction import FEATURE_ORDER, extract_features  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def data_fingerprint(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.as_posix()):
        h.update(p.as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(read_bytes(p))
        h.update(b"\0")
    return h.hexdigest()


def load_cached_series(path: Path) -> Tuple[List[float], List[float]]:
    closes: List[float] = []
    volumes: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                close = float(row.get("close") or row.get("Close") or row.get("CLOSE") or 0.0)
                vol = float(row.get("volume") or row.get("Volume") or row.get("VOLUME") or 0.0)
            except Exception:
                continue
            if close <= 0:
                continue
            closes.append(close)
            volumes.append(vol)
    return closes, volumes


def future_realized_vol(prices: List[float], start: int, horizon: int) -> float:
    window = prices[start : start + horizon + 1]
    if len(window) < 3:
        return 0.0
    rets = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        cur = window[i]
        if prev:
            rets.append((cur - prev) / prev)
    return float(np.std(np.array(rets, dtype=float), ddof=0)) if len(rets) >= 2 else 0.0


def future_max_drawdown(prices: List[float], start: int, horizon: int) -> float:
    window = prices[start : start + horizon + 1]
    if len(window) < 3:
        return 0.0
    peak = window[0]
    worst = 0.0
    for p in window:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (peak - p) / peak
            if dd > worst:
                worst = dd
    return float(max(0.0, worst))


def per_symbol_time_split(symbols: Sequence[str], train_frac: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic split: for each symbol, first train_frac rows are train."""

    if not symbols:
        return np.array([], dtype=int), np.array([], dtype=int)

    symbols_arr = np.array([str(s) for s in symbols], dtype=object)
    n = len(symbols_arr)
    train_mask = np.zeros((n,), dtype=bool)

    for sym in sorted(set(symbols_arr.tolist())):
        idx = np.where(symbols_arr == sym)[0]
        if idx.size == 0:
            continue
        n_train = max(1, int(idx.size * float(train_frac)))
        train_mask[idx[:n_train]] = True

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    return train_idx.astype(int), test_idx.astype(int)


class RidgeFit:
    def __init__(
        self,
        *,
        intercept: float,
        coef: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        metrics: Dict[str, Any],
        calibration: Dict[str, Any],
    ) -> None:
        self.intercept = float(intercept)
        self.coef = coef
        self.mean = mean
        self.std = std
        self.metrics = metrics
        self.calibration = calibration


def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    symbols: Sequence[str],
    alpha: float,
    train_frac: float = 0.7,
    decimals: int = 10,
) -> RidgeFit:
    train_idx, test_idx = per_symbol_time_split(symbols, train_frac=train_frac)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx] if len(test_idx) else X_train[:0]
    y_test = y[test_idx] if len(test_idx) else y_train[:0]

    mean_vec = X_train.mean(axis=0)
    std_vec = X_train.std(axis=0, ddof=0)
    std_vec = np.where(std_vec > 0, std_vec, 1.0)

    mean_vec = np.round(mean_vec, decimals)
    std_vec = np.round(std_vec, decimals)
    std_vec = np.where(std_vec > 0, std_vec, 1.0)

    X_train_z = (X_train - mean_vec) / std_vec
    X_test_z = (X_test - mean_vec) / std_vec if len(X_test) else X_test

    ones_train = np.ones((len(X_train_z), 1), dtype=float)
    Xa = np.concatenate([ones_train, X_train_z], axis=1)
    I = np.eye(Xa.shape[1], dtype=float)
    I[0, 0] = 0.0

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        A = Xa.T @ Xa + (float(alpha) * I)
        b = Xa.T @ y_train
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("Non-finite ridge system matrix; check feature scaling.")
    w = np.linalg.solve(A, b)
    if not np.isfinite(w).all():
        raise ValueError("Non-finite ridge solution; check feature scaling.")

    intercept = float(np.round(float(w[0]), decimals))
    coef = np.round(w[1:].astype(float), decimals)

    def _predict(Xz: np.ndarray) -> np.ndarray:
        if Xz.size == 0:
            return np.zeros((0,), dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return intercept + (Xz @ coef)

    yhat_train = _predict(X_train_z)
    yhat_test = _predict(X_test_z)

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        return float(np.mean(np.abs(a - b)))

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size < 2:
            return 0.0
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        return float(1.0 - (ss_res / (ss_tot or 1.0)))

    train_metrics = {"mae": round(_mae(y_train, yhat_train), 8), "r2": round(_r2(y_train, yhat_train), 6)}
    test_metrics = {"mae": round(_mae(y_test, yhat_test), 8), "r2": round(_r2(y_test, yhat_test), 6)}

    p50 = float(np.quantile(y_train, 0.50)) if y_train.size else 0.0
    p90 = float(np.quantile(y_train, 0.90)) if y_train.size else max(p50, 1e-9)
    calibration = {"p50": round(p50, decimals), "p90": round(p90, decimals)}

    metrics = {
        "split": {"type": "per_symbol_time", "train_frac": float(train_frac)},
        "train": train_metrics,
        "test": test_metrics,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    return RidgeFit(
        intercept=intercept,
        coef=coef,
        mean=mean_vec,
        std=std_vec,
        metrics=metrics,
        calibration=calibration,
    )


def round_list(values: np.ndarray, decimals: int = 10) -> List[float]:
    return [float(round(float(v), decimals)) for v in values.tolist()]


def artifact(
    *,
    name: str,
    target: str,
    horizon: int,
    alpha: float,
    fit: RidgeFit,
    training_meta: Dict[str, Any],
    decimals: int = 10,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "model_type": "ridge_regression",
        "target": target,
        "horizon": horizon,
        "feature_order": list(FEATURE_ORDER),
        "scaler": {"mean": round_list(fit.mean, decimals), "std": round_list(fit.std, decimals)},
        "coef": round_list(fit.coef, decimals),
        "intercept": float(round(fit.intercept, decimals)),
        "alpha": float(alpha),
        "calibration": fit.calibration,
        "evaluation": {"metrics": fit.metrics},
        "training": training_meta,
        "determinism": {
            "random_seed": 0,
            "notes": "Per-symbol time split + closed-form ridge; scaler/weights rounded; offline-only.",
        },
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_training_paths(data_dir: Path, asset_classes: List[str], symbols: Optional[List[str]]) -> List[Path]:
    paths: List[Path] = []
    for asset in asset_classes:
        folder = data_dir / asset
        if not folder.exists():
            continue
        if symbols:
            for sym in symbols:
                candidate = folder / f"{sym}.csv"
                if candidate.exists():
                    paths.append(candidate)
                else:
                    for p in folder.glob("*.csv"):
                        if p.stem.lower() == sym.lower():
                            paths.append(p)
                            break
        else:
            paths.extend(sorted(folder.glob("*.csv"), key=lambda p: p.as_posix()))

    seen = set()
    unique: List[Path] = []
    for p in sorted(paths, key=lambda x: x.as_posix()):
        if p.as_posix() in seen:
            continue
        seen.add(p.as_posix())
        unique.append(p)
    return unique


def build_dataset(
    csv_paths: Sequence[Path],
    min_history: int,
    horizon_vol: int,
    horizon_dd: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    rows_X: List[List[float]] = []
    rows_y_vol: List[float] = []
    rows_y_dd: List[float] = []
    row_symbols: List[str] = []

    per_symbol_counts: Dict[str, int] = {}

    max_h = max(horizon_vol, horizon_dd)
    for path in csv_paths:
        prices, volumes = load_cached_series(path)
        if len(prices) < (min_history + max_h + 5):
            continue

        sym = path.stem
        count = 0
        for step in range(min_history, len(prices) - max_h - 1):
            md = make_market_data(
                pair=sym,
                prices=prices,
                step=step,
                volume_history=volumes,
                data_source="cached_csv",
            )
            fv = extract_features(md)
            if fv.values.get("data_sufficiency") == 0.0:
                continue

            x = fv.as_list()
            yv = future_realized_vol(prices, start=step, horizon=horizon_vol)
            ydd = future_max_drawdown(prices, start=step, horizon=horizon_dd)

            if not all(math.isfinite(float(v)) for v in x):
                continue
            if not (math.isfinite(float(yv)) and math.isfinite(float(ydd))):
                continue

            rows_X.append([float(v) for v in x])
            rows_y_vol.append(float(yv))
            rows_y_dd.append(float(ydd))
            row_symbols.append(sym)
            count += 1

        if count:
            per_symbol_counts[sym] = count

    X = np.array(rows_X, dtype=float)
    y_vol = np.array(rows_y_vol, dtype=float)
    y_dd = np.array(rows_y_dd, dtype=float)

    meta = {
        "min_history": int(min_history),
        "horizon_vol": int(horizon_vol),
        "horizon_dd": int(horizon_dd),
        "symbols_used": sorted(per_symbol_counts.keys()),
        "samples_per_symbol": {k: int(v) for k, v in sorted(per_symbol_counts.items())},
        "n_samples_total": int(len(rows_X)),
    }
    return X, y_vol, y_dd, row_symbols, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(_repo_root() / "data"),
        help="Root data directory containing asset class folders (fx/, equities/)",
    )
    parser.add_argument(
        "--asset-classes",
        default="fx,equities",
        help="Comma-separated asset classes to include (default: fx,equities)",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol stems to include (default: all cached CSVs found)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Per-symbol train fraction")
    parser.add_argument("--min-history", type=int, default=60, help="Warmup steps before sampling features")
    parser.add_argument("--horizon-vol", type=int, default=5, help="Volatility target horizon (steps)")
    parser.add_argument("--horizon-dd", type=int, default=10, help="Drawdown target horizon (steps)")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "models"),
        help="Where to write model artifacts",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    asset_classes = [s.strip() for s in str(args.asset_classes).split(",") if s.strip()]
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()] or None
    alpha = float(args.alpha)

    csv_paths = resolve_training_paths(data_dir, asset_classes=asset_classes, symbols=symbols)
    if not csv_paths:
        raise SystemExit(f"No CSVs found under {data_dir} for asset_classes={asset_classes}")

    fingerprint = data_fingerprint(csv_paths)

    X, y_vol, y_dd, row_symbols, ds_meta = build_dataset(
        csv_paths=csv_paths,
        min_history=int(args.min_history),
        horizon_vol=int(args.horizon_vol),
        horizon_dd=int(args.horizon_dd),
    )
    if X.size == 0:
        raise SystemExit("No usable training samples (check CSV schema/length).")

    train_frac = float(args.train_frac)

    fit_vol = ridge_fit(X, y_vol, row_symbols, alpha=alpha, train_frac=train_frac)
    fit_dd = ridge_fit(X, y_dd, row_symbols, alpha=alpha, train_frac=train_frac)

    training_meta = {
        "data_dir": str(data_dir),
        "asset_classes": asset_classes,
        "symbols_requested": symbols or "ALL",
        "csv_count": int(len(csv_paths)),
        "csv_fingerprint_sha256": fingerprint,
        **ds_meta,
    }

    out_dir = Path(args.out_dir)
    vol_art = artifact(
        name="offline_ridge_volatility_v1",
        target="realized_volatility_5",
        horizon=int(args.horizon_vol),
        alpha=alpha,
        fit=fit_vol,
        training_meta=training_meta,
    )
    dd_art = artifact(
        name="offline_ridge_drawdown_v1",
        target="max_drawdown_10",
        horizon=int(args.horizon_dd),
        alpha=alpha,
        fit=fit_dd,
        training_meta=training_meta,
    )

    write_json(out_dir / "offline_ridge_volatility_v1.json", vol_art)
    write_json(out_dir / "offline_ridge_drawdown_v1.json", dd_art)

    print("[ATLAS ML] Wrote model artifacts:")
    print(f" - {out_dir / 'offline_ridge_volatility_v1.json'}")
    print(f" - {out_dir / 'offline_ridge_drawdown_v1.json'}")
    print("[ATLAS ML] Test metrics (per-symbol time split):")
    print(f" - vol: {vol_art['evaluation']['metrics']['test']}")
    print(f" - dd : {dd_art['evaluation']['metrics']['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
