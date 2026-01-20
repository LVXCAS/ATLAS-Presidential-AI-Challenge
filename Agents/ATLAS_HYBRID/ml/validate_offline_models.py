#!/usr/bin/env python3

from __future__ import annotations

"""Deterministic validation script for offline ML model artifacts.

Validates that the saved JSON artifacts are self-consistent by rebuilding the
same offline dataset and verifying the stored test MAE (within tolerance).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

HYBRID_DIR = Path(__file__).resolve().parents[1]
if str(HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(HYBRID_DIR))

from ml.feature_extraction import FEATURE_ORDER  # noqa: E402
from ml.offline_linear_model import OfflineLinearModel  # noqa: E402
from ml.train_offline_models import (  # noqa: E402
    build_dataset,
    data_fingerprint,
    per_symbol_time_split,
    resolve_training_paths,
)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _row_to_features(row: np.ndarray) -> Dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(FEATURE_ORDER)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        default=str(Path(__file__).parent / "models"),
        help="Directory containing offline ML JSON artifacts",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Allowed MAE drift vs artifact")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    vol_path = models_dir / "offline_ridge_volatility_v1.json"
    dd_path = models_dir / "offline_ridge_drawdown_v1.json"

    vol_art = json.loads(vol_path.read_text(encoding="utf-8"))
    dd_art = json.loads(dd_path.read_text(encoding="utf-8"))

    training = vol_art.get("training") or {}
    training_dd = dd_art.get("training") or {}
    if training != training_dd:
        raise SystemExit("Training metadata mismatch between volatility and drawdown artifacts.")

    eval_metrics = (vol_art.get("evaluation") or {}).get("metrics") or {}
    split = eval_metrics.get("split") or {}
    train_frac = float(split.get("train_frac", 0.7) or 0.7)

    data_dir = Path(training.get("data_dir") or "data")
    asset_classes = list(training.get("asset_classes") or ["fx", "equities"])
    symbols_requested = training.get("symbols_requested", "ALL")
    symbols = None if symbols_requested == "ALL" else list(symbols_requested or [])

    csv_paths = resolve_training_paths(data_dir, asset_classes=asset_classes, symbols=symbols)
    fingerprint = data_fingerprint(csv_paths)

    if str(fingerprint) != str(training.get("csv_fingerprint_sha256")):
        raise SystemExit("Cached CSV fingerprint mismatch; data files changed since training.")

    X, y_vol, y_dd, row_symbols, meta = build_dataset(
        csv_paths=csv_paths,
        min_history=int(training.get("min_history", 60)),
        horizon_vol=int(training.get("horizon_vol", 5)),
        horizon_dd=int(training.get("horizon_dd", 10)),
    )

    train_idx, test_idx = per_symbol_time_split(row_symbols, train_frac=train_frac)

    vol_model = OfflineLinearModel.load(vol_path)
    dd_model = OfflineLinearModel.load(dd_path)

    yv_test = y_vol[test_idx]
    ydd_test = y_dd[test_idx]

    pv = np.array([vol_model.predict_from_features(_row_to_features(X[i])) for i in test_idx], dtype=float)
    pdd = np.array([dd_model.predict_from_features(_row_to_features(X[i])) for i in test_idx], dtype=float)

    if not np.isfinite(pv).all() or not np.isfinite(pdd).all():
        raise SystemExit("Non-finite predictions found.")

    mae_vol = _mae(yv_test, pv)
    mae_dd = _mae(ydd_test, pdd)

    art_mae_vol = float(((vol_art.get("evaluation") or {}).get("metrics") or {}).get("test", {}).get("mae", 0.0) or 0.0)
    art_mae_dd = float(((dd_art.get("evaluation") or {}).get("metrics") or {}).get("test", {}).get("mae", 0.0) or 0.0)

    vol_ok = abs(mae_vol - art_mae_vol) <= float(args.tolerance)
    dd_ok = abs(mae_dd - art_mae_dd) <= float(args.tolerance)

    report: Dict[str, Any] = {
        "csv_count": int(training.get("csv_count", 0) or 0),
        "symbols_used": list(training.get("symbols_used") or []),
        "samples_total": int(meta.get("n_samples_total", 0) or 0),
        "samples_test": int(len(test_idx)),
        "split": {"type": str(split.get("type", "")), "train_frac": train_frac},
        "volatility_model": {"mae": round(float(mae_vol), 8), "artifact_mae": round(art_mae_vol, 8), "ok": bool(vol_ok)},
        "drawdown_model": {"mae": round(float(mae_dd), 8), "artifact_mae": round(art_mae_dd, 8), "ok": bool(dd_ok)},
        "tolerance": float(args.tolerance),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if not (vol_ok and dd_ok):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
