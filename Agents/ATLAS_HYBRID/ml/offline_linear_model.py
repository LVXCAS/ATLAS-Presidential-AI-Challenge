from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


@dataclass(frozen=True)
class OfflineLinearModel:
    """Deterministic, offline-trained linear model with standardization."""

    name: str
    target: str
    horizon: int
    feature_order: Tuple[str, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    coef: Tuple[float, ...]
    intercept: float
    calibration: Dict[str, Any]
    evaluation: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "OfflineLinearModel":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        schema = int(raw.get("schema_version", 0) or 0)
        if schema != 1:
            raise ValueError(f"Unsupported model schema_version={schema}")

        feature_order = tuple(str(x) for x in (raw.get("feature_order") or []))
        mean = tuple(float(x) for x in (raw.get("scaler") or {}).get("mean", []))
        std = tuple(float(x) for x in (raw.get("scaler") or {}).get("std", []))
        coef = tuple(float(x) for x in (raw.get("coef") or []))

        if not (feature_order and mean and std and coef):
            raise ValueError("Invalid model artifact: missing arrays")
        if not (len(feature_order) == len(mean) == len(std) == len(coef)):
            raise ValueError("Invalid model artifact: array lengths do not match")

        return OfflineLinearModel(
            name=str(raw.get("name", "offline_linear_model")),
            target=str(raw.get("target", "")),
            horizon=int(raw.get("horizon", 1) or 1),
            feature_order=feature_order,
            mean=mean,
            std=std,
            coef=coef,
            intercept=float(raw.get("intercept", 0.0) or 0.0),
            calibration=raw.get("calibration", {}) if isinstance(raw.get("calibration"), dict) else {},
            evaluation=raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {},
        )

    def predict_from_features(self, feature_values: Dict[str, float]) -> float:
        x = [float(feature_values.get(name, 0.0) or 0.0) for name in self.feature_order]
        xz = []
        for xi, mu, sd in zip(x, self.mean, self.std):
            denom = sd if sd and sd > 0 else 1.0
            xz.append((xi - mu) / denom)
        return float(self.intercept + _dot(self.coef, xz))

    def top_contributions(
        self,
        feature_values: Dict[str, float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        x = [float(feature_values.get(name, 0.0) or 0.0) for name in self.feature_order]
        contribs: List[Tuple[str, float, float]] = []
        for name, xi, mu, sd, w in zip(self.feature_order, x, self.mean, self.std, self.coef):
            denom = sd if sd and sd > 0 else 1.0
            xz = (xi - mu) / denom
            contribs.append((name, float(w * xz), float(xz)))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        return [
            {"feature": n, "contribution": round(c, 6), "z": round(z, 3)}
            for n, c, z in contribs[: max(1, int(top_k))]
        ]

    def calibrated_risk(self, prediction: float) -> float:
        """Map a raw prediction into a 0..1 risk using stored calibration points."""

        p = float(prediction)
        p50 = self.calibration.get("p50")
        p90 = self.calibration.get("p90")
        try:
            p50f = float(p50)
            p90f = float(p90)
        except Exception:
            return 0.5
        denom = (p90f - p50f) or 1.0
        return _clamp01((p - p50f) / denom)
