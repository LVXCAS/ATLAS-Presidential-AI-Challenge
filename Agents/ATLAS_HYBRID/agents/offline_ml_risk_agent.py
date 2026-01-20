"""Offline ML Risk Agent (Track II, deterministic).

This agent loads two offline-trained ridge regression models and converts their
forecasts into a normalized 0..1 risk/uncertainty score.

Models (trained offline on cached CSVs):
- realized_volatility_5: near-term realized volatility forecast
- max_drawdown_10: near-term drawdown-risk forecast

The agent does NOT use live data and does NOT execute trades.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .base_agent import AgentAssessment, BaseAgent
from ml.feature_extraction import extract_features
from ml.offline_linear_model import OfflineLinearModel


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class OfflineMLRiskAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.1):
        super().__init__(name="OfflineMLRiskAgent", initial_weight=initial_weight)

        self.models_dir = Path(__file__).resolve().parents[1] / "ml" / "models"
        self.vol_path = self.models_dir / "offline_ridge_volatility_v1.json"
        self.dd_path = self.models_dir / "offline_ridge_drawdown_v1.json"

        self.vol_model: Optional[OfflineLinearModel] = None
        self.dd_model: Optional[OfflineLinearModel] = None
        self.vol_sha256: Optional[str] = None
        self.dd_sha256: Optional[str] = None

        self._load_models()

    def _load_models(self) -> None:
        if self.vol_path.exists():
            self.vol_model = OfflineLinearModel.load(self.vol_path)
            self.vol_sha256 = _sha256_file(self.vol_path)
        if self.dd_path.exists():
            self.dd_model = OfflineLinearModel.load(self.dd_path)
            self.dd_sha256 = _sha256_file(self.dd_path)

    def analyze(self, market_data: Dict[str, Any]) -> AgentAssessment:
        if self.vol_model is None or self.dd_model is None:
            return AgentAssessment(
                score=0.5,
                explanation="Offline ML models missing; skipping ML risk lens.",
                details={
                    "error": "missing_model_artifacts",
                    "expected_paths": {
                        "volatility": str(self.vol_path),
                        "drawdown": str(self.dd_path),
                    },
                    "data_sufficiency": "insufficient",
                },
            )

        fv = extract_features(market_data)
        if fv.values.get("data_sufficiency") == 0.0:
            return AgentAssessment(
                score=0.5,
                explanation="Not enough history for ML features; ML risk is uncertain.",
                details={"data_sufficiency": "insufficient"},
            )

        features = fv.values

        pred_vol = max(0.0, float(self.vol_model.predict_from_features(features)))
        pred_dd = max(0.0, float(self.dd_model.predict_from_features(features)))

        vol_risk = float(self.vol_model.calibrated_risk(pred_vol))
        dd_risk = float(self.dd_model.calibrated_risk(pred_dd))

        score = _clamp01((0.45 * vol_risk) + (0.55 * dd_risk))

        if dd_risk >= vol_risk and dd_risk >= 0.55:
            explanation = f"ML forecast warns of drawdown risk (next ~10 steps, est. {pred_dd:.1%})."
        elif vol_risk > dd_risk and vol_risk >= 0.55:
            explanation = f"ML forecast warns of higher volatility (next ~5 steps, est. {pred_vol:.3f})."
        elif score <= 0.30:
            explanation = "ML forecast suggests near-term conditions are relatively stable."
        else:
            explanation = "ML forecast is mixed; keep caution when signals conflict."

        details = {
            "predictions": {
                "realized_volatility_5": round(pred_vol, 10),
                "max_drawdown_10": round(pred_dd, 10),
            },
            "risk_components": {
                "volatility_risk": round(vol_risk, 3),
                "drawdown_risk": round(dd_risk, 3),
            },
            "top_features": {
                "volatility_model": self.vol_model.top_contributions(features, top_k=3),
                "drawdown_model": self.dd_model.top_contributions(features, top_k=3),
            },
            "models": [
                {
                    "name": self.vol_model.name,
                    "target": self.vol_model.target,
                    "horizon": self.vol_model.horizon,
                    "feature_order": list(self.vol_model.feature_order),
                    "evaluation": self.vol_model.evaluation,
                    "calibration": self.vol_model.calibration,
                    "artifact_sha256": self.vol_sha256,
                },
                {
                    "name": self.dd_model.name,
                    "target": self.dd_model.target,
                    "horizon": self.dd_model.horizon,
                    "feature_order": list(self.dd_model.feature_order),
                    "evaluation": self.dd_model.evaluation,
                    "calibration": self.dd_model.calibration,
                    "artifact_sha256": self.dd_sha256,
                },
            ],
        }

        return AgentAssessment(score=score, explanation=explanation, details=details)
