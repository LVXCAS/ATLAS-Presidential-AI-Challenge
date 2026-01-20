from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _safe_pstdev(values: List[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _returns(prices: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        cur = prices[i]
        if prev:
            out.append((cur - prev) / prev)
    return out


def _max_drawdown(prices: List[float]) -> float:
    peak = None
    worst = 0.0
    for p in prices:
        if p is None:
            continue
        if peak is None or p > peak:
            peak = p
        if peak and peak > 0:
            dd = (peak - p) / peak
            if dd > worst:
                worst = dd
    return max(0.0, float(worst))


def _trend_slope(prices: List[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    x = list(range(n))
    x_mean = _safe_mean(x)
    y_mean = _safe_mean(prices)
    denom = sum((xi - x_mean) ** 2 for xi in x) or 1.0
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, prices)) / denom
    last = prices[-1] or 1.0
    return float(slope / last)


FEATURE_ORDER: Tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_5",
    "vol_20",
    "atr_pips_14",
    "adx",
    "rsi",
    "rsi_distance",
    "ema_sep",
    "bb_width",
    "macd_hist_abs",
    "volume_z_20",
    "drawdown_20",
    "trend_slope_20",
)


@dataclass(frozen=True)
class FeatureVector:
    """Named ML feature vector for deterministic ordering."""

    values: Dict[str, float]

    def as_list(self, order: Tuple[str, ...] = FEATURE_ORDER) -> List[float]:
        return [float(self.values.get(name, 0.0) or 0.0) for name in order]


def extract_features(market_data: Dict[str, Any]) -> FeatureVector:
    """Extract deterministic, explainable features from ATLAS market_data.

    This function also winsorizes extreme values to keep offline training and
    inference numerically stable and reproducible across cached datasets.
    """

    indicators = market_data.get("indicators", {}) or {}
    prices = [float(p) for p in (market_data.get("price_history") or []) if p is not None]
    volumes = [float(v) for v in (market_data.get("volume_history") or []) if v is not None]

    price = float(market_data.get("price", prices[-1] if prices else 0.0) or 0.0)
    if price <= 0 or len(prices) < 25:
        return FeatureVector(values={"data_sufficiency": 0.0})

    rets = _returns(prices)

    def _ret_k(k: int) -> float:
        if len(prices) <= k:
            return 0.0
        prev = prices[-k - 1]
        cur = prices[-1]
        if not prev:
            return 0.0
        return (cur - prev) / prev

    ret_1 = _clip(_ret_k(1), -0.25, 0.25)
    ret_5 = _clip(_ret_k(5), -0.50, 0.50)
    ret_20 = _clip(_ret_k(20), -0.80, 0.80)

    vol_5 = _safe_pstdev(rets[-5:]) if len(rets) >= 5 else _safe_pstdev(rets)
    vol_20 = _safe_pstdev(rets[-20:]) if len(rets) >= 20 else _safe_pstdev(rets)
    vol_5 = _clip(vol_5, 0.0, 0.25)
    vol_20 = _clip(vol_20, 0.0, 0.25)

    rsi = _clip(float(indicators.get("rsi", 50.0) or 50.0), 0.0, 100.0)
    adx = _clip(float(indicators.get("adx", 20.0) or 20.0), 0.0, 100.0)
    atr = float(indicators.get("atr", 0.0) or 0.0)
    atr_pips = (atr / price) * 10000.0 if price else 0.0
    atr_pips = _clip(atr_pips, 0.0, 500.0)

    ema50 = float(indicators.get("ema50", price) or price)
    ema200 = float(indicators.get("ema200", price) or price)
    ema_sep = abs(ema50 - ema200) / price if price else 0.0
    ema_sep = _clip(ema_sep, 0.0, 0.25)

    bb_upper = float(indicators.get("bb_upper", price) or price)
    bb_lower = float(indicators.get("bb_lower", price) or price)
    bb_width = (bb_upper - bb_lower) / price if price else 0.0
    bb_width = _clip(bb_width, 0.0, 0.50)

    macd_hist = float(indicators.get("macd_hist", 0.0) or 0.0)
    macd_hist_abs = _clip(abs(macd_hist), 0.0, 1.0)

    rsi_distance = _clip(abs(rsi - 50.0), 0.0, 50.0)

    vol_window = volumes[-20:] if len(volumes) >= 20 else volumes
    vol_mean = _safe_mean(vol_window)
    vol_sd = _safe_pstdev(vol_window)
    last_vol = volumes[-1] if volumes else 0.0
    volume_z_20 = ((last_vol - vol_mean) / vol_sd) if vol_sd > 0 else 0.0
    volume_z_20 = _clip(volume_z_20, -5.0, 5.0)

    dd_window = prices[-20:] if len(prices) >= 20 else prices
    drawdown_20 = _clip(_max_drawdown(dd_window), 0.0, 1.0)

    slope_window = prices[-20:] if len(prices) >= 20 else prices
    trend_slope_20 = _clip(_trend_slope(slope_window), -0.05, 0.05)

    return FeatureVector(
        values={
            "ret_1": float(ret_1),
            "ret_5": float(ret_5),
            "ret_20": float(ret_20),
            "vol_5": float(vol_5),
            "vol_20": float(vol_20),
            "atr_pips_14": float(atr_pips),
            "adx": float(adx),
            "rsi": float(rsi),
            "rsi_distance": float(rsi_distance),
            "ema_sep": float(ema_sep),
            "bb_width": float(bb_width),
            "macd_hist_abs": float(macd_hist_abs),
            "volume_z_20": float(volume_z_20),
            "drawdown_20": float(drawdown_20),
            "trend_slope_20": float(trend_slope_20),
        }
    )
