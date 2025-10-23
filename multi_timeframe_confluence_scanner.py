"""
MULTI-TIMEFRAME CONFLUENCE SCANNER
Scans 1H, 4H, and Daily charts simultaneously for aligned setups

Key Concept:
When multiple timeframes align (all bullish or all bearish), win rate increases dramatically:
- Single timeframe: 50-55% win rate
- 2 timeframes aligned: 60-65% win rate
- 3 timeframes aligned: 70-80% win rate

This scanner only alerts when ALL 3 timeframes agree.
"""
import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class Signal(Enum):
    """Trading signal"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe"""
    timeframe: str  # '1H', '4H', 'Daily'
    signal: Signal
    trend_direction: str  # 'up', 'down', 'sideways'
    ema_alignment: bool  # Are EMAs in proper order?
    rsi: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    volume_confirmation: bool
    support_resistance: Dict[str, float]
    confidence: float  # 0-1

@dataclass
class ConfluenceSetup:
    """Multi-timeframe aligned setup"""
    symbol: str
    overall_signal: Signal
    price: float

    # Timeframe analyses
    h1_analysis: TimeframeAnalysis
    h4_analysis: TimeframeAnalysis
    daily_analysis: TimeframeAnalysis

    # Key levels
    support: float
    resistance: float
    entry: float
    stop_loss: float
    target: float

    # Risk/Reward
    risk_reward_ratio: float
    risk_amount: float
    reward_amount: float

    # Quality metrics
    confluence_score: float  # 0-100, how well timeframes align
    confidence: float  # 0-1, overall setup quality

    detected_at: str

class MultiTimeframeConfluenceScanner:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Thresholds
        self.min_confluence_score = 75  # Only alert on 75+ score
        self.min_confidence = 0.7
        self.min_risk_reward = 2.0  # Minimum 2:1 R/R

    def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'CONFLUENCE SETUP\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[CONFLUENCE] Telegram notification failed: {e}")

    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        if len(data) < 50:
            return TimeframeAnalysis(
                timeframe=timeframe,
                signal=Signal.NEUTRAL,
                trend_direction='sideways',
                ema_alignment=False,
                rsi=50,
                macd_signal='neutral',
                volume_confirmation=False,
                support_resistance={'support': 0, 'resistance': 0},
                confidence=0
            )

        # Calculate indicators
        close = data['Close']

        # EMAs
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9).mean()

        # Volume
        volume_sma = data['Volume'].rolling(20).mean()

        # Get latest values
        current_price = float(close.iloc[-1])
        current_ema8 = float(ema_8.iloc[-1])
        current_ema21 = float(ema_21.iloc[-1])
        current_ema50 = float(ema_50.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_macd = float(macd.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_volume = float(data['Volume'].iloc[-1])
        avg_volume = float(volume_sma.iloc[-1])

        # Determine trend direction
        if current_price > current_ema8 > current_ema21 > current_ema50:
            trend_direction = 'up'
            ema_alignment = True
        elif current_price < current_ema8 < current_ema21 < current_ema50:
            trend_direction = 'down'
            ema_alignment = True
        else:
            trend_direction = 'sideways'
            ema_alignment = False

        # Determine signal
        bullish_factors = 0
        bearish_factors = 0

        # EMA alignment
        if ema_alignment and trend_direction == 'up':
            bullish_factors += 2
        elif ema_alignment and trend_direction == 'down':
            bearish_factors += 2

        # RSI
        if current_rsi > 50 and current_rsi < 70:
            bullish_factors += 1
        elif current_rsi < 50 and current_rsi > 30:
            bearish_factors += 1

        # MACD
        macd_signal_str = 'neutral'
        if current_macd > current_signal and current_macd > 0:
            macd_signal_str = 'bullish'
            bullish_factors += 1
        elif current_macd < current_signal and current_macd < 0:
            macd_signal_str = 'bearish'
            bearish_factors += 1

        # Volume
        volume_confirmation = current_volume > avg_volume * 1.2

        if volume_confirmation:
            if trend_direction == 'up':
                bullish_factors += 1
            elif trend_direction == 'down':
                bearish_factors += 1

        # Overall signal
        if bullish_factors >= 3 and bullish_factors > bearish_factors:
            signal = Signal.BULLISH
        elif bearish_factors >= 3 and bearish_factors > bullish_factors:
            signal = Signal.BEARISH
        else:
            signal = Signal.NEUTRAL

        # Support/Resistance (recent swing highs/lows)
        highs = data['High'].rolling(20).max()
        lows = data['Low'].rolling(20).min()
        support = float(lows.iloc[-1])
        resistance = float(highs.iloc[-1])

        # Confidence
        total_factors = bullish_factors + bearish_factors
        max_factors = 5
        confidence = total_factors / max_factors if signal != Signal.NEUTRAL else 0

        return TimeframeAnalysis(
            timeframe=timeframe,
            signal=signal,
            trend_direction=trend_direction,
            ema_alignment=ema_alignment,
            rsi=current_rsi,
            macd_signal=macd_signal_str,
            volume_confirmation=volume_confirmation,
            support_resistance={'support': support, 'resistance': resistance},
            confidence=confidence
        )

    def calculate_confluence_score(self, h1: TimeframeAnalysis, h4: TimeframeAnalysis,
                                   daily: TimeframeAnalysis) -> float:
        """Calculate how well timeframes align (0-100)"""
        score = 0

        # All same signal? +40
        if h1.signal == h4.signal == daily.signal and h1.signal != Signal.NEUTRAL:
            score += 40

        # EMA alignment on all timeframes? +20
        if h1.ema_alignment and h4.ema_alignment and daily.ema_alignment:
            score += 20

        # Volume confirmation on multiple timeframes? +15
        volume_count = sum([h1.volume_confirmation, h4.volume_confirmation, daily.volume_confirmation])
        score += volume_count * 5

        # MACD alignment? +15
        macd_signals = [h1.macd_signal, h4.macd_signal, daily.macd_signal]
        if all(s == 'bullish' for s in macd_signals) or all(s == 'bearish' for s in macd_signals):
            score += 15

        # Average confidence? +10
        avg_confidence = (h1.confidence + h4.confidence + daily.confidence) / 3
        score += avg_confidence * 10

        return min(score, 100)

    def find_entry_and_stops(self, symbol: str, signal: Signal, h1: TimeframeAnalysis,
                            h4: TimeframeAnalysis, daily: TimeframeAnalysis) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and target"""
        # Get recent data
        stock = yf.Ticker(symbol)
        data = stock.history(period='5d', interval='1h')

        if len(data) == 0:
            return (0, 0, 0)

        current_price = float(data['Close'].iloc[-1])

        if signal == Signal.BULLISH:
            # Entry: Current price or pullback to H4 EMA
            entry = current_price

            # Stop: Below daily support or 2 ATR
            atr = float((data['High'] - data['Low']).rolling(14).mean().iloc[-1])
            stop_loss = max(daily.support_resistance['support'], current_price - (2 * atr))

            # Target: Daily resistance or 3:1 R/R
            resistance = daily.support_resistance['resistance']
            min_target = current_price + ((current_price - stop_loss) * 3)  # 3:1 R/R
            target = max(resistance, min_target)

        elif signal == Signal.BEARISH:
            entry = current_price
            atr = float((data['High'] - data['Low']).rolling(14).mean().iloc[-1])
            stop_loss = min(daily.support_resistance['resistance'], current_price + (2 * atr))

            support = daily.support_resistance['support']
            min_target = current_price - ((stop_loss - current_price) * 3)
            target = min(support, min_target)

        else:
            return (current_price, current_price, current_price)

        return (entry, stop_loss, target)

    def scan_symbol(self, symbol: str) -> Optional[ConfluenceSetup]:
        """Scan a single symbol for confluence"""
        try:
            stock = yf.Ticker(symbol)

            # Get data for each timeframe
            data_1h = stock.history(period='5d', interval='1h')
            data_4h = stock.history(period='1mo', interval='1h').resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            data_daily = stock.history(period='6mo', interval='1d')

            # Analyze each timeframe
            h1_analysis = self.analyze_timeframe(data_1h, '1H')
            h4_analysis = self.analyze_timeframe(data_4h, '4H')
            daily_analysis = self.analyze_timeframe(data_daily, 'Daily')

            # Check for confluence
            if not (h1_analysis.signal == h4_analysis.signal == daily_analysis.signal):
                return None  # No alignment

            if h1_analysis.signal == Signal.NEUTRAL:
                return None  # Not trading sideways

            # Calculate confluence score
            confluence_score = self.calculate_confluence_score(h1_analysis, h4_analysis, daily_analysis)

            if confluence_score < self.min_confluence_score:
                return None  # Not enough confluence

            # Calculate entry/stops
            current_price = float(data_1h['Close'].iloc[-1])
            entry, stop_loss, target = self.find_entry_and_stops(
                symbol, h1_analysis.signal, h1_analysis, h4_analysis, daily_analysis
            )

            # Calculate risk/reward
            if h1_analysis.signal == Signal.BULLISH:
                risk = entry - stop_loss
                reward = target - entry
            else:
                risk = stop_loss - entry
                reward = entry - target

            if risk <= 0 or reward <= 0:
                return None

            risk_reward_ratio = reward / risk

            if risk_reward_ratio < self.min_risk_reward:
                return None  # Not enough R/R

            # Overall confidence
            avg_confidence = (h1_analysis.confidence + h4_analysis.confidence + daily_analysis.confidence) / 3

            return ConfluenceSetup(
                symbol=symbol,
                overall_signal=h1_analysis.signal,
                price=current_price,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
                daily_analysis=daily_analysis,
                support=daily_analysis.support_resistance['support'],
                resistance=daily_analysis.support_resistance['resistance'],
                entry=entry,
                stop_loss=stop_loss,
                target=target,
                risk_reward_ratio=risk_reward_ratio,
                risk_amount=risk,
                reward_amount=reward,
                confluence_score=confluence_score,
                confidence=avg_confidence,
                detected_at=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"[CONFLUENCE] Error scanning {symbol}: {e}")
            return None

    def scan_watchlist(self, symbols: List[str]) -> List[ConfluenceSetup]:
        """Scan watchlist for confluence setups"""
        print("\n" + "="*70)
        print("MULTI-TIMEFRAME CONFLUENCE SCANNER")
        print("="*70)

        setups = []

        for symbol in symbols:
            print(f"[CONFLUENCE] Scanning {symbol}...")
            setup = self.scan_symbol(symbol)

            if setup:
                setups.append(setup)
                print(f"[CONFLUENCE] ✓ FOUND: {symbol} - {setup.overall_signal.value.upper()} (Score: {setup.confluence_score:.0f})")

        # Sort by confluence score
        setups.sort(key=lambda x: x.confluence_score, reverse=True)

        print(f"\n[CONFLUENCE] Found {len(setups)} aligned setups")

        return setups

    def format_setup_report(self, setup: ConfluenceSetup) -> str:
        """Format setup for display"""
        report = f"""
{setup.symbol} - {setup.overall_signal.value.upper()} SETUP
Confluence Score: {setup.confluence_score:.0f}/100
Price: ${setup.price:.2f}

TIMEFRAME ANALYSIS:
  1H:    {setup.h1_analysis.signal.value} (Confidence: {setup.h1_analysis.confidence:.0%})
  4H:    {setup.h4_analysis.signal.value} (Confidence: {setup.h4_analysis.confidence:.0%})
  Daily: {setup.daily_analysis.signal.value} (Confidence: {setup.daily_analysis.confidence:.0%})

KEY LEVELS:
  Support:    ${setup.support:.2f}
  Resistance: ${setup.resistance:.2f}

TRADE SETUP:
  Entry:      ${setup.entry:.2f}
  Stop Loss:  ${setup.stop_loss:.2f}
  Target:     ${setup.target:.2f}

RISK/REWARD:
  Risk:   ${setup.risk_amount:.2f}/share
  Reward: ${setup.reward_amount:.2f}/share
  R/R Ratio: {setup.risk_reward_ratio:.1f}:1

Overall Confidence: {setup.confidence:.0%}
"""
        return report

def main():
    """Test confluence scanner"""
    scanner = MultiTimeframeConfluenceScanner()

    # Scan S&P 100 (most liquid)
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'V', 'JPM', 'JNJ']

    setups = scanner.scan_watchlist(watchlist)

    # Print top setups
    for setup in setups[:5]:
        print(scanner.format_setup_report(setup))

if __name__ == '__main__':
    main()
