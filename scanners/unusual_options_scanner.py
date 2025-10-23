#!/usr/bin/env python3
"""
UNUSUAL OPTIONS ACTIVITY SCANNER
Uses OpenBB to detect institutional options flow

Strategy:
1. Detect options with volume >> open interest
2. Filter for large trades (whales/institutions)
3. Follow the smart money

Why This Works:
- Institutions often know something retail doesn't
- Unusual volume precedes price moves
- Options provide leverage and defined risk
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

try:
    from data.openbb_data_fetcher import OpenBBDataFetcher
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False


@dataclass
class UnusualOptionsSignal:
    """Unusual options activity signal"""
    symbol: str
    option_symbol: str
    type: str  # 'call' or 'put'
    strike: float
    expiration: str
    volume: int
    open_interest: int
    volume_oi_ratio: float
    premium: float
    total_premium: float
    underlying_price: float
    moneyness: str  # 'ITM', 'ATM', 'OTM'
    confidence: float
    detected_at: str


class UnusualOptionsScanner:
    """
    Scanner for unusual options activity

    Detects:
    - High volume/OI ratios (>2x)
    - Large premium trades (>$100K)
    - Institutional-sized orders
    - Directional bets
    """

    def __init__(self):
        """Initialize scanner"""

        if not OPENBB_AVAILABLE:
            print("[WARNING] OpenBB not available")
            print("Install with: pip install openbb")
            self.openbb = None
        else:
            self.openbb = OpenBBDataFetcher()
            print("[UNUSUAL OPTIONS SCANNER] Initialized")

    def classify_moneyness(self, strike: float, underlying_price: float, option_type: str) -> str:
        """
        Classify option as ITM, ATM, or OTM

        Args:
            strike: Strike price
            underlying_price: Current stock price
            option_type: 'call' or 'put'

        Returns:
            'ITM', 'ATM', or 'OTM'
        """

        pct_diff = abs(strike - underlying_price) / underlying_price

        if pct_diff < 0.02:  # Within 2%
            return 'ATM'

        if option_type == 'call':
            return 'ITM' if strike < underlying_price else 'OTM'
        else:  # put
            return 'ITM' if strike > underlying_price else 'OTM'

    def calculate_confidence(self, signal: Dict) -> float:
        """
        Calculate confidence score for unusual activity

        Factors:
        - Volume/OI ratio (higher = more unusual)
        - Total premium (larger = more conviction)
        - Moneyness (ATM/ITM = higher confidence)

        Returns:
            Confidence score 0-1
        """

        confidence = 0.5  # Base

        # Volume/OI ratio
        vol_oi = signal['volume_oi_ratio']
        if vol_oi > 5:
            confidence += 0.2
        elif vol_oi > 3:
            confidence += 0.1

        # Total premium
        if signal['total_premium'] > 1_000_000:  # $1M+
            confidence += 0.2
        elif signal['total_premium'] > 500_000:
            confidence += 0.1

        # Moneyness
        if signal['moneyness'] in ['ATM', 'ITM']:
            confidence += 0.1

        return min(confidence, 0.95)

    def scan(self,
             min_volume: int = 1000,
             min_oi_ratio: float = 2.0,
             min_premium: float = 100000) -> List[UnusualOptionsSignal]:
        """
        Scan for unusual options activity

        Args:
            min_volume: Minimum option volume
            min_oi_ratio: Minimum volume/OI ratio
            min_premium: Minimum total premium ($)

        Returns:
            List of unusual options signals
        """

        if not self.openbb:
            print("[ERROR] OpenBB not initialized")
            return []

        print("\n" + "="*70)
        print("UNUSUAL OPTIONS ACTIVITY SCAN")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Min Volume: {min_volume:,}")
        print(f"Min Vol/OI Ratio: {min_oi_ratio}x")
        print(f"Min Premium: ${min_premium:,}")
        print("="*70 + "\n")

        # Get unusual activity from OpenBB
        df = self.openbb.get_unusual_options_activity(
            min_volume=min_volume,
            min_oi_ratio=min_oi_ratio
        )

        if df.empty:
            print("[NO ACTIVITY] No unusual options detected")
            return []

        signals = []

        for _, row in df.iterrows():
            try:
                # Extract data
                symbol = row.get('symbol', row.get('ticker', ''))
                strike = float(row.get('strike', 0))
                underlying_price = float(row.get('underlying_price', row.get('stock_price', 0)))
                volume = int(row.get('volume', 0))
                oi = int(row.get('open_interest', 1))
                premium = float(row.get('premium', row.get('last_price', 0)))
                option_type = row.get('type', row.get('option_type', 'call')).lower()

                # Calculate metrics
                total_premium = premium * volume * 100  # 100 shares per contract
                vol_oi_ratio = volume / oi if oi > 0 else 0

                # Filter by premium
                if total_premium < min_premium:
                    continue

                # Classify
                moneyness = self.classify_moneyness(strike, underlying_price, option_type)

                # Build signal dict for confidence calc
                signal_dict = {
                    'volume_oi_ratio': vol_oi_ratio,
                    'total_premium': total_premium,
                    'moneyness': moneyness
                }

                confidence = self.calculate_confidence(signal_dict)

                # Create signal
                signal = UnusualOptionsSignal(
                    symbol=symbol,
                    option_symbol=row.get('contract_symbol', row.get('option_symbol', '')),
                    type=option_type,
                    strike=strike,
                    expiration=str(row.get('expiration', row.get('expiration_date', ''))),
                    volume=volume,
                    open_interest=oi,
                    volume_oi_ratio=vol_oi_ratio,
                    premium=premium,
                    total_premium=total_premium,
                    underlying_price=underlying_price,
                    moneyness=moneyness,
                    confidence=confidence,
                    detected_at=datetime.now().isoformat()
                )

                signals.append(signal)

            except Exception as e:
                print(f"[WARNING] Error processing row: {e}")
                continue

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)

        print(f"\n[SCAN COMPLETE] {len(signals)} unusual options signals")
        return signals

    def format_report(self, signals: List[UnusualOptionsSignal]) -> str:
        """Format signals as readable report"""

        if not signals:
            return "\nNo unusual options activity detected\n"

        report = "\n" + "="*70 + "\n"
        report += "UNUSUAL OPTIONS ACTIVITY REPORT\n"
        report += "="*70 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Signals: {len(signals)}\n"
        report += "="*70 + "\n\n"

        for i, signal in enumerate(signals[:10], 1):  # Top 10
            report += f"{i}. {signal.symbol} - {signal.type.upper()}\n"
            report += f"   Strike: ${signal.strike:.2f} ({signal.moneyness})\n"
            report += f"   Expiration: {signal.expiration}\n"
            report += f"   Volume: {signal.volume:,} (vs OI: {signal.open_interest:,})\n"
            report += f"   Vol/OI Ratio: {signal.volume_oi_ratio:.1f}x\n"
            report += f"   Total Premium: ${signal.total_premium:,.0f}\n"
            report += f"   Underlying: ${signal.underlying_price:.2f}\n"
            report += f"   Confidence: {signal.confidence:.0%}\n\n"

        report += "="*70 + "\n"
        return report


def main():
    """Demo unusual options scanner"""

    print("\n" + "="*70)
    print("UNUSUAL OPTIONS SCANNER DEMO")
    print("="*70)

    scanner = UnusualOptionsScanner()

    if not scanner.openbb:
        print("\n[ERROR] OpenBB not available")
        print("Install with: pip install openbb")
        return

    # Scan for unusual activity
    signals = scanner.scan(
        min_volume=500,
        min_oi_ratio=2.0,
        min_premium=50000  # $50K+
    )

    # Print report
    report = scanner.format_report(signals)
    print(report)

    print("Demo complete")
    print("="*70)


if __name__ == "__main__":
    main()
