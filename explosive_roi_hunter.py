#!/usr/bin/env python3
"""
EXPLOSIVE ROI HUNTER
24/7 system hunting for Intel/SNAP/LYFT/RIVN-style explosive moves
Focus on setups BEFORE they explode, not after
Target: 50-200%+ ROI opportunities
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

class ExplosiveROIHunter:
    """Hunts for explosive ROI opportunities like Intel puts success"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Focus on explosive momentum candidates
        self.explosive_candidates = [
            # Recent explosive movers (patterns repeat)
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',  # EV momentum
            'SNAP', 'META', 'PINS', 'TWTR', 'RBLX',  # Social/tech volatile
            'INTC', 'AMD', 'NVDA', 'AVGO', 'MU',  # Chip momentum

            # High beta momentum plays
            'PLTR', 'COIN', 'ROKU', 'NFLX', 'UBER', 'LYFT',
            'SQ', 'PYPL', 'SHOP', 'SNOW', 'NET', 'CRWD',

            # Meme/momentum potential
            'AMC', 'GME', 'BBBY', 'CLOV', 'SOFI', 'WISH',

            # Options with explosive potential
            'SPY', 'QQQ', 'IWM', 'TSLA', 'AAPL', 'MSFT',

            # Biotech/catalyst plays
            'MRNA', 'PFE', 'JNJ', 'GILD', 'REGN', 'BIIB'
        ]

    async def scan_for_explosive_setups(self):
        """Scan for Intel-puts-style explosive setup patterns"""

        print("EXPLOSIVE ROI HUNTER - ACTIVE SCANNING")
        print("=" * 60)
        print("Hunting for Intel/SNAP/LYFT/RIVN-style explosive moves")
        print("Focus: Identify BEFORE explosion, not after")
        print("=" * 60)

        explosive_setups = []
        scan_time = datetime.now()

        print(f"\n=== SCANNING {len(self.explosive_candidates)} EXPLOSIVE CANDIDATES ===")

        for symbol in self.explosive_candidates:
            try:
                # Get recent price action
                bars = self.alpaca.get_bars(symbol, '1Day', limit=10).df
                if len(bars) < 5:
                    continue

                current_price = float(bars.iloc[-1]['close'])

                # Calculate explosive potential indicators
                volatility_score = self.calculate_volatility_score(bars)
                momentum_score = self.calculate_momentum_score(bars)
                options_potential = self.estimate_options_explosive_potential(symbol, current_price)

                # Intel-puts-style setup criteria
                if (volatility_score > 0.15 or  # High volatility
                    momentum_score > 0.8 or     # Strong momentum building
                    options_potential > 1.5):    # High options leverage potential

                    setup = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'volatility_score': volatility_score,
                        'momentum_score': momentum_score,
                        'options_potential': options_potential,
                        'explosive_score': volatility_score + momentum_score + (options_potential * 0.5),
                        'setup_type': self.classify_explosive_setup(symbol, volatility_score, momentum_score),
                        'roi_potential': self.estimate_roi_potential(volatility_score, momentum_score, options_potential),
                        'scan_time': scan_time.isoformat()
                    }

                    explosive_setups.append(setup)

            except Exception as e:
                continue

        # Sort by explosive potential
        explosive_setups.sort(key=lambda x: x['explosive_score'], reverse=True)

        # Display top explosive opportunities
        print(f"\n=== TOP EXPLOSIVE SETUPS (Intel-puts-style) ===")
        print("Symbol | Price | Vol | Mom | OptPot | ROI Est | Setup Type")
        print("-" * 70)

        for setup in explosive_setups[:10]:
            print(f"{setup['symbol']:>6} | ${setup['current_price']:>5.2f} | "
                  f"{setup['volatility_score']:>4.2f} | {setup['momentum_score']:>4.2f} | "
                  f"{setup['options_potential']:>4.1f}x | {setup['roi_potential']:>6.0%} | "
                  f"{setup['setup_type']}")

        return explosive_setups

    def calculate_volatility_score(self, bars):
        """Calculate volatility score for explosive potential"""
        daily_returns = bars['close'].pct_change().dropna()
        return float(daily_returns.std() * 15.81)  # Annualized volatility

    def calculate_momentum_score(self, bars):
        """Calculate momentum building score"""
        if len(bars) < 5:
            return 0

        # Price momentum
        price_momentum = (bars['close'].iloc[-1] - bars['close'].iloc[-5]) / bars['close'].iloc[-5]

        # Volume momentum
        volume_momentum = (bars['volume'].iloc[-3:].mean() - bars['volume'].iloc[-10:-3].mean()) / bars['volume'].iloc[-10:-3].mean()

        return abs(price_momentum) + (volume_momentum * 0.3)

    def estimate_options_explosive_potential(self, symbol, current_price):
        """Estimate options leverage potential for explosive moves"""
        # High volatility stocks have higher options leverage
        vol_multiplier = {
            'RIVN': 3.0, 'SNAP': 2.5, 'PLTR': 2.8, 'COIN': 3.2,
            'TESLA': 2.2, 'AMD': 2.0, 'NVDA': 1.8, 'SPY': 1.5
        }

        return vol_multiplier.get(symbol, 2.0)  # Default 2x leverage estimate

    def classify_explosive_setup(self, symbol, volatility, momentum):
        """Classify the type of explosive setup"""
        if volatility > 0.25 and momentum > 1.0:
            return "BREAKOUT_READY"
        elif volatility > 0.20:
            return "HIGH_VOLATILITY"
        elif momentum > 0.8:
            return "MOMENTUM_BUILD"
        else:
            return "WATCH_LIST"

    def estimate_roi_potential(self, volatility, momentum, options_leverage):
        """Estimate ROI potential like Intel puts (+70.6%)"""
        base_potential = (volatility + momentum) * options_leverage

        # Intel puts achieved 70.6%, RIVN puts 89.8%
        # Scale to realistic explosive potential
        roi_estimate = min(base_potential * 0.8, 2.0)  # Cap at 200%

        return roi_estimate

    async def generate_explosive_trading_signals(self, explosive_setups):
        """Generate trading signals for explosive opportunities"""

        if not explosive_setups:
            print("No explosive setups identified currently")
            return []

        print(f"\n=== EXPLOSIVE TRADING SIGNALS ===")
        print("Intel-puts-style high-ROI opportunities")
        print("-" * 50)

        signals = []
        top_setups = explosive_setups[:5]  # Top 5 explosive opportunities

        for setup in top_setups:
            symbol = setup['symbol']
            roi_potential = setup['roi_potential']
            setup_type = setup['setup_type']

            # Determine strategy based on setup
            if setup_type == "BREAKOUT_READY":
                strategy = "MOMENTUM_CALLS"
                allocation = 0.15  # 15% for highest conviction
            elif setup_type == "HIGH_VOLATILITY":
                strategy = "STRADDLE_PLAY"
                allocation = 0.10  # 10% for volatility plays
            elif setup_type == "MOMENTUM_BUILD":
                strategy = "MOMENTUM_CALLS"
                allocation = 0.08  # 8% for building momentum
            else:
                strategy = "WATCH"
                allocation = 0.05  # 5% for watch list

            signal = {
                'symbol': symbol,
                'strategy': strategy,
                'roi_potential': roi_potential,
                'allocation': allocation,
                'setup_type': setup_type,
                'explosive_score': setup['explosive_score'],
                'rationale': f"{setup_type} setup with {roi_potential:.0%} ROI potential"
            }

            signals.append(signal)

            print(f"{symbol:>6} | {strategy:>15} | {roi_potential:>6.0%} | {allocation:>5.1%} | {setup['setup_type']}")

        return signals

    async def run_explosive_hunter(self):
        """Run continuous explosive ROI hunting"""

        print("EXPLOSIVE ROI HUNTER")
        print("=" * 80)
        print("24/7 hunting for Intel/SNAP/LYFT/RIVN-style explosive moves")
        print("Target: Identify setups BEFORE they explode")
        print("=" * 80)

        # Scan for explosive setups
        explosive_setups = await self.scan_for_explosive_setups()

        if not explosive_setups:
            print("No explosive setups detected - continuing hunt")
            return []

        # Generate trading signals
        signals = await self.generate_explosive_trading_signals(explosive_setups)

        print("=" * 80)
        print("EXPLOSIVE HUNT COMPLETE")
        print(f"Found {len(explosive_setups)} potential setups")
        print(f"Generated {len(signals)} high-ROI signals")
        print("Continuing 24/7 hunt for next explosive opportunity...")
        print("=" * 80)

        # Save results
        hunt_results = {
            'hunt_timestamp': datetime.now().isoformat(),
            'explosive_setups': explosive_setups,
            'trading_signals': signals,
            'total_opportunities': len(explosive_setups),
            'target_roi': '50-200%'
        }

        filename = f'explosive_hunt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(hunt_results, f, indent=2)

        return signals

async def main():
    """Run explosive ROI hunter"""
    hunter = ExplosiveROIHunter()
    signals = await hunter.run_explosive_hunter()
    return signals

if __name__ == "__main__":
    asyncio.run(main())