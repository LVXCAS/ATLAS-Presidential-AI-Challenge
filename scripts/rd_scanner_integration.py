"""
R&D → Scanner Integration Layer

Connects the autonomous R&D system to the Week 1 continuous scanner:
1. R&D discovers and validates strategies (overnight/continuous)
2. Integration layer formats discoveries for scanner consumption
3. Scanner uses R&D insights to enhance opportunity scoring
4. Performance feedback loops back to R&D for learning

This creates a closed-loop autonomous trading empire.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import glob

class RDScannerBridge:
    """Bridge between R&D system and continuous scanner"""

    def __init__(self):
        self.rd_strategies_dir = "."
        self.latest_rd_file = None
        self.active_strategies = []
        print("[BRIDGE] R&D Scanner Bridge initialized")

    def load_latest_rd_discoveries(self) -> Optional[Dict[str, Any]]:
        """Load the most recent R&D validated strategies"""

        # Find all R&D validation files
        rd_files = glob.glob(os.path.join(self.rd_strategies_dir, "rd_validated_strategies_*.json"))

        if not rd_files:
            print("[BRIDGE] No R&D discovery files found")
            return None

        # Get most recent
        latest_file = max(rd_files, key=os.path.getctime)
        self.latest_rd_file = latest_file

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)

            print(f"[BRIDGE] Loaded R&D discoveries from {os.path.basename(latest_file)}")
            print(f"  Validated strategies: {data['validated_strategies']}")
            print(f"  Ready for deployment: {data['ready_for_deployment']}")

            self.active_strategies = data['strategies']
            return data

        except Exception as e:
            print(f"[BRIDGE] Error loading R&D file: {e}")
            return None

    def enhance_scanner_scoring(self, symbol: str, base_score: float) -> float:
        """Enhance scanner's opportunity score with R&D insights"""

        if not self.active_strategies:
            return base_score

        # Check if R&D has validated this symbol
        rd_boost = 0.0

        for strategy in self.active_strategies:
            if strategy['symbol'] == symbol:
                strategy_type = strategy['type']

                if strategy_type == 'momentum':
                    # Momentum strategies get boost based on historical return
                    historical_return = strategy.get('historical_return', 0)
                    if historical_return > 0.3:  # 30%+ historical return
                        rd_boost += 0.5  # Significant boost
                    elif historical_return > 0.1:
                        rd_boost += 0.3

                elif strategy_type == 'volatility':
                    # Volatility strategies good for options
                    vol_percentile = strategy.get('vol_percentile', 0)
                    if vol_percentile > 70:  # High volatility
                        rd_boost += 0.4

                print(f"[BRIDGE] {symbol}: R&D boost +{rd_boost:.2f} ({strategy_type})")

        enhanced_score = min(base_score + rd_boost, 5.0)  # Cap at 5.0
        return enhanced_score

    def get_rd_validated_symbols(self) -> List[str]:
        """Get list of symbols validated by R&D"""
        return list(set(s['symbol'] for s in self.active_strategies))

    def get_strategy_details(self, symbol: str) -> List[Dict[str, Any]]:
        """Get R&D strategy details for a specific symbol"""
        return [s for s in self.active_strategies if s['symbol'] == symbol]

    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate report of R&D strategies ready for scanner deployment"""

        if not self.active_strategies:
            return {'status': 'no_strategies', 'count': 0}

        # Categorize by type
        momentum_strategies = [s for s in self.active_strategies if s['type'] == 'momentum']
        volatility_strategies = [s for s in self.active_strategies if s['type'] == 'volatility']

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(self.active_strategies),
            'momentum_strategies': len(momentum_strategies),
            'volatility_strategies': len(volatility_strategies),
            'symbols': list(set(s['symbol'] for s in self.active_strategies)),
            'deployment_ready': True,
            'strategies_by_symbol': {}
        }

        # Group by symbol
        for strategy in self.active_strategies:
            symbol = strategy['symbol']
            if symbol not in report['strategies_by_symbol']:
                report['strategies_by_symbol'][symbol] = []
            report['strategies_by_symbol'][symbol].append({
                'type': strategy['type'],
                'score': strategy.get('momentum_score') or strategy.get('realized_vol', 0),
                'live_price': strategy['live_validation']['live_price']
            })

        return report

def demonstrate_integration():
    """Demonstrate how R&D integrates with scanner"""

    print("="*70)
    print("R&D <--> SCANNER INTEGRATION DEMONSTRATION")
    print("="*70)

    bridge = RDScannerBridge()

    # Load latest R&D discoveries
    rd_data = bridge.load_latest_rd_discoveries()

    if not rd_data:
        print("\n[INFO] No R&D discoveries yet. Run hybrid_rd_system.py first.")
        return

    print("\n" + "="*70)
    print("SCANNER ENHANCEMENT SIMULATION")
    print("="*70)

    # Simulate scanner finding opportunities
    test_symbols = ['INTC', 'AMD', 'NVDA', 'AAPL', 'TSLA']

    for symbol in test_symbols:
        # Simulate base score from scanner
        base_score = 4.0  # Below Week 1 threshold

        # Enhance with R&D insights
        enhanced_score = bridge.enhance_scanner_scoring(symbol, base_score)

        threshold_status = "QUALIFIED" if enhanced_score >= 4.5 else "below threshold"

        print(f"\n{symbol}:")
        print(f"  Scanner base score: {base_score:.2f}")
        print(f"  R&D enhanced score: {enhanced_score:.2f}")
        print(f"  Status: {threshold_status}")

        if enhanced_score >= 4.5:
            # Show R&D strategy details
            strategies = bridge.get_strategy_details(symbol)
            for strategy in strategies:
                print(f"  R&D Strategy: {strategy['type']}")
                if 'historical_return' in strategy:
                    print(f"    Historical return: {strategy['historical_return']:.1%}")

    # Generate deployment report
    print("\n" + "="*70)
    print("DEPLOYMENT REPORT")
    print("="*70)

    report = bridge.generate_deployment_report()
    print(f"Total strategies ready: {report['total_strategies']}")
    print(f"Symbols: {', '.join(report['symbols'])}")
    print(f"\nStrategies by symbol:")
    for symbol, strategies in report['strategies_by_symbol'].items():
        print(f"  {symbol}: {', '.join(s['type'] for s in strategies)}")

    # Save integration report
    filename = f"rd_scanner_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nIntegration report saved: {filename}")

    print("\n" + "="*70)
    print("INTEGRATION STATUS: OPERATIONAL")
    print("="*70)
    print("\nHow it works:")
    print("1. R&D system researches & validates strategies (hybrid_rd_system.py)")
    print("2. Bridge loads validated strategies")
    print("3. Scanner queries bridge for R&D insights")
    print("4. Opportunities get boosted scores if R&D validated them")
    print("5. More strategies pass Week 1 threshold (4.5+) with R&D backing")

if __name__ == "__main__":
    demonstrate_integration()
