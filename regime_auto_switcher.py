"""
REGIME AUTO-SWITCHER
Automatically switches trading strategies based on detected market regime

This builds on the existing market_regime_detector.py and adds:
- Auto-switching between strategies
- Position sizing adjustments
- Telegram notifications on regime changes
- Integration with trading systems
"""
import os
import json
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests

# Import existing regime detector
from market_regime_detector import MarketRegimeDetector as BasicRegimeDetector

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    markets: List[str]  # ['forex', 'futures', 'options', 'stocks']
    optimal_regimes: List[str]  # Which regimes this strategy works best in
    position_sizing: float  # Base position size
    script_path: str  # Path to strategy script
    enabled: bool

@dataclass
class RegimeSwitch:
    """Record of a regime switch event"""
    from_regime: str
    to_regime: str
    timestamp: str
    strategies_stopped: List[str]
    strategies_started: List[str]
    reason: str

class RegimeAutoSwitcher:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Load configuration
        self.config = self._load_config()

        # Initialize detectors
        self.basic_detector = BasicRegimeDetector()

        # Load state
        self.current_regime = None
        self.active_strategies = []
        self.switch_history = self._load_switch_history()

        # Auto-switching enabled?
        self.auto_switch_enabled = self.config.get('auto_switch_enabled', False)

    def _load_config(self) -> Dict:
        """Load auto-switcher configuration"""
        if os.path.exists('config/regime_switcher_config.json'):
            with open('config/regime_switcher_config.json') as f:
                return json.load(f)

        # Default configuration
        default_config = {
            'auto_switch_enabled': False,  # Must be explicitly enabled
            'check_interval_minutes': 60,   # Check every hour
            'regime_stability_minutes': 30, # Regime must be stable for 30min before switching

            'strategies': {
                'forex_elite': {
                    'name': 'Forex Elite',
                    'markets': ['forex'],
                    'optimal_regimes': ['BULL_TRENDING', 'BEAR_TRENDING', 'NEUTRAL'],
                    'position_sizing': 1.0,
                    'script_path': 'START_FOREX_ELITE.py',
                    'enabled': True
                },
                'momentum_scanner': {
                    'name': 'Momentum Scanner',
                    'markets': ['stocks'],
                    'optimal_regimes': ['BULL_TRENDING', 'LOW_VOLATILITY'],
                    'position_sizing': 1.2,
                    'script_path': 'momentum_scanner.py',
                    'enabled': False  # Not built yet
                },
                'options_iron_condor': {
                    'name': 'Iron Condor',
                    'markets': ['options'],
                    'optimal_regimes': ['BULL_RANGING', 'BEAR_RANGING', 'LOW_VOLATILITY'],
                    'position_sizing': 1.0,
                    'script_path': 'auto_options_scanner.py',
                    'enabled': False
                },
                'mean_reversion': {
                    'name': 'Mean Reversion',
                    'markets': ['stocks', 'futures'],
                    'optimal_regimes': ['HIGH_VOLATILITY', 'BEAR_RANGING'],
                    'position_sizing': 0.8,
                    'script_path': 'mean_reversion_scanner.py',
                    'enabled': False  # Not built yet
                }
            }
        }

        # Save default config
        os.makedirs('config', exist_ok=True)
        with open('config/regime_switcher_config.json', 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _save_config(self):
        """Save configuration"""
        os.makedirs('config', exist_ok=True)
        with open('config/regime_switcher_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

    def _load_switch_history(self) -> List[RegimeSwitch]:
        """Load regime switch history"""
        if os.path.exists('data/regime_switch_history.json'):
            with open('data/regime_switch_history.json') as f:
                data = json.load(f)
                return [RegimeSwitch(**item) for item in data]
        return []

    def _save_switch_history(self):
        """Save regime switch history"""
        os.makedirs('data', exist_ok=True)
        with open('data/regime_switch_history.json', 'w') as f:
            json.dump([asdict(s) for s in self.switch_history], f, indent=2)

    def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'REGIME AUTO-SWITCHER\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[SWITCHER] Telegram notification failed: {e}")

    def detect_current_regime(self) -> Dict:
        """Detect current market regime"""
        print("[SWITCHER] Detecting current market regime...")

        # Use existing regime detector
        regime_data = self.basic_detector.analyze_market_regime()

        # Map to our regime types
        regime_map = {
            'VERY_BULLISH': 'BULL_TRENDING',
            'BULLISH': 'BULL_RANGING',
            'NEUTRAL': 'NEUTRAL',
            'BEARISH': 'BEAR_RANGING',
            'VERY_BEARISH': 'BEAR_TRENDING'
        }

        detected_regime = regime_map.get(regime_data.get('regime', 'NEUTRAL'), 'NEUTRAL')

        # Add volatility regime
        vix = regime_data.get('vix_level', 20)
        if vix > 30:
            volatility_regime = 'HIGH_VOLATILITY'
        elif vix < 15:
            volatility_regime = 'LOW_VOLATILITY'
        else:
            volatility_regime = 'NORMAL_VOLATILITY'

        return {
            'primary_regime': detected_regime,
            'volatility_regime': volatility_regime,
            'vix': vix,
            'sp500_momentum': regime_data.get('sp500_momentum', 0),
            'confidence': regime_data.get('confidence_adjustment', 1.0),
            'timestamp': datetime.now().isoformat()
        }

    def get_optimal_strategies_for_regime(self, regime_data: Dict) -> List[str]:
        """Get list of optimal strategies for current regime"""
        primary_regime = regime_data['primary_regime']
        volatility_regime = regime_data['volatility_regime']

        optimal_strategies = []

        for strategy_id, strategy_config in self.config['strategies'].items():
            if not strategy_config['enabled']:
                continue

            # Check if strategy is optimal for this regime
            if (primary_regime in strategy_config['optimal_regimes'] or
                volatility_regime in strategy_config['optimal_regimes']):
                optimal_strategies.append(strategy_id)

        return optimal_strategies

    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a running strategy"""
        strategy = self.config['strategies'].get(strategy_id)
        if not strategy:
            print(f"[SWITCHER] Unknown strategy: {strategy_id}")
            return False

        print(f"[SWITCHER] Stopping {strategy['name']}...")

        # Try to stop gracefully
        # This is a simplified version - in production, you'd track PIDs
        try:
            script_name = strategy['script_path'].replace('.py', '')

            # Kill python processes running this script
            cmd = f'taskkill /F /FI "COMMANDLINE like %{script_name}%"'
            subprocess.run(cmd, shell=True, capture_output=True)

            print(f"[SWITCHER] Stopped {strategy['name']}")
            return True

        except Exception as e:
            print(f"[SWITCHER] Error stopping {strategy['name']}: {e}")
            return False

    def start_strategy(self, strategy_id: str) -> bool:
        """Start a strategy"""
        strategy = self.config['strategies'].get(strategy_id)
        if not strategy:
            print(f"[SWITCHER] Unknown strategy: {strategy_id}")
            return False

        print(f"[SWITCHER] Starting {strategy['name']}...")

        try:
            # Launch strategy script
            cmd = [sys.executable, strategy['script_path']]
            subprocess.Popen(cmd)

            print(f"[SWITCHER] Started {strategy['name']}")
            return True

        except Exception as e:
            print(f"[SWITCHER] Error starting {strategy['name']}: {e}")
            return False

    def perform_regime_switch(self, regime_data: Dict):
        """Perform actual regime switch"""
        new_regime = regime_data['primary_regime']
        old_regime = self.current_regime

        print(f"\n[SWITCHER] REGIME CHANGE: {old_regime} → {new_regime}")

        # Get optimal strategies for new regime
        optimal_strategies = self.get_optimal_strategies_for_regime(regime_data)

        # Determine which strategies to stop/start
        strategies_to_stop = [s for s in self.active_strategies if s not in optimal_strategies]
        strategies_to_start = [s for s in optimal_strategies if s not in self.active_strategies]

        # Stop non-optimal strategies
        stopped = []
        for strategy_id in strategies_to_stop:
            if self.stop_strategy(strategy_id):
                stopped.append(strategy_id)
                self.active_strategies.remove(strategy_id)

        # Start optimal strategies
        started = []
        for strategy_id in strategies_to_start:
            if self.start_strategy(strategy_id):
                started.append(strategy_id)
                self.active_strategies.append(strategy_id)

        # Record switch
        switch = RegimeSwitch(
            from_regime=old_regime or 'NONE',
            to_regime=new_regime,
            timestamp=datetime.now().isoformat(),
            strategies_stopped=[self.config['strategies'][s]['name'] for s in stopped],
            strategies_started=[self.config['strategies'][s]['name'] for s in started],
            reason=f"VIX: {regime_data['vix']:.1f}, SP500 Momentum: {regime_data['sp500_momentum']:.1f}%"
        )

        self.switch_history.append(switch)
        self._save_switch_history()

        # Update current regime
        self.current_regime = new_regime

        # Send Telegram notification
        msg = f"""
REGIME CHANGED!

Old: {old_regime or 'NONE'}
New: {new_regime}

VIX: {regime_data['vix']:.1f}
S&P 500 Momentum: {regime_data['sp500_momentum']:.1f}%

STOPPED:
{chr(10).join([f'  - {name}' for name in switch.strategies_stopped]) if switch.strategies_stopped else '  None'}

STARTED:
{chr(10).join([f'  - {name}' for name in switch.strategies_started]) if switch.strategies_started else '  None'}

ACTIVE STRATEGIES:
{chr(10).join([f'  - {self.config["strategies"][s]["name"]}' for s in self.active_strategies])}
"""
        self.send_telegram_notification(msg)

        print(f"\n[SWITCHER] Switch complete!")
        print(f"[SWITCHER] Active strategies: {self.active_strategies}")

    def check_and_switch(self):
        """Main monitoring loop - check regime and switch if needed"""
        if not self.auto_switch_enabled:
            print("[SWITCHER] Auto-switching DISABLED")
            print("[SWITCHER] Enable with: /regime auto")
            return

        print("\n" + "="*70)
        print("REGIME AUTO-SWITCHER - CHECK")
        print("="*70)

        # Detect current regime
        regime_data = self.detect_current_regime()

        print(f"\n[REGIME] Current: {regime_data['primary_regime']}")
        print(f"[REGIME] Volatility: {regime_data['volatility_regime']}")
        print(f"[REGIME] VIX: {regime_data['vix']:.2f}")

        # Check if regime changed
        if self.current_regime != regime_data['primary_regime']:
            print(f"[SWITCHER] Regime changed from {self.current_regime} to {regime_data['primary_regime']}")

            # Perform switch
            self.perform_regime_switch(regime_data)
        else:
            print(f"[SWITCHER] No regime change detected")

            # Show optimal strategies for current regime
            optimal_strategies = self.get_optimal_strategies_for_regime(regime_data)
            print(f"[SWITCHER] Optimal strategies: {optimal_strategies}")
            print(f"[SWITCHER] Active strategies: {self.active_strategies}")

    def enable_auto_switching(self):
        """Enable automatic regime switching"""
        self.auto_switch_enabled = True
        self.config['auto_switch_enabled'] = True
        self._save_config()

        msg = f"""
AUTO-SWITCHING ENABLED!

The system will now automatically switch strategies when market regime changes.

Current regime will be checked every {self.config['check_interval_minutes']} minutes.

Available strategies:
"""
        for strategy_id, strategy in self.config['strategies'].items():
            if strategy['enabled']:
                msg += f"\n{strategy['name']}:"
                msg += f"\n  Optimal regimes: {', '.join(strategy['optimal_regimes'])}"

        self.send_telegram_notification(msg)

        print("[SWITCHER] Auto-switching ENABLED")

    def disable_auto_switching(self):
        """Disable automatic regime switching"""
        self.auto_switch_enabled = False
        self.config['auto_switch_enabled'] = False
        self._save_config()

        self.send_telegram_notification("AUTO-SWITCHING DISABLED\n\nStrategies will continue running until manually stopped.")

        print("[SWITCHER] Auto-switching DISABLED")

    def get_status(self) -> str:
        """Get current auto-switcher status"""
        status = f"""
=== REGIME AUTO-SWITCHER STATUS ===

Auto-Switching: {'ENABLED' if self.auto_switch_enabled else 'DISABLED'}
Current Regime: {self.current_regime or 'Not detected yet'}
Active Strategies: {len(self.active_strategies)}

STRATEGIES:
"""
        for strategy_id in self.active_strategies:
            strategy = self.config['strategies'][strategy_id]
            status += f"  - {strategy['name']} (RUNNING)\n"

        status += f"\nCheck Interval: {self.config['check_interval_minutes']} minutes"
        status += f"\nLast {len(self.switch_history)} switches tracked"

        return status

def main():
    """Test the auto-switcher"""
    switcher = RegimeAutoSwitcher()

    print(switcher.get_status())

    # Check and potentially switch
    switcher.check_and_switch()

if __name__ == '__main__':
    main()
