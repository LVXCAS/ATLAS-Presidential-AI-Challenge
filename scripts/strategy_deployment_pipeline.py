"""
STRATEGY DEPLOYMENT PIPELINE
Automatically deploy R&D discoveries through validation to production

Pipeline Stages:
1. Discovery (R&D) → Parse top strategies from R&D logs
2. Validation (Backtest) → Test on fresh out-of-sample data
3. Paper Trading → Run live for 1 week with fake money
4. Production → Auto-promote if Sharpe > 2.0
"""
import os
import json
import glob
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import yfinance as yf
import numpy as np
import pandas as pd

@dataclass
class StrategyCandidate:
    """R&D discovered strategy"""
    name: str
    strategy_type: str
    market_type: str  # forex, futures, options, stocks
    parameters: Dict
    backtest_sharpe: float
    backtest_win_rate: float
    backtest_total_return: float
    discovery_date: str
    status: str  # discovered, validating, paper_trading, live, rejected

@dataclass
class ValidationResult:
    """Validation backtest results"""
    strategy_name: str
    validation_sharpe: float
    validation_win_rate: float
    validation_return: float
    max_drawdown: float
    total_trades: int
    profitable_trades: int
    avg_trade: float
    validated_at: str
    passed: bool
    rejection_reason: Optional[str]

@dataclass
class PaperTradingStats:
    """Paper trading performance"""
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    ready_for_live: bool
    promotion_reason: Optional[str]

class StrategyDeploymentPipeline:
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Thresholds for auto-promotion
        self.min_validation_sharpe = 1.5
        self.min_paper_sharpe = 2.0
        self.min_win_rate = 0.55
        self.min_trades_for_validation = 30

        # Load pipeline state
        self.pipeline_state = self._load_pipeline_state()

    def _load_pipeline_state(self) -> Dict:
        """Load deployment pipeline state"""
        if os.path.exists('data/pipeline_state.json'):
            with open('data/pipeline_state.json') as f:
                return json.load(f)
        return {
            'discovered': [],
            'validating': [],
            'paper_trading': [],
            'live': [],
            'rejected': []
        }

    def _save_pipeline_state(self):
        """Save pipeline state"""
        os.makedirs('data', exist_ok=True)
        with open('data/pipeline_state.json', 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)

    def send_telegram_notification(self, message: str):
        """Send pipeline notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'STRATEGY PIPELINE\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[PIPELINE] Telegram notification failed: {e}")

    def parse_rd_discoveries(self) -> List[StrategyCandidate]:
        """Parse R&D logs to find new strategy discoveries"""
        print("[PIPELINE] Parsing R&D discoveries...")

        candidates = []

        # Parse Forex/Futures R&D
        forex_futures_logs = glob.glob('logs/forex_futures_strategies_*.json')
        for log_file in sorted(forex_futures_logs, reverse=True)[:5]:  # Last 5 runs
            try:
                with open(log_file) as f:
                    strategies = json.load(f)

                for strategy in strategies:
                    # Skip if already in pipeline
                    if strategy['name'] in [s['name'] for s in self.pipeline_state['discovered']]:
                        continue

                    candidate = StrategyCandidate(
                        name=strategy['name'],
                        strategy_type=strategy['type'],
                        market_type=strategy['market'],
                        parameters=strategy.get('parameters', {}),
                        backtest_sharpe=strategy['expected_sharpe'],
                        backtest_win_rate=strategy.get('win_rate', 0),
                        backtest_total_return=strategy.get('total_return', 0),
                        discovery_date=datetime.now().isoformat(),
                        status='discovered'
                    )

                    # Only add if Sharpe > 1.0 (basic quality filter)
                    if candidate.backtest_sharpe > 1.0:
                        candidates.append(candidate)

            except Exception as e:
                print(f"[PIPELINE] Error parsing {log_file}: {e}")

        # Parse Stock/Options R&D
        stock_options_logs = glob.glob('logs/mega_elite_strategies_*.json')
        for log_file in sorted(stock_options_logs, reverse=True)[:5]:
            try:
                with open(log_file) as f:
                    strategies = json.load(f)

                for strategy in strategies:
                    if strategy['name'] in [s['name'] for s in self.pipeline_state['discovered']]:
                        continue

                    candidate = StrategyCandidate(
                        name=strategy['name'],
                        strategy_type=strategy.get('type', 'unknown'),
                        market_type='stocks/options',
                        parameters=strategy.get('parameters', {}),
                        backtest_sharpe=strategy.get('sharpe', 0),
                        backtest_win_rate=strategy.get('win_rate', 0),
                        backtest_total_return=strategy.get('return', 0),
                        discovery_date=datetime.now().isoformat(),
                        status='discovered'
                    )

                    if candidate.backtest_sharpe > 1.0:
                        candidates.append(candidate)

            except Exception as e:
                print(f"[PIPELINE] Error parsing {log_file}: {e}")

        print(f"[PIPELINE] Found {len(candidates)} new strategy candidates")
        return candidates

    def validate_strategy(self, candidate: StrategyCandidate) -> ValidationResult:
        """Validate strategy on fresh out-of-sample data"""
        print(f"[PIPELINE] Validating {candidate.name}...")

        try:
            # Get fresh data (last 3 months, not used in R&D)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            # Determine symbol based on market type
            if candidate.market_type == 'forex':
                # Use currency ETFs as proxy
                symbols = ['FXE', 'FXY']  # EUR, JPY
            elif candidate.market_type == 'futures':
                # Use index ETFs as proxy
                symbols = ['SPY', 'QQQ']
            else:
                # Use S&P 500 stocks
                symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

            # Backtest on fresh data
            all_trades = []

            for symbol in symbols:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                    if len(data) < 20:
                        continue

                    # Apply strategy based on type
                    if candidate.strategy_type in ['ema_crossover', 'momentum']:
                        trades = self._backtest_ema_strategy(data, candidate.parameters)
                    elif candidate.strategy_type in ['rsi_mean_reversion', 'mean_reversion']:
                        trades = self._backtest_rsi_strategy(data, candidate.parameters)
                    elif candidate.strategy_type == 'breakout':
                        trades = self._backtest_breakout_strategy(data, candidate.parameters)
                    else:
                        trades = []

                    all_trades.extend(trades)

                except Exception as e:
                    print(f"[VALIDATION] Error on {symbol}: {e}")
                    continue

            # Calculate validation metrics
            if len(all_trades) < self.min_trades_for_validation:
                return ValidationResult(
                    strategy_name=candidate.name,
                    validation_sharpe=0,
                    validation_win_rate=0,
                    validation_return=0,
                    max_drawdown=0,
                    total_trades=len(all_trades),
                    profitable_trades=0,
                    avg_trade=0,
                    validated_at=datetime.now().isoformat(),
                    passed=False,
                    rejection_reason=f"Insufficient trades: {len(all_trades)} < {self.min_trades_for_validation}"
                )

            trades_array = np.array(all_trades)
            profitable_trades = len([t for t in all_trades if t > 0])
            win_rate = profitable_trades / len(all_trades)

            # Calculate Sharpe ratio
            if len(trades_array) > 0 and np.std(trades_array) > 0:
                sharpe = np.mean(trades_array) / np.std(trades_array) * np.sqrt(252)
            else:
                sharpe = 0

            # Calculate max drawdown
            cumulative_returns = np.cumsum(trades_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Determine if passed validation
            passed = (
                sharpe >= self.min_validation_sharpe and
                win_rate >= self.min_win_rate and
                max_drawdown < 0.15  # Max 15% drawdown
            )

            rejection_reason = None
            if not passed:
                if sharpe < self.min_validation_sharpe:
                    rejection_reason = f"Low Sharpe: {sharpe:.2f} < {self.min_validation_sharpe}"
                elif win_rate < self.min_win_rate:
                    rejection_reason = f"Low win rate: {win_rate:.1%} < {self.min_win_rate:.1%}"
                elif max_drawdown >= 0.15:
                    rejection_reason = f"High drawdown: {max_drawdown:.1%} >= 15%"

            return ValidationResult(
                strategy_name=candidate.name,
                validation_sharpe=sharpe,
                validation_win_rate=win_rate,
                validation_return=sum(all_trades),
                max_drawdown=max_drawdown,
                total_trades=len(all_trades),
                profitable_trades=profitable_trades,
                avg_trade=np.mean(trades_array),
                validated_at=datetime.now().isoformat(),
                passed=passed,
                rejection_reason=rejection_reason
            )

        except Exception as e:
            print(f"[VALIDATION] Error: {e}")
            return ValidationResult(
                strategy_name=candidate.name,
                validation_sharpe=0,
                validation_win_rate=0,
                validation_return=0,
                max_drawdown=0,
                total_trades=0,
                profitable_trades=0,
                avg_trade=0,
                validated_at=datetime.now().isoformat(),
                passed=False,
                rejection_reason=f"Validation error: {str(e)}"
            )

    def _backtest_ema_strategy(self, data: pd.DataFrame, params: Dict) -> List[float]:
        """Backtest EMA crossover strategy"""
        fast_period = params.get('ema_fast', 10)
        slow_period = params.get('ema_slow', 20)

        data['ema_fast'] = data['Close'].ewm(span=fast_period).mean()
        data['ema_slow'] = data['Close'].ewm(span=slow_period).mean()
        data['signal'] = np.where(data['ema_fast'] > data['ema_slow'], 1, -1)
        data['position'] = data['signal'].shift(1)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'] * data['returns']

        return data['strategy_returns'].dropna().tolist()

    def _backtest_rsi_strategy(self, data: pd.DataFrame, params: Dict) -> List[float]:
        """Backtest RSI mean reversion strategy"""
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('rsi_oversold', 30)
        overbought = params.get('rsi_overbought', 70)

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Generate signals
        data['signal'] = 0
        data.loc[data['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
        data.loc[data['rsi'] > overbought, 'signal'] = -1  # Sell when overbought

        data['position'] = data['signal'].shift(1)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'] * data['returns']

        return data['strategy_returns'].dropna().tolist()

    def _backtest_breakout_strategy(self, data: pd.DataFrame, params: Dict) -> List[float]:
        """Backtest breakout strategy"""
        lookback = params.get('lookback', 20)

        data['high_band'] = data['High'].rolling(window=lookback).max()
        data['low_band'] = data['Low'].rolling(window=lookback).min()

        data['signal'] = 0
        data.loc[data['Close'] > data['high_band'].shift(1), 'signal'] = 1
        data.loc[data['Close'] < data['low_band'].shift(1), 'signal'] = -1

        data['position'] = data['signal'].shift(1)
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['position'] * data['returns']

        return data['strategy_returns'].dropna().tolist()

    def deploy_to_paper_trading(self, candidate: StrategyCandidate, validation: ValidationResult):
        """Deploy strategy to paper trading"""
        print(f"[PIPELINE] Deploying {candidate.name} to paper trading...")

        # Create deployment config
        deployment_config = {
            'strategy_name': candidate.name,
            'strategy_type': candidate.strategy_type,
            'market_type': candidate.market_type,
            'parameters': candidate.parameters,
            'validation_sharpe': validation.validation_sharpe,
            'validation_win_rate': validation.validation_win_rate,
            'deployed_at': datetime.now().isoformat(),
            'deployment_duration_days': 7,
            'status': 'paper_trading'
        }

        # Save deployment config
        os.makedirs('deployments', exist_ok=True)
        config_file = f'deployments/{candidate.name}_paper_config.json'
        with open(config_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)

        print(f"[PIPELINE] Deployment config saved: {config_file}")

        # Send Telegram notification
        msg = f"""
PAPER TRADING DEPLOYMENT

Strategy: {candidate.name}
Type: {candidate.strategy_type}
Market: {candidate.market_type}

Backtest Sharpe: {candidate.backtest_sharpe:.2f}
Validation Sharpe: {validation.validation_sharpe:.2f}
Win Rate: {validation.validation_win_rate:.1%}

Status: Now trading with paper money for 7 days
Next Review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
"""
        self.send_telegram_notification(msg)

        return True

    def check_paper_trading_performance(self, strategy_name: str) -> Optional[PaperTradingStats]:
        """Check paper trading performance and decide if ready for live"""
        print(f"[PIPELINE] Checking paper trading performance: {strategy_name}")

        # Load paper trading logs
        paper_logs = glob.glob(f'deployments/{strategy_name}_paper_trades_*.json')

        if not paper_logs:
            print(f"[PIPELINE] No paper trading logs found for {strategy_name}")
            return None

        all_trades = []
        start_date = None
        end_date = None

        for log_file in paper_logs:
            try:
                with open(log_file) as f:
                    trades = json.load(f)
                    for trade in trades:
                        pnl = trade.get('pnl', 0)
                        all_trades.append(pnl)

                        trade_date = datetime.fromisoformat(trade.get('timestamp', datetime.now().isoformat()))
                        if start_date is None or trade_date < start_date:
                            start_date = trade_date
                        if end_date is None or trade_date > end_date:
                            end_date = trade_date
            except Exception as e:
                print(f"[PIPELINE] Error reading {log_file}: {e}")

        if len(all_trades) < 10:
            print(f"[PIPELINE] Insufficient paper trades: {len(all_trades)} < 10")
            return None

        # Calculate metrics
        trades_array = np.array(all_trades)
        winning_trades = len([t for t in all_trades if t > 0])
        losing_trades = len([t for t in all_trades if t < 0])
        win_rate = winning_trades / len(all_trades)

        sharpe = 0
        if np.std(trades_array) > 0:
            sharpe = np.mean(trades_array) / np.std(trades_array) * np.sqrt(252)

        # Calculate drawdown
        cumulative = np.cumsum(trades_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Determine if ready for live
        ready_for_live = (
            sharpe >= self.min_paper_sharpe and
            win_rate >= self.min_win_rate and
            max_drawdown < 0.10  # Stricter for live: 10% max
        )

        promotion_reason = None
        if ready_for_live:
            promotion_reason = f"Sharpe {sharpe:.2f} > {self.min_paper_sharpe}, WR {win_rate:.1%} > {self.min_win_rate:.1%}"

        return PaperTradingStats(
            strategy_name=strategy_name,
            start_date=start_date.isoformat() if start_date else '',
            end_date=end_date.isoformat() if end_date else '',
            total_trades=len(all_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=sum(all_trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            ready_for_live=ready_for_live,
            promotion_reason=promotion_reason
        )

    def promote_to_live(self, strategy_name: str, paper_stats: PaperTradingStats):
        """Promote strategy from paper to live trading"""
        print(f"[PIPELINE] PROMOTING {strategy_name} TO LIVE TRADING!")

        # Send urgent Telegram notification
        msg = f"""
STRATEGY PROMOTED TO LIVE!

{strategy_name}

Paper Trading Results:
- Total Trades: {paper_stats.total_trades}
- Win Rate: {paper_stats.win_rate:.1%}
- Sharpe Ratio: {paper_stats.sharpe_ratio:.2f}
- Total P&L (paper): ${paper_stats.total_pnl:,.2f}
- Max Drawdown: {paper_stats.max_drawdown:.1%}

Reason: {paper_stats.promotion_reason}

Status: NOW TRADING WITH REAL MONEY!

Use /stop to pause if needed
"""
        self.send_telegram_notification(msg)

        # Update pipeline state
        self.pipeline_state['live'].append({
            'strategy_name': strategy_name,
            'promoted_at': datetime.now().isoformat(),
            'paper_sharpe': paper_stats.sharpe_ratio,
            'paper_win_rate': paper_stats.win_rate
        })
        self._save_pipeline_state()

        print(f"[PIPELINE] {strategy_name} is now LIVE!")
        return True

    def run_full_pipeline(self):
        """Run complete deployment pipeline"""
        print("\n" + "="*70)
        print("STRATEGY DEPLOYMENT PIPELINE - FULL RUN")
        print("="*70 + "\n")

        # Stage 1: Discover new strategies
        candidates = self.parse_rd_discoveries()

        if not candidates:
            print("[PIPELINE] No new strategies discovered")
            return

        # Stage 2: Validate top 3 candidates
        print(f"\n[PIPELINE] Validating top 3 candidates...")
        top_candidates = sorted(candidates, key=lambda x: x.backtest_sharpe, reverse=True)[:3]

        for candidate in top_candidates:
            print(f"\n{'='*70}")
            print(f"VALIDATING: {candidate.name}")
            print(f"{'='*70}")

            validation = self.validate_strategy(candidate)

            print(f"\nValidation Results:")
            print(f"  Sharpe: {validation.validation_sharpe:.2f}")
            print(f"  Win Rate: {validation.validation_win_rate:.1%}")
            print(f"  Trades: {validation.total_trades}")
            print(f"  Passed: {validation.passed}")

            if validation.passed:
                # Stage 3: Deploy to paper trading
                self.deploy_to_paper_trading(candidate, validation)

                # Update pipeline state
                self.pipeline_state['paper_trading'].append(asdict(candidate))
                self._save_pipeline_state()
            else:
                print(f"  REJECTED: {validation.rejection_reason}")
                self.pipeline_state['rejected'].append(asdict(candidate))
                self._save_pipeline_state()

        # Stage 4: Check existing paper trading strategies
        print(f"\n[PIPELINE] Checking paper trading strategies for promotion...")
        for strategy_data in self.pipeline_state['paper_trading']:
            paper_stats = self.check_paper_trading_performance(strategy_data['name'])

            if paper_stats and paper_stats.ready_for_live:
                self.promote_to_live(strategy_data['name'], paper_stats)

        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}\n")

def main():
    """Run the strategy deployment pipeline"""
    pipeline = StrategyDeploymentPipeline()
    pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()
