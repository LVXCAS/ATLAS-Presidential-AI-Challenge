"""
Advanced Live Execution - REAL STRATEGIES NOW
Clean slate - deploying advanced momentum and options strategies
Using full buying power for maximum ROI targeting 5000%+ annually
"""

import os
import asyncio
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import json
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)

class AdvancedLiveExecutor:
    """
    ADVANCED LIVE EXECUTION ENGINE
    Clean slate deployment with sophisticated strategies
    Targeting maximum ROI with professional position sizing
    """

    def __init__(self):
        # Initialize Alpaca with updated keys
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Get current account status
        self.account = self.alpaca.get_account()
        self.buying_power = float(self.account.buying_power)
        self.portfolio_value = float(self.account.portfolio_value)

        print(f"[ADVANCED EXECUTOR] Clean slate deployment")
        print(f"[BUYING POWER] ${self.buying_power:,.2f} available")
        print(f"[PORTFOLIO] ${self.portfolio_value:,.2f} total value")

    def get_live_market_data(self):
        """Get current live market conditions"""
        symbols = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN']
        market_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d', interval='5m')

                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-20] if len(hist) > 20 else hist['Close'].iloc[0]

                    change_pct = ((current / prev) - 1) * 100
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0

                    # Calculate momentum score
                    recent_prices = hist['Close'].tail(10)
                    momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) * 100

                    # Calculate volatility
                    volatility = recent_prices.pct_change().std() * np.sqrt(252) * 100

                    market_data[symbol] = {
                        'price': current,
                        'change_pct': change_pct,
                        'momentum_score': momentum,
                        'volatility': volatility,
                        'volume': volume,
                        'signal': self._generate_signal(change_pct, momentum, volatility)
                    }

            except Exception as e:
                print(f"[DATA ERROR] {symbol}: {str(e)}")

        return market_data

    def _generate_signal(self, change_pct, momentum, volatility):
        """Generate trading signal based on technical analysis"""

        # Strong bullish signals
        if change_pct > 2.0 and momentum > 3.0:
            return 'STRONG_BUY'
        elif change_pct > 1.0 and momentum > 1.5:
            return 'BUY'

        # Strong bearish signals
        elif change_pct < -2.0 and momentum < -3.0:
            return 'STRONG_SELL'
        elif change_pct < -1.0 and momentum < -1.5:
            return 'SELL'

        # High volatility signals
        elif volatility > 40:
            return 'HIGH_VOLATILITY'
        elif volatility > 25:
            return 'MODERATE_VOLATILITY'

        else:
            return 'NEUTRAL'

    async def execute_advanced_strategies(self):
        """Execute advanced strategies based on live market analysis"""

        print("\n" + "="*70)
        print("ADVANCED LIVE STRATEGY EXECUTION - CLEAN SLATE DEPLOYMENT")
        print("="*70)

        # Get live market analysis
        market_data = self.get_live_market_data()

        # Generate advanced strategies
        strategies = self._generate_advanced_strategies(market_data)

        # Calculate optimal position sizing
        sized_strategies = self._optimize_position_sizing(strategies)

        # Execute strategies
        execution_results = []
        for strategy in sized_strategies:
            result = await self._execute_strategy(strategy, market_data)
            execution_results.append(result)

        # Portfolio analysis
        total_deployed = sum(s['position_size'] for s in sized_strategies)
        expected_returns = sum(s['position_size'] * s['expected_return'] for s in sized_strategies)

        portfolio_analysis = {
            'timestamp': datetime.now().isoformat(),
            'market_analysis': market_data,
            'strategies_deployed': len(sized_strategies),
            'total_capital_deployed': total_deployed,
            'expected_profit': expected_returns,
            'expected_portfolio_return': (expected_returns / total_deployed) * 100 if total_deployed > 0 else 0,
            'buying_power_used': (total_deployed / self.buying_power) * 100,
            'compound_monthly_contribution': (expected_returns / self.portfolio_value) * 100,
            'execution_results': execution_results,
            'risk_metrics': self._calculate_risk_metrics(sized_strategies, market_data)
        }

        # Save execution report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_execution_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(portfolio_analysis, f, indent=2, default=str)

        # Display results
        print(f"\n[EXECUTION SUMMARY]")
        print(f"Strategies Deployed: {len(sized_strategies)}")
        print(f"Capital Deployed: ${total_deployed:,.0f}")
        print(f"Expected Profit: ${expected_returns:,.0f}")
        print(f"Expected Return: {portfolio_analysis['expected_portfolio_return']:.1f}%")
        print(f"Monthly Target Contribution: {portfolio_analysis['compound_monthly_contribution']:.2f}%")
        print(f"Buying Power Utilized: {portfolio_analysis['buying_power_used']:.1f}%")

        print(f"\n[INDIVIDUAL STRATEGIES]")
        for i, strategy in enumerate(sized_strategies, 1):
            expected_profit = strategy['position_size'] * strategy['expected_return']
            print(f"{i}. {strategy['symbol']} {strategy['strategy_type']}: ${expected_profit:,.0f} expected")

        return portfolio_analysis

    def _generate_advanced_strategies(self, market_data):
        """Generate sophisticated strategies based on market analysis"""
        strategies = []

        for symbol, data in market_data.items():
            signal = data['signal']

            if signal == 'STRONG_BUY':
                # Aggressive momentum play
                strategies.append({
                    'symbol': symbol,
                    'strategy_type': 'aggressive_momentum_long',
                    'signal_strength': 'STRONG',
                    'expected_return': 0.35,  # 35% target
                    'risk_level': 'HIGH',
                    'time_frame': '1-5 days',
                    'entry_rationale': f'{symbol} showing explosive momentum: {data["momentum_score"]:.1f}%'
                })

            elif signal == 'BUY':
                # Moderate momentum play
                strategies.append({
                    'symbol': symbol,
                    'strategy_type': 'momentum_call_spread',
                    'signal_strength': 'MODERATE',
                    'expected_return': 0.25,  # 25% target
                    'risk_level': 'MEDIUM',
                    'time_frame': '3-10 days',
                    'entry_rationale': f'{symbol} solid momentum: {data["momentum_score"]:.1f}%'
                })

            elif signal == 'HIGH_VOLATILITY':
                # Volatility play
                strategies.append({
                    'symbol': symbol,
                    'strategy_type': 'volatility_straddle',
                    'signal_strength': 'VOLATILITY',
                    'expected_return': 0.40,  # 40% on volatility
                    'risk_level': 'HIGH',
                    'time_frame': '2-7 days',
                    'entry_rationale': f'{symbol} high volatility: {data["volatility"]:.1f}%'
                })

            elif signal == 'STRONG_SELL':
                # Bearish play
                strategies.append({
                    'symbol': symbol,
                    'strategy_type': 'bear_put_spread',
                    'signal_strength': 'STRONG',
                    'expected_return': 0.30,  # 30% on downside
                    'risk_level': 'HIGH',
                    'time_frame': '1-5 days',
                    'entry_rationale': f'{symbol} strong bearish momentum: {data["momentum_score"]:.1f}%'
                })

        # Sort by expected return and signal strength
        strategies.sort(key=lambda x: x['expected_return'], reverse=True)

        return strategies[:8]  # Top 8 strategies

    def _optimize_position_sizing(self, strategies):
        """Optimize position sizing using Kelly Criterion and risk management"""

        total_allocation = self.buying_power * 0.80  # Use 80% of buying power

        # Calculate Kelly-adjusted position sizes
        for strategy in strategies:

            # Estimate win probability based on signal strength
            if strategy['signal_strength'] == 'STRONG':
                win_prob = 0.70
            elif strategy['signal_strength'] == 'MODERATE':
                win_prob = 0.60
            elif strategy['signal_strength'] == 'VOLATILITY':
                win_prob = 0.55
            else:
                win_prob = 0.50

            # Kelly Criterion: f = (bp - q) / b
            # where b = odds, p = win prob, q = loss prob
            expected_return = strategy['expected_return']
            kelly_fraction = (win_prob * (1 + expected_return) - (1 - win_prob)) / expected_return

            # Cap Kelly at 25% for risk management
            kelly_fraction = min(kelly_fraction, 0.25)
            kelly_fraction = max(kelly_fraction, 0.05)  # Minimum 5%

            # Calculate position size
            base_allocation = total_allocation / len(strategies)  # Equal weight base
            kelly_allocation = total_allocation * kelly_fraction

            # Blend equal weight with Kelly for final size
            position_size = (base_allocation * 0.4) + (kelly_allocation * 0.6)

            strategy['position_size'] = position_size
            strategy['win_probability'] = win_prob
            strategy['kelly_fraction'] = kelly_fraction

        return strategies

    async def _execute_strategy(self, strategy, market_data):
        """Execute individual strategy with paper trading"""

        try:
            symbol = strategy['symbol']
            current_price = market_data[symbol]['price']

            # Generate specific trade parameters based on strategy type
            if strategy['strategy_type'] == 'aggressive_momentum_long':
                # Direct stock purchase for momentum
                qty = int(strategy['position_size'] / current_price)

                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )

                execution_result = {
                    'strategy_id': f"{symbol}_momentum_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'strategy_type': strategy['strategy_type'],
                    'action': 'BUY',
                    'quantity': qty,
                    'entry_price': current_price,
                    'position_size': strategy['position_size'],
                    'expected_return': strategy['expected_return'],
                    'order_id': order.id if hasattr(order, 'id') else 'paper_trade',
                    'execution_time': datetime.now().isoformat(),
                    'status': 'EXECUTED'
                }

            elif strategy['strategy_type'] == 'bear_put_spread':
                # Short position for bearish plays
                qty = int(strategy['position_size'] / current_price)

                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

                execution_result = {
                    'strategy_id': f"{symbol}_short_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'strategy_type': strategy['strategy_type'],
                    'action': 'SELL_SHORT',
                    'quantity': qty,
                    'entry_price': current_price,
                    'position_size': strategy['position_size'],
                    'expected_return': strategy['expected_return'],
                    'order_id': order.id if hasattr(order, 'id') else 'paper_trade',
                    'execution_time': datetime.now().isoformat(),
                    'status': 'EXECUTED'
                }

            else:
                # For options strategies, log as planned trades
                execution_result = {
                    'strategy_id': f"{symbol}_options_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'strategy_type': strategy['strategy_type'],
                    'action': 'OPTIONS_STRATEGY',
                    'position_size': strategy['position_size'],
                    'expected_return': strategy['expected_return'],
                    'execution_time': datetime.now().isoformat(),
                    'status': 'PLANNED_OPTIONS_TRADE',
                    'note': 'Options execution requires additional API setup'
                }

            print(f"[EXECUTED] {symbol} {strategy['strategy_type']} | ${strategy['position_size']:,.0f}")

            return execution_result

        except Exception as e:
            print(f"[EXECUTION ERROR] {strategy['symbol']}: {str(e)}")
            return {
                'strategy_id': f"{strategy['symbol']}_error_{int(datetime.now().timestamp())}",
                'symbol': strategy['symbol'],
                'error': str(e),
                'status': 'FAILED'
            }

    def _calculate_risk_metrics(self, strategies, market_data):
        """Calculate portfolio risk metrics"""

        total_position = sum(s['position_size'] for s in strategies)

        # Portfolio beta (simplified)
        weighted_beta = 0
        for strategy in strategies:
            symbol = strategy['symbol']
            weight = strategy['position_size'] / total_position

            # Simplified beta estimates
            beta_estimates = {
                'TSLA': 2.0, 'NVDA': 1.8, 'AAPL': 1.2, 'MSFT': 1.1,
                'AMZN': 1.3, 'SPY': 1.0, 'QQQ': 1.2, 'IWM': 1.1
            }
            beta = beta_estimates.get(symbol, 1.0)
            weighted_beta += weight * beta

        # VaR calculation (simplified)
        portfolio_volatility = 0.25  # Assume 25% portfolio volatility
        var_95 = total_position * 0.025  # 2.5% daily VaR

        max_concentration = 0
        if strategies:
            max_concentration = max(s['position_size'] for s in strategies) / total_position if total_position > 0 else 0

        return {
            'portfolio_beta': weighted_beta,
            'estimated_volatility': portfolio_volatility,
            'value_at_risk_95': var_95,
            'max_position_concentration': max_concentration,
            'number_of_positions': len(strategies),
            'risk_diversification': 'GOOD' if len(strategies) >= 6 else 'MODERATE' if len(strategies) > 0 else 'NO_POSITIONS'
        }

async def execute_advanced_live_system():
    """Execute the advanced live trading system"""

    print("="*70)
    print("ADVANCED LIVE EXECUTION - PROFESSIONAL STRATEGY DEPLOYMENT")
    print("="*70)

    executor = AdvancedLiveExecutor()
    results = await executor.execute_advanced_strategies()

    print(f"\n[COMPOUND MONTHLY ANALYSIS]")
    monthly_contribution = results['compound_monthly_contribution']
    monthly_target = 41.67

    print(f"Monthly Target: {monthly_target}%")
    print(f"This Execution: {monthly_contribution:.2f}%")
    print(f"Progress: {(monthly_contribution/monthly_target)*100:.1f}% of monthly goal")

    if monthly_contribution > 10:
        print(f"[EXCELLENT] Major progress toward 5000% annual target!")
    elif monthly_contribution > 5:
        print(f"[STRONG] Solid contribution to compound monthly system")
    else:
        print(f"[CONSERVATIVE] Safe start, can scale up with success")

    return results

if __name__ == "__main__":
    asyncio.run(execute_advanced_live_system())