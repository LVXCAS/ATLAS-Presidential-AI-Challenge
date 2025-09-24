"""
LEVERAGE ENHANCED REAL STRATEGIES
=================================
Take our validated real strategies (146.5% pairs trading) and enhance them with:
1. Higher leverage (2x, 4x, 8x)
2. Higher frequency rebalancing
3. Multiple strategy combinations
4. Risk-controlled scaling

Goal: Scale validated 146.5% strategy to 1000%+ using real backtesting
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def enhanced_pairs_trading_with_leverage(leverage_multiplier=1.0, rebalance_freq='monthly'):
    """
    Enhanced version of our validated pairs trading strategy with leverage
    """
    print(f"Testing Pairs Trading with {leverage_multiplier}x leverage, {rebalance_freq} rebalancing")

    # Load real market data (same as validated strategy)
    symbols = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'EFA', 'EEM', 'VEA', 'VWO', 'IEFA']

    try:
        # Download real historical data
        data = {}
        for symbol in symbols:
            ticker_data = yf.download(symbol, start='2020-01-01', end='2024-09-20', progress=False)
            if len(ticker_data) > 500:  # Ensure sufficient data
                data[symbol] = ticker_data['Close']

        print(f"Loaded data for {len(data)} symbols")

        if len(data) < 4:
            return None

        # Create pairs and find best correlations
        pairs = []
        for i, sym1 in enumerate(data.keys()):
            for sym2 in list(data.keys())[i+1:]:
                # Align data
                df1, df2 = data[sym1].align(data[sym2], join='inner')
                if len(df1) > 100:
                    correlation = df1.pct_change().corr(df2.pct_change())
                    if abs(correlation) > 0.3:  # Meaningful correlation
                        pairs.append((sym1, sym2, abs(correlation)))

        # Sort by correlation strength
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:3]  # Top 3 pairs

        print(f"Selected pairs: {[f'{p[0]}-{p[1]}' for p in top_pairs]}")

        # Enhanced pairs trading with leverage
        portfolio_value = 100000
        trades = []

        for pair_symbol1, pair_symbol2, corr in top_pairs:
            df1, df2 = data[pair_symbol1].align(data[pair_symbol2], join='inner')

            # Calculate spread
            ratio = df1 / df2
            ratio_ma = ratio.rolling(20).mean()
            ratio_std = ratio.rolling(20).std()
            z_score = (ratio - ratio_ma) / ratio_std

            # Enhanced trading logic with leverage
            position_allocation = 1.0 / len(top_pairs)  # Equal allocation across pairs

            i = 21  # Start after indicator calculation
            while i < len(z_score) - 5:
                current_z = z_score.iloc[i]

                if abs(current_z) > 1.5:  # Entry threshold (less aggressive for real trading)
                    # Determine position direction
                    if current_z > 1.5:  # sym1 overvalued vs sym2
                        direction1, direction2 = -1, 1  # Short sym1, long sym2
                    else:  # sym1 undervalued vs sym2
                        direction1, direction2 = 1, -1   # Long sym1, short sym2

                    # Entry prices
                    entry_price1 = df1.iloc[i+1]  # Next day
                    entry_price2 = df2.iloc[i+1]

                    # Exit when spread normalizes or after max 10 days
                    exit_day = i + 1
                    for j in range(i+2, min(i+11, len(z_score))):
                        if abs(z_score.iloc[j]) < 0.5:  # Mean reversion
                            exit_day = j
                            break
                    else:
                        exit_day = min(i+10, len(z_score)-1)  # Max 10 days

                    # Exit prices
                    exit_price1 = df1.iloc[exit_day]
                    exit_price2 = df2.iloc[exit_day]

                    # Calculate returns
                    return1 = (exit_price1 / entry_price1 - 1) * direction1
                    return2 = (exit_price2 / entry_price2 - 1) * direction2

                    # Combined return (50-50 allocation between assets)
                    combined_return = (return1 + return2) / 2

                    # Apply leverage
                    leveraged_return = combined_return * leverage_multiplier

                    # Transaction costs (more realistic)
                    transaction_cost = 0.001 * leverage_multiplier  # Higher cost with leverage
                    net_return = leveraged_return - transaction_cost

                    # Position sizing (Kelly-like approach)
                    position_size = portfolio_value * position_allocation * 0.2  # 20% max per trade

                    # Risk management: Cap single trade loss
                    max_loss_per_trade = portfolio_value * 0.05  # 5% max loss per trade
                    if position_size * abs(net_return) > max_loss_per_trade and net_return < 0:
                        position_size = max_loss_per_trade / abs(net_return)

                    # Execute trade
                    trade_pnl = position_size * net_return
                    portfolio_value += trade_pnl

                    trades.append({
                        'pair': f"{pair_symbol1}-{pair_symbol2}",
                        'entry_date': df1.index[i+1],
                        'exit_date': df1.index[exit_day],
                        'days_held': exit_day - (i+1),
                        'return': net_return,
                        'leverage': leverage_multiplier,
                        'portfolio_value': portfolio_value
                    })

                    # Skip ahead to avoid overlapping trades
                    i = exit_day + 1
                else:
                    i += 1

        if len(trades) > 5:
            # Calculate performance metrics
            returns = [t['return'] for t in trades]
            total_return = (portfolio_value - 100000) / 100000

            # Annualize based on actual time period
            start_date = min(trade['entry_date'] for trade in trades)
            end_date = max(trade['exit_date'] for trade in trades)
            years_elapsed = (end_date - start_date).days / 365.25
            annual_return = (portfolio_value / 100000) ** (1/years_elapsed) - 1

            # Risk metrics
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252 / np.mean([t['days_held'] for t in trades]))
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0

            # Drawdown calculation
            portfolio_values = [100000] + [t['portfolio_value'] for t in trades]
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)

            return {
                'strategy': f'Enhanced_Pairs_Trading_{leverage_multiplier}x',
                'leverage': leverage_multiplier,
                'annual_return_pct': annual_return * 100,
                'total_return_pct': total_return * 100,
                'final_value': portfolio_value,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'years_elapsed': years_elapsed,
                'is_1000_plus': annual_return >= 10.0  # 1000%+
            }

    except Exception as e:
        print(f"Error in enhanced pairs trading: {e}")
        return None

def test_leverage_scaling():
    """Test multiple leverage levels on validated strategy"""
    print("=" * 80)
    print("LEVERAGE ENHANCED REAL STRATEGY TESTING")
    print("Scaling validated pairs trading strategy with leverage")
    print("=" * 80)

    leverage_levels = [1.0, 2.0, 4.0, 6.0, 8.0]
    results = []

    for leverage in leverage_levels:
        result = enhanced_pairs_trading_with_leverage(leverage)
        if result:
            results.append(result)

            print(f"\\n{result['strategy']}:")
            print(f"  Annual Return: {result['annual_return_pct']:.1f}%")
            print(f"  Total Trades: {result['total_trades']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.1%}")
            print(f"  Final Value: ${result['final_value']:,.0f}")

            if result['is_1000_plus']:
                print(f"  *** 1000%+ TARGET ACHIEVED! ***")
            elif result['annual_return_pct'] > 500:
                print(f"  *** 500%+ ACHIEVED ***")

    print(f"\\n" + "=" * 80)
    print("LEVERAGE SCALING RESULTS")
    print("=" * 80)

    if results:
        # Find best risk-adjusted performance
        best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        best_return = max(results, key=lambda x: x['annual_return_pct'])

        print(f"\\nBEST RISK-ADJUSTED: {best_sharpe['strategy']}")
        print(f"  Annual Return: {best_sharpe['annual_return_pct']:.1f}%")
        print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")

        print(f"\\nBEST ABSOLUTE RETURN: {best_return['strategy']}")
        print(f"  Annual Return: {best_return['annual_return_pct']:.1f}%")
        print(f"  Max Drawdown: {best_return['max_drawdown']:.1%}")

        # Check for 1000%+ achievement
        high_return_strategies = [r for r in results if r['is_1000_plus']]

        if high_return_strategies:
            print(f"\\n*** SUCCESS! {len(high_return_strategies)} strategies achieved 1000%+ ***")
            for strategy in high_return_strategies:
                print(f"  {strategy['strategy']}: {strategy['annual_return_pct']:.1f}% annual")
        else:
            print(f"\\nBest achieved: {best_return['annual_return_pct']:.1f}% annual")
            print(f"To reach 1000%+: Need {1000 / best_return['annual_return_pct']:.1f}x more leverage or better strategies")

    else:
        print("\\nNo successful leverage tests - data loading issues")

    return results

if __name__ == "__main__":
    results = test_leverage_scaling()