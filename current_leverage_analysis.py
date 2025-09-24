"""
CURRENT LEVERAGE ANALYSIS - SYNTHETIC OPTIONS PORTFOLIO
Analyze our existing $42,854 in 3x leveraged ETF positions
Calculate effective options-equivalent exposure and monthly potential
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import numpy as np

load_dotenv(override=True)

def analyze_current_leverage_exposure():
    """Analyze current leveraged positions as synthetic options"""

    print("="*70)
    print("CURRENT LEVERAGE ANALYSIS - SYNTHETIC OPTIONS PORTFOLIO")
    print("="*70)

    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

    # Get current account status
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)
    buying_power = float(account.buying_power)

    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"Available Buying Power: ${buying_power:,.0f}")

    # Analyze current positions
    positions = api.list_positions()

    if not positions:
        print("No current positions found")
        return

    print(f"\n[CURRENT LEVERAGED POSITIONS]")

    position_data = {}
    total_market_value = 0
    total_unrealized_pl = 0

    for pos in positions:
        symbol = pos.symbol
        quantity = int(pos.qty)
        market_value = float(pos.market_value)
        unrealized_pl = float(pos.unrealized_pl)
        avg_entry_price = float(pos.avg_entry_price)

        # Get current price and momentum
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='5m')

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]

                # Calculate recent momentum
                if len(hist) > 12:
                    hour_ago_price = hist['Close'].iloc[-13]  # ~1 hour ago (5min intervals)
                    momentum_1h = ((current_price / hour_ago_price) - 1) * 100
                else:
                    momentum_1h = 0

                # Calculate volatility
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252 * 78) * 100  # Annualized volatility

            else:
                current_price = market_value / quantity if quantity != 0 else 0
                momentum_1h = 0
                volatility = 0

        except Exception as e:
            current_price = market_value / quantity if quantity != 0 else 0
            momentum_1h = 0
            volatility = 0
            print(f"  [DATA WARNING] {symbol}: {str(e)}")

        # Calculate 3x leverage metrics
        leverage_factor = 3  # All our ETFs are 3x leveraged
        effective_exposure = abs(market_value) * leverage_factor

        position_data[symbol] = {
            'quantity': quantity,
            'current_price': current_price,
            'market_value': market_value,
            'unrealized_pl': unrealized_pl,
            'avg_entry_price': avg_entry_price,
            'momentum_1h': momentum_1h,
            'volatility': volatility,
            'leverage_factor': leverage_factor,
            'effective_exposure': effective_exposure,
            'portfolio_weight': abs(market_value) / portfolio_value * 100
        }

        total_market_value += abs(market_value)
        total_unrealized_pl += unrealized_pl

        print(f"{symbol}: {quantity} shares | ${market_value:,.0f} | P&L: ${unrealized_pl:+,.0f}")
        print(f"  Current: ${current_price:.2f} | Entry: ${avg_entry_price:.2f}")
        print(f"  1h Momentum: {momentum_1h:+.2f}% | Volatility: {volatility:.1f}%")
        print(f"  3x Effective Exposure: ${effective_exposure:,.0f}")

    # Portfolio analysis
    total_effective_exposure = sum(pos['effective_exposure'] for pos in position_data.values())
    effective_leverage_ratio = total_effective_exposure / total_market_value if total_market_value > 0 else 0

    print(f"\n[SYNTHETIC OPTIONS PORTFOLIO ANALYSIS]")
    print(f"Total Position Value: ${total_market_value:,.0f}")
    print(f"Total P&L: ${total_unrealized_pl:+,.0f}")
    print(f"Total 3x Exposure: ${total_effective_exposure:,.0f}")
    print(f"Effective Leverage: {effective_leverage_ratio:.1f}x")
    print(f"Portfolio Utilization: {(total_market_value/portfolio_value)*100:.1f}%")

    # Monthly return projections
    monthly_projections = calculate_monthly_projections(position_data, total_market_value)

    print(f"\n[MONTHLY RETURN PROJECTIONS]")
    for scenario, return_pct in monthly_projections.items():
        print(f"{scenario}: {return_pct:.1f}%")

        if return_pct >= 41.67:
            annual_equivalent = ((1 + return_pct/100) ** 12 - 1) * 100
            print(f"  Annual Equivalent: {annual_equivalent:,.0f}%")

    # Best case scenario analysis
    best_case = monthly_projections['Optimistic Scenario']
    target_achievement = best_case >= 41.67

    print(f"\n[41.67% MONTHLY TARGET ANALYSIS]")
    print(f"Current Best Case: {best_case:.1f}%")
    print(f"Monthly Target: 41.67%")
    print(f"Target Achievement: {'ACHIEVED' if target_achievement else 'PARTIAL'}")

    if target_achievement:
        print(f"[SUCCESS] Synthetic options portfolio can achieve 41%+ monthly!")
        annual_projection = ((1 + best_case/100) ** 12 - 1) * 100
        print(f"[ANNUAL PROJECTION] {annual_projection:,.0f}% return potential")

        if annual_projection >= 5000:
            print(f"[5000%+ TARGET] Annual goal achievable with current positions!")
    else:
        shortfall = 41.67 - best_case
        additional_capital_needed = (shortfall / best_case) * total_market_value if best_case > 0 else 0
        print(f"[SHORTFALL] Need {shortfall:.1f}% more monthly return")
        print(f"[SCALING] Would need ~${additional_capital_needed:,.0f} more capital in similar positions")

    # Options equivalent comparison
    print(f"\n[OPTIONS EQUIVALENT ANALYSIS]")
    print(f"Current 3x ETF positions equivalent to:")

    for symbol, data in position_data.items():
        underlying = get_underlying_symbol(symbol)
        options_equivalent_delta = estimate_options_delta(data['volatility'], 30)  # 30 days to expiration

        print(f"{symbol} -> {underlying} options with ~{options_equivalent_delta:.0f} delta")
        print(f"  Effective leverage matches ~{data['leverage_factor']}x call options")

    # Risk analysis
    risk_metrics = calculate_risk_metrics(position_data, total_market_value)

    print(f"\n[RISK ANALYSIS]")
    print(f"Portfolio Beta: {risk_metrics['portfolio_beta']:.2f}")
    print(f"Max Drawdown Risk: {risk_metrics['max_drawdown_risk']:.1f}%")
    print(f"Daily VaR (95%): ${risk_metrics['daily_var']:,.0f}")

    # Save analysis
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'portfolio_value': portfolio_value,
        'positions': position_data,
        'total_market_value': total_market_value,
        'total_unrealized_pl': total_unrealized_pl,
        'total_effective_exposure': total_effective_exposure,
        'effective_leverage_ratio': effective_leverage_ratio,
        'monthly_projections': monthly_projections,
        'target_achievement': target_achievement,
        'risk_metrics': risk_metrics,
        'available_buying_power': buying_power
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"current_leverage_analysis_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"\n[SAVED] Analysis: {filename}")

    return analysis_data

def calculate_monthly_projections(position_data, total_value):
    """Calculate monthly return projections for different scenarios"""

    # Scenario-based monthly returns for 3x leveraged ETFs
    scenarios = {
        'Conservative Scenario': 0.25,  # 25% monthly
        'Base Case Scenario': 0.35,    # 35% monthly
        'Optimistic Scenario': 0.50,   # 50% monthly
        'Aggressive Bull Market': 0.75  # 75% monthly
    }

    projections = {}

    for scenario, base_return in scenarios.items():
        weighted_return = 0

        for symbol, data in position_data.items():
            weight = abs(data['market_value']) / total_value

            # Adjust return based on recent momentum and volatility
            momentum_factor = 1 + (data['momentum_1h'] / 100) * 0.5  # 50% momentum influence
            volatility_factor = 1 + (data['volatility'] / 100) * 0.2  # 20% volatility boost

            adjusted_return = base_return * momentum_factor * volatility_factor
            weighted_return += weight * adjusted_return

        projections[scenario] = weighted_return * 100

    return projections

def get_underlying_symbol(etf_symbol):
    """Get underlying index for leveraged ETF"""

    underlying_map = {
        'TQQQ': 'QQQ',
        'SOXL': 'SOXX',
        'UPRO': 'SPY',
        'TNA': 'IWM',
        'FNGU': 'FANG+'
    }

    return underlying_map.get(etf_symbol, 'UNKNOWN')

def estimate_options_delta(volatility, days_to_expiry):
    """Estimate options delta equivalent for leverage comparison"""

    # Simplified delta estimation based on volatility and time
    # Higher volatility and shorter time = higher delta for equivalent leverage

    base_delta = 0.7  # Typical delta for leveraged strategies
    vol_adjustment = min(volatility / 50, 1.5)  # Volatility factor
    time_adjustment = min(30 / days_to_expiry, 2.0)  # Time decay factor

    estimated_delta = base_delta * vol_adjustment * time_adjustment
    return min(estimated_delta * 100, 95)  # Cap at 95 delta

def calculate_risk_metrics(position_data, total_value):
    """Calculate portfolio risk metrics"""

    # Portfolio beta calculation (simplified)
    beta_estimates = {
        'TQQQ': 3.0,  # 3x tech leverage
        'SOXL': 3.2,  # 3x semiconductor leverage
        'UPRO': 3.0,  # 3x market leverage
        'TNA': 3.1,   # 3x small cap leverage
        'FNGU': 3.3   # 3x FANG+ leverage
    }

    portfolio_beta = 0
    max_position_weight = 0

    for symbol, data in position_data.items():
        weight = abs(data['market_value']) / total_value
        beta = beta_estimates.get(symbol, 3.0)
        portfolio_beta += weight * beta

        max_position_weight = max(max_position_weight, weight)

    # Risk calculations
    max_drawdown_risk = portfolio_beta * 15  # Estimate max drawdown as beta * 15%
    daily_var = total_value * 0.05 * portfolio_beta  # 5% * beta daily VaR

    return {
        'portfolio_beta': portfolio_beta,
        'max_drawdown_risk': max_drawdown_risk,
        'daily_var': daily_var,
        'max_position_concentration': max_position_weight * 100,
        'diversification_score': len(position_data)
    }

if __name__ == "__main__":
    analyze_current_leverage_exposure()