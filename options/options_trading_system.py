"""
Comprehensive Options Trading System

Advanced options trading capabilities with Greeks calculation, volatility modeling,
sophisticated strategies, and institutional-grade risk management.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Financial mathematics
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import quantlib as ql

# Market data and pricing
import yfinance as yf
from ib_insync import Option, Stock, Future

# ML for volatility prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStrategy(Enum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COLLAR = "collar"

class ExpirationType(Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    LEAPS = "leaps"

@dataclass
class OptionContract:
    """Comprehensive option contract representation"""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: OptionType

    # Market data
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    volume: int = 0
    open_interest: int = 0

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Volatility
    implied_volatility: float = 0.0
    historical_volatility: float = 0.0

    # Pricing
    theoretical_price: float = 0.0
    intrinsic_value: float = 0.0
    time_value: float = 0.0

    # Risk metrics
    probability_itm: float = 0.0
    probability_profit: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven: float = 0.0

    # Metadata
    liquidity_score: float = 0.0
    bid_ask_spread: float = 0.0
    exchange: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class OptionPosition:
    """Options position with comprehensive tracking"""
    contracts: List[OptionContract]
    quantity: List[int]  # Positive for long, negative for short
    strategy: OptionStrategy
    entry_date: datetime
    entry_prices: List[float]

    # Current metrics
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0

    # Risk metrics
    max_profit_potential: float = 0.0
    max_loss_potential: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)
    probability_profit: float = 0.0

    # Management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_conditions: Dict[str, Any] = field(default_factory=dict)

class BlackScholesCalculator:
    """Black-Scholes options pricing with Greeks"""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        return BlackScholesCalculator.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price"""
        if T <= 0:
            return max(S - K, 0)

        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)

        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(price, 0)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price"""
        if T <= 0:
            return max(K - S, 0)

        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)

        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(price, 0)

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: OptionType) -> Dict[str, float]:
        """Calculate all Greeks"""
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == OptionType.CALL and S > K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)

        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        if option_type == OptionType.CALL:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float,
                          r: float, option_type: OptionType,
                          tolerance: float = 1e-6, max_iterations: int = 100) -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        def objective(sigma):
            if option_type == OptionType.CALL:
                theoretical = BlackScholesCalculator.call_price(S, K, T, r, sigma)
            else:
                theoretical = BlackScholesCalculator.put_price(S, K, T, r, sigma)
            return abs(theoretical - market_price)

        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            return result.x
        except:
            return 0.20  # Default 20% volatility if calculation fails

class VolatilityModeler:
    """Advanced volatility modeling and prediction"""

    def __init__(self):
        self.garch_models = {}
        self.ml_models = {}
        self.volatility_surfaces = {}

    def calculate_historical_volatility(self, prices: pd.Series, window: int = 30) -> float:
        """Calculate historical volatility"""
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.rolling(window).std().iloc[-1] * np.sqrt(252)

    def calculate_garch_volatility(self, returns: pd.Series, symbol: str) -> float:
        """Calculate GARCH volatility forecast"""
        try:
            from arch import arch_model

            if symbol not in self.garch_models:
                model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                self.garch_models[symbol] = model.fit(disp='off')

            forecast = self.garch_models[symbol].forecast(horizon=1)
            return np.sqrt(forecast.variance.iloc[-1, 0] / 100) * np.sqrt(252)

        except ImportError:
            # Fallback to exponential smoothing
            return self._exponential_smoothing_volatility(returns)

    def _exponential_smoothing_volatility(self, returns: pd.Series, alpha: float = 0.06) -> float:
        """Exponential smoothing volatility estimate"""
        squared_returns = returns**2
        ewm_var = squared_returns.ewm(alpha=alpha).mean().iloc[-1]
        return np.sqrt(ewm_var * 252)

    def predict_volatility_ml(self, symbol: str, features: pd.DataFrame) -> float:
        """ML-based volatility prediction"""
        if symbol not in self.ml_models:
            self.ml_models[symbol] = RandomForestRegressor(n_estimators=100, random_state=42)

            # Train on historical data (simplified)
            # In practice, you'd use more sophisticated features
            X_train = features.iloc[:-1]
            y_train = features['volatility'].shift(-1).iloc[:-1]

            self.ml_models[symbol].fit(X_train.drop('volatility', axis=1), y_train)

        # Predict next period volatility
        latest_features = features.iloc[-1:].drop('volatility', axis=1)
        prediction = self.ml_models[symbol].predict(latest_features)[0]

        return max(prediction, 0.05)  # Minimum 5% volatility

    def build_volatility_surface(self, symbol: str, option_data: List[OptionContract]) -> Dict[str, Any]:
        """Build implied volatility surface"""

        # Group by expiration and strike
        iv_data = []
        for option in option_data:
            days_to_expiry = (option.expiration - datetime.now()).days
            moneyness = option.strike / self._get_underlying_price(option.underlying)

            iv_data.append({
                'days_to_expiry': days_to_expiry,
                'moneyness': moneyness,
                'strike': option.strike,
                'implied_vol': option.implied_volatility,
                'option_type': option.option_type.value
            })

        if not iv_data:
            return {}

        df = pd.DataFrame(iv_data)

        # Create volatility surface
        surface = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'data': df,
            'smile_parameters': self._fit_volatility_smile(df),
            'term_structure': self._calculate_term_structure(df)
        }

        self.volatility_surfaces[symbol] = surface
        return surface

    def _get_underlying_price(self, symbol: str) -> float:
        """Get current underlying price"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period="1d")['Close'].iloc[-1]
        except:
            return 100.0  # Default fallback

    def _fit_volatility_smile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Fit volatility smile parameters"""
        try:
            # Simple polynomial fit for demonstration
            # In practice, use more sophisticated models like Heston
            calls = df[df['option_type'] == 'call']
            if len(calls) > 3:
                coeffs = np.polyfit(calls['moneyness'], calls['implied_vol'], 2)
                return {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}
        except:
            pass

        return {'a': 0, 'b': 0, 'c': 0.2}

    def _calculate_term_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility term structure"""
        try:
            # Group by expiration and calculate average IV
            term_structure = df.groupby('days_to_expiry')['implied_vol'].mean().to_dict()
            return term_structure
        except:
            return {}

class OptionsStrategyEngine:
    """Advanced options strategy implementation and analysis"""

    def __init__(self, volatility_modeler: VolatilityModeler):
        self.volatility_modeler = volatility_modeler
        self.bs_calculator = BlackScholesCalculator()

    def analyze_strategy(self, strategy: OptionStrategy, contracts: List[OptionContract],
                        quantities: List[int], underlying_price: float) -> Dict[str, Any]:
        """Comprehensive strategy analysis"""

        analysis = {
            'strategy': strategy.value,
            'contracts': len(contracts),
            'net_premium': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'breakeven_points': [],
            'greeks': {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0},
            'probability_profit': 0.0,
            'expected_return': 0.0,
            'risk_reward_ratio': 0.0,
            'liquidity_score': 0.0
        }

        # Calculate net premium and Greeks
        for contract, qty in zip(contracts, quantities):
            premium = contract.ask if qty > 0 else contract.bid
            analysis['net_premium'] += premium * qty * 100  # Contract multiplier

            # Aggregate Greeks
            analysis['greeks']['delta'] += contract.delta * qty
            analysis['greeks']['gamma'] += contract.gamma * qty
            analysis['greeks']['theta'] += contract.theta * qty
            analysis['greeks']['vega'] += contract.vega * qty

        # Strategy-specific analysis
        if strategy == OptionStrategy.LONG_CALL:
            analysis.update(self._analyze_long_call(contracts[0], quantities[0], underlying_price))
        elif strategy == OptionStrategy.LONG_PUT:
            analysis.update(self._analyze_long_put(contracts[0], quantities[0], underlying_price))
        elif strategy == OptionStrategy.BULL_CALL_SPREAD:
            analysis.update(self._analyze_bull_call_spread(contracts, quantities, underlying_price))
        elif strategy == OptionStrategy.IRON_CONDOR:
            analysis.update(self._analyze_iron_condor(contracts, quantities, underlying_price))
        elif strategy == OptionStrategy.STRADDLE:
            analysis.update(self._analyze_straddle(contracts, quantities, underlying_price))

        # Calculate probability of profit
        analysis['probability_profit'] = self._calculate_probability_profit(
            contracts, quantities, underlying_price
        )

        return analysis

    def _analyze_long_call(self, contract: OptionContract, quantity: int,
                          underlying_price: float) -> Dict[str, Any]:
        """Analyze long call strategy"""
        premium_paid = contract.ask * quantity * 100
        strike = contract.strike

        return {
            'max_profit': float('inf'),  # Unlimited upside
            'max_loss': premium_paid,
            'breakeven_points': [strike + contract.ask],
            'profit_threshold': strike,
            'strategy_type': 'bullish'
        }

    def _analyze_long_put(self, contract: OptionContract, quantity: int,
                         underlying_price: float) -> Dict[str, Any]:
        """Analyze long put strategy"""
        premium_paid = contract.ask * quantity * 100
        strike = contract.strike

        return {
            'max_profit': (strike - contract.ask) * quantity * 100,
            'max_loss': premium_paid,
            'breakeven_points': [strike - contract.ask],
            'profit_threshold': strike,
            'strategy_type': 'bearish'
        }

    def _analyze_bull_call_spread(self, contracts: List[OptionContract],
                                 quantities: List[int], underlying_price: float) -> Dict[str, Any]:
        """Analyze bull call spread strategy"""
        # Assume [long_call, short_call] with [1, -1] quantities
        long_call = contracts[0]
        short_call = contracts[1]

        net_premium = (long_call.ask - short_call.bid) * 100
        max_profit = (short_call.strike - long_call.strike) * 100 - net_premium
        max_loss = net_premium

        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_points': [long_call.strike + net_premium / 100],
            'profit_range': [long_call.strike, short_call.strike],
            'strategy_type': 'bullish'
        }

    def _analyze_iron_condor(self, contracts: List[OptionContract],
                           quantities: List[int], underlying_price: float) -> Dict[str, Any]:
        """Analyze iron condor strategy"""
        # Complex multi-leg strategy analysis
        net_credit = sum(
            (contract.bid if qty < 0 else -contract.ask) * abs(qty) * 100
            for contract, qty in zip(contracts, quantities)
        )

        strikes = sorted([contract.strike for contract in contracts])
        profit_range = [strikes[1], strikes[2]]  # Between middle strikes

        return {
            'max_profit': net_credit,
            'max_loss': (strikes[1] - strikes[0]) * 100 - net_credit,
            'breakeven_points': [
                strikes[0] + net_credit / 100,
                strikes[3] - net_credit / 100
            ],
            'profit_range': profit_range,
            'strategy_type': 'neutral'
        }

    def _analyze_straddle(self, contracts: List[OptionContract],
                         quantities: List[int], underlying_price: float) -> Dict[str, Any]:
        """Analyze straddle strategy"""
        call_contract = next(c for c in contracts if c.option_type == OptionType.CALL)
        put_contract = next(c for c in contracts if c.option_type == OptionType.PUT)

        total_premium = (call_contract.ask + put_contract.ask) * 100
        strike = call_contract.strike

        return {
            'max_profit': float('inf'),  # Unlimited if large move
            'max_loss': total_premium,
            'breakeven_points': [
                strike - total_premium / 100,
                strike + total_premium / 100
            ],
            'profit_requirement': total_premium / 100,  # Minimum move needed
            'strategy_type': 'volatility'
        }

    def _calculate_probability_profit(self, contracts: List[OptionContract],
                                    quantities: List[int], underlying_price: float) -> float:
        """Calculate probability of profit using Monte Carlo simulation"""

        # Simplified calculation - in practice, use more sophisticated methods
        # Get average implied volatility
        avg_iv = np.mean([contract.implied_volatility for contract in contracts])

        # Average time to expiration
        avg_tte = np.mean([
            (contract.expiration - datetime.now()).days / 365
            for contract in contracts
        ])

        if avg_tte <= 0:
            return 0.0

        # Monte Carlo simulation
        n_simulations = 10000
        profitable_outcomes = 0

        for _ in range(n_simulations):
            # Simulate price at expiration
            random_return = np.random.normal(0, avg_iv * np.sqrt(avg_tte))
            simulated_price = underlying_price * np.exp(random_return)

            # Calculate strategy P&L at expiration
            total_pnl = 0
            for contract, qty in zip(contracts, quantities):
                if contract.option_type == OptionType.CALL:
                    option_value = max(simulated_price - contract.strike, 0)
                else:
                    option_value = max(contract.strike - simulated_price, 0)

                # Account for premium paid/received
                if qty > 0:  # Long position
                    pnl = (option_value - contract.ask) * qty * 100
                else:  # Short position
                    pnl = (contract.bid - option_value) * abs(qty) * 100

                total_pnl += pnl

            if total_pnl > 0:
                profitable_outcomes += 1

        return profitable_outcomes / n_simulations

    def recommend_strategies(self, symbol: str, market_outlook: str,
                           volatility_outlook: str, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Recommend options strategies based on market outlook"""

        recommendations = []

        # Bullish strategies
        if market_outlook.lower() == 'bullish':
            if volatility_outlook.lower() == 'low':
                recommendations.append({
                    'strategy': OptionStrategy.LONG_CALL,
                    'reasoning': 'Directional bullish play with limited downside',
                    'risk_level': 'medium',
                    'complexity': 'low'
                })
                recommendations.append({
                    'strategy': OptionStrategy.BULL_CALL_SPREAD,
                    'reasoning': 'Lower cost bullish play with defined risk',
                    'risk_level': 'low',
                    'complexity': 'medium'
                })
            else:
                recommendations.append({
                    'strategy': OptionStrategy.COVERED_CALL,
                    'reasoning': 'Generate income while maintaining upside exposure',
                    'risk_level': 'low',
                    'complexity': 'low'
                })

        # Bearish strategies
        elif market_outlook.lower() == 'bearish':
            recommendations.append({
                'strategy': OptionStrategy.LONG_PUT,
                'reasoning': 'Direct bearish exposure with limited risk',
                'risk_level': 'medium',
                'complexity': 'low'
            })
            recommendations.append({
                'strategy': OptionStrategy.BEAR_PUT_SPREAD,
                'reasoning': 'Lower cost bearish play with defined risk/reward',
                'risk_level': 'low',
                'complexity': 'medium'
            })

        # Neutral strategies
        elif market_outlook.lower() == 'neutral':
            if volatility_outlook.lower() == 'high':
                recommendations.append({
                    'strategy': OptionStrategy.STRADDLE,
                    'reasoning': 'Profit from large moves in either direction',
                    'risk_level': 'medium',
                    'complexity': 'medium'
                })
            else:
                recommendations.append({
                    'strategy': OptionStrategy.IRON_CONDOR,
                    'reasoning': 'Generate income in low volatility environment',
                    'risk_level': 'medium',
                    'complexity': 'high'
                })

        return recommendations

class OptionsRiskManager:
    """Comprehensive options risk management"""

    def __init__(self):
        self.position_limits = {
            'max_delta_exposure': 100.0,
            'max_gamma_exposure': 50.0,
            'max_vega_exposure': 1000.0,
            'max_theta_decay': 500.0,
            'max_single_position': 0.05,  # 5% of portfolio
            'max_concentration': 0.20,    # 20% in options
        }

    def validate_position(self, position: OptionPosition, portfolio_value: float,
                         current_positions: List[OptionPosition]) -> Dict[str, Any]:
        """Validate new options position against risk limits"""

        validation = {
            'approved': True,
            'warnings': [],
            'errors': [],
            'risk_metrics': {},
            'recommendations': []
        }

        # Calculate position value
        position_value = abs(position.current_value)
        position_percentage = position_value / portfolio_value

        # Check position size limits
        if position_percentage > self.position_limits['max_single_position']:
            validation['errors'].append(
                f"Position size {position_percentage:.1%} exceeds limit "
                f"{self.position_limits['max_single_position']:.1%}"
            )
            validation['approved'] = False

        # Calculate total options exposure
        total_options_value = position_value + sum(
            abs(pos.current_value) for pos in current_positions
        )
        options_concentration = total_options_value / portfolio_value

        if options_concentration > self.position_limits['max_concentration']:
            validation['warnings'].append(
                f"Total options exposure {options_concentration:.1%} approaches limit "
                f"{self.position_limits['max_concentration']:.1%}"
            )

        # Check Greeks exposure
        total_delta = position.total_delta + sum(pos.total_delta for pos in current_positions)
        total_gamma = position.total_gamma + sum(pos.total_gamma for pos in current_positions)
        total_vega = position.total_vega + sum(pos.total_vega for pos in current_positions)
        total_theta = position.total_theta + sum(pos.total_theta for pos in current_positions)

        if abs(total_delta) > self.position_limits['max_delta_exposure']:
            validation['errors'].append(f"Delta exposure {total_delta:.1f} exceeds limit")
            validation['approved'] = False

        if abs(total_gamma) > self.position_limits['max_gamma_exposure']:
            validation['warnings'].append(f"Gamma exposure {total_gamma:.1f} is high")

        if abs(total_vega) > self.position_limits['max_vega_exposure']:
            validation['warnings'].append(f"Vega exposure {total_vega:.1f} is high")

        # Time decay analysis
        if total_theta < -self.position_limits['max_theta_decay']:
            validation['warnings'].append(
                f"High theta decay {total_theta:.1f} - position losing ${abs(total_theta):.0f}/day"
            )

        # Risk metrics
        validation['risk_metrics'] = {
            'position_size_pct': position_percentage,
            'options_concentration': options_concentration,
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'total_theta': total_theta,
            'daily_theta_cost': abs(total_theta)
        }

        # Recommendations
        if validation['warnings']:
            validation['recommendations'].append("Consider reducing position size")
            validation['recommendations'].append("Monitor Greeks exposure closely")

        if total_theta < -100:
            validation['recommendations'].append("High time decay - consider shorter timeframes")

        return validation

    def calculate_var(self, positions: List[OptionPosition], confidence: float = 0.95,
                     time_horizon: int = 1) -> Dict[str, float]:
        """Calculate Value at Risk for options portfolio"""

        # Simplified VaR calculation
        # In practice, use Monte Carlo or historical simulation

        total_delta = sum(pos.total_delta for pos in positions)
        total_gamma = sum(pos.total_gamma for pos in positions)
        total_vega = sum(pos.total_vega for pos in positions)

        # Assume daily volatility of underlying
        daily_vol = 0.02  # 2% daily volatility

        # Delta-normal VaR
        z_score = norm.ppf(confidence)
        delta_var = abs(total_delta * daily_vol * z_score)

        # Gamma adjustment (convexity)
        gamma_adjustment = 0.5 * total_gamma * (daily_vol * z_score) ** 2

        # Vega risk (assume 10% volatility change)
        vega_var = abs(total_vega * 0.1 * z_score)

        total_var = delta_var + gamma_adjustment + vega_var

        return {
            'total_var': total_var,
            'delta_var': delta_var,
            'gamma_adjustment': gamma_adjustment,
            'vega_var': vega_var,
            'confidence_level': confidence,
            'time_horizon': time_horizon
        }

class OptionsDataManager:
    """Manage options market data and chains"""

    def __init__(self, broker_manager=None):
        self.broker_manager = broker_manager
        self.options_chains = {}
        self.last_updated = {}

    async def get_options_chain(self, symbol: str, expiration_range: int = 45) -> Dict[str, List[OptionContract]]:
        """Get complete options chain for symbol"""

        # Check cache
        if symbol in self.options_chains:
            last_update = self.last_updated.get(symbol, datetime.min)
            if (datetime.now() - last_update).seconds < 300:  # 5-minute cache
                return self.options_chains[symbol]

        try:
            # Get options data from yfinance (free alternative)
            ticker = yf.Ticker(symbol)

            # Get underlying price
            underlying_price = ticker.history(period="1d")['Close'].iloc[-1]

            # Get option expiration dates
            expirations = ticker.options

            options_chain = {'calls': [], 'puts': []}

            for exp_date in expirations[:3]:  # Limit to first 3 expirations
                try:
                    # Get options chain for this expiration
                    chain = ticker.option_chain(exp_date)

                    # Process calls
                    for _, row in chain.calls.iterrows():
                        contract = self._create_option_contract(
                            symbol, row, OptionType.CALL, exp_date, underlying_price
                        )
                        options_chain['calls'].append(contract)

                    # Process puts
                    for _, row in chain.puts.iterrows():
                        contract = self._create_option_contract(
                            symbol, row, OptionType.PUT, exp_date, underlying_price
                        )
                        options_chain['puts'].append(contract)

                except Exception as e:
                    logging.warning(f"Error processing {symbol} {exp_date}: {e}")
                    continue

            # Cache results
            self.options_chains[symbol] = options_chain
            self.last_updated[symbol] = datetime.now()

            return options_chain

        except Exception as e:
            logging.error(f"Error fetching options chain for {symbol}: {e}")
            return {'calls': [], 'puts': []}

    def _create_option_contract(self, underlying: str, row: pd.Series,
                              option_type: OptionType, expiration: str,
                              underlying_price: float) -> OptionContract:
        """Create OptionContract from market data"""

        # Parse expiration date
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')

        # Calculate time to expiration
        tte = (exp_date - datetime.now()).days / 365.0

        # Basic contract info
        strike = row['strike']
        bid = row.get('bid', 0.0)
        ask = row.get('ask', 0.0)
        last_price = row.get('lastPrice', (bid + ask) / 2 if bid and ask else 0.0)
        volume = row.get('volume', 0)
        open_interest = row.get('openInterest', 0)
        implied_vol = row.get('impliedVolatility', 0.0)

        # Calculate Greeks and pricing
        if tte > 0 and implied_vol > 0:
            greeks = BlackScholesCalculator.calculate_greeks(
                underlying_price, strike, tte, 0.02, implied_vol, option_type
            )

            if option_type == OptionType.CALL:
                theoretical_price = BlackScholesCalculator.call_price(
                    underlying_price, strike, tte, 0.02, implied_vol
                )
                intrinsic_value = max(underlying_price - strike, 0)
            else:
                theoretical_price = BlackScholesCalculator.put_price(
                    underlying_price, strike, tte, 0.02, implied_vol
                )
                intrinsic_value = max(strike - underlying_price, 0)
        else:
            greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            if option_type == OptionType.CALL:
                intrinsic_value = max(underlying_price - strike, 0)
            else:
                intrinsic_value = max(strike - underlying_price, 0)
            theoretical_price = intrinsic_value

        time_value = max(last_price - intrinsic_value, 0)

        # Create contract
        return OptionContract(
            symbol=f"{underlying}_{exp_date.strftime('%y%m%d')}_{option_type.value[0].upper()}{int(strike):05d}",
            underlying=underlying,
            strike=strike,
            expiration=exp_date,
            option_type=option_type,
            bid=bid,
            ask=ask,
            last_price=last_price,
            volume=volume,
            open_interest=open_interest,
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            rho=greeks['rho'],
            implied_volatility=implied_vol,
            theoretical_price=theoretical_price,
            intrinsic_value=intrinsic_value,
            time_value=time_value,
            bid_ask_spread=ask - bid if ask and bid else 0,
            liquidity_score=min(volume * open_interest, 1000) / 1000  # Simplified liquidity score
        )

# Example usage and testing
async def example_options_trading():
    """Example of options trading system usage"""

    print("Options Trading System Demo")
    print("=" * 50)

    # Initialize components
    volatility_modeler = VolatilityModeler()
    strategy_engine = OptionsStrategyEngine(volatility_modeler)
    risk_manager = OptionsRiskManager()
    data_manager = OptionsDataManager()

    # Get options chain
    symbol = "AAPL"
    print(f"\nFetching options chain for {symbol}...")

    options_chain = await data_manager.get_options_chain(symbol)

    if options_chain['calls'] and options_chain['puts']:
        print(f"Found {len(options_chain['calls'])} calls and {len(options_chain['puts'])} puts")

        # Show some examples
        print("\nSample Call Options:")
        for i, call in enumerate(options_chain['calls'][:3]):
            print(f"  {call.symbol}: Strike ${call.strike}, IV {call.implied_volatility:.1%}, "
                  f"Delta {call.delta:.3f}, Bid ${call.bid:.2f}")

        print("\nSample Put Options:")
        for i, put in enumerate(options_chain['puts'][:3]):
            print(f"  {put.symbol}: Strike ${put.strike}, IV {put.implied_volatility:.1%}, "
                  f"Delta {put.delta:.3f}, Bid ${put.bid:.2f}")

        # Analyze a simple long call strategy
        if options_chain['calls']:
            call_option = options_chain['calls'][0]
            underlying_price = 150.0  # Example price

            print(f"\nAnalyzing Long Call Strategy:")
            analysis = strategy_engine.analyze_strategy(
                OptionStrategy.LONG_CALL,
                [call_option],
                [1],
                underlying_price
            )

            print(f"  Max Loss: ${analysis['max_loss']:.2f}")
            print(f"  Breakeven: ${analysis['breakeven_points'][0]:.2f}")
            print(f"  Probability of Profit: {analysis['probability_profit']:.1%}")

    # Strategy recommendations
    print(f"\nStrategy Recommendations for {symbol}:")
    recommendations = strategy_engine.recommend_strategies(
        symbol, 'bullish', 'low', 'medium'
    )

    for rec in recommendations:
        print(f"  {rec['strategy'].value}: {rec['reasoning']}")

    print("\nOptions trading system demonstration complete!")

if __name__ == "__main__":
    asyncio.run(example_options_trading())