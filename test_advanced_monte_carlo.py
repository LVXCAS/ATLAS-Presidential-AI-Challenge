#!/usr/bin/env python3
"""
Advanced Monte Carlo Profitability Test
Complex market scenarios with stochastic processes, correlation effects, and realistic market dynamics
"""

import asyncio
import numpy as np
import pandas as pd
import random
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import math
from scipy import stats
from dataclasses import dataclass

# Add current directory to path
sys.path.append('.')

@dataclass
class MarketState:
    """Complex market state with multiple factors"""
    vix_level: float
    market_sentiment: float  # -1 to 1
    sector_rotation: str
    fed_policy_stance: str
    earnings_season: bool
    geopolitical_risk: float  # 0 to 1
    liquidity_conditions: float  # 0 to 1
    correlation_regime: str  # 'low', 'medium', 'high'
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'

@dataclass
class EconomicCycle:
    """Economic cycle state"""
    phase: str  # 'expansion', 'peak', 'contraction', 'trough'
    inflation_rate: float
    interest_rates: float
    unemployment_rate: float
    gdp_growth: float

class AdvancedMarketSimulator:
    """Sophisticated market simulation with multiple stochastic processes"""
    
    def __init__(self):
        self.symbols = {
            'AAPL': {'sector': 'tech', 'beta': 1.2, 'base_vol': 0.25, 'avg_price': 225},
            'MSFT': {'sector': 'tech', 'beta': 1.1, 'base_vol': 0.23, 'avg_price': 280},
            'GOOGL': {'sector': 'tech', 'beta': 1.3, 'base_vol': 0.28, 'avg_price': 140},
            'AMZN': {'sector': 'consumer', 'beta': 1.4, 'base_vol': 0.32, 'avg_price': 145},
            'TSLA': {'sector': 'auto', 'beta': 2.1, 'base_vol': 0.45, 'avg_price': 200},
            'NVDA': {'sector': 'tech', 'beta': 1.8, 'base_vol': 0.38, 'avg_price': 450},
            'SPY': {'sector': 'broad', 'beta': 1.0, 'base_vol': 0.18, 'avg_price': 450},
            'QQQ': {'sector': 'tech', 'beta': 1.2, 'base_vol': 0.22, 'avg_price': 390},
            'IWM': {'sector': 'small', 'beta': 1.3, 'base_vol': 0.28, 'avg_price': 200},
            'XLF': {'sector': 'finance', 'beta': 1.5, 'base_vol': 0.30, 'avg_price': 35}
        }
        
        self.market_regimes = {
            'bull_market': {'prob': 0.35, 'return_drift': 0.08, 'vol_multiplier': 0.9},
            'bear_market': {'prob': 0.15, 'return_drift': -0.12, 'vol_multiplier': 1.5},
            'sideways': {'prob': 0.30, 'return_drift': 0.02, 'vol_multiplier': 0.8},
            'volatile_bull': {'prob': 0.10, 'return_drift': 0.15, 'vol_multiplier': 1.8},
            'volatile_bear': {'prob': 0.07, 'return_drift': -0.18, 'vol_multiplier': 2.2},
            'crisis': {'prob': 0.03, 'return_drift': -0.35, 'vol_multiplier': 3.5}
        }
        
        # State variables for persistence
        self.current_market_state = None
        self.current_economic_cycle = None
        self.correlation_matrix = self._generate_correlation_matrix()
        
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate realistic correlation matrix between assets"""
        n_assets = len(self.symbols)
        
        # Base correlation structure
        base_correlations = np.array([
            [1.00, 0.75, 0.70, 0.60, 0.45, 0.65, 0.80, 0.85, 0.55, 0.50],  # AAPL
            [0.75, 1.00, 0.80, 0.65, 0.40, 0.70, 0.75, 0.80, 0.50, 0.45],  # MSFT
            [0.70, 0.80, 1.00, 0.60, 0.50, 0.75, 0.70, 0.85, 0.45, 0.40],  # GOOGL
            [0.60, 0.65, 0.60, 1.00, 0.35, 0.55, 0.65, 0.70, 0.60, 0.55],  # AMZN
            [0.45, 0.40, 0.50, 0.35, 1.00, 0.60, 0.50, 0.55, 0.40, 0.30],  # TSLA
            [0.65, 0.70, 0.75, 0.55, 0.60, 1.00, 0.60, 0.75, 0.45, 0.35],  # NVDA
            [0.80, 0.75, 0.70, 0.65, 0.50, 0.60, 1.00, 0.90, 0.85, 0.75],  # SPY
            [0.85, 0.80, 0.85, 0.70, 0.55, 0.75, 0.90, 1.00, 0.65, 0.55],  # QQQ
            [0.55, 0.50, 0.45, 0.60, 0.40, 0.45, 0.85, 0.65, 1.00, 0.70],  # IWM
            [0.50, 0.45, 0.40, 0.55, 0.30, 0.35, 0.75, 0.55, 0.70, 1.00]   # XLF
        ])
        
        return base_correlations
    
    def _generate_market_state(self) -> MarketState:
        """Generate complex market state with multiple factors"""
        
        # VIX level with mean reversion
        if self.current_market_state:
            vix_drift = 0.05 * (20 - self.current_market_state.vix_level)  # Mean revert to 20
            vix_shock = np.random.normal(0, 3)
            vix_level = max(10, min(80, self.current_market_state.vix_level + vix_drift + vix_shock))
        else:
            vix_level = np.random.gamma(4, 5)  # Skewed distribution for VIX
        
        # Market sentiment with persistence
        if self.current_market_state:
            sentiment_persistence = 0.7
            sentiment_innovation = np.random.normal(0, 0.3)
            market_sentiment = np.clip(
                sentiment_persistence * self.current_market_state.market_sentiment + sentiment_innovation,
                -1, 1
            )
        else:
            market_sentiment = np.random.normal(0, 0.5)
            market_sentiment = np.clip(market_sentiment, -1, 1)
        
        # Sector rotation
        sector_rotations = ['growth', 'value', 'defensive', 'cyclical', 'momentum']
        sector_weights = [0.25, 0.20, 0.15, 0.25, 0.15] if market_sentiment > 0 else [0.15, 0.30, 0.25, 0.15, 0.15]
        sector_rotation = np.random.choice(sector_rotations, p=sector_weights)
        
        # Fed policy
        fed_policies = ['dovish', 'neutral', 'hawkish']
        fed_weights = [0.4, 0.4, 0.2] if vix_level > 25 else [0.2, 0.5, 0.3]
        fed_policy_stance = np.random.choice(fed_policies, p=fed_weights)
        
        # Other factors
        earnings_season = np.random.random() < 0.25  # 25% of time
        geopolitical_risk = np.random.beta(2, 5)  # Skewed toward low risk
        liquidity_conditions = np.random.beta(5, 2)  # Skewed toward high liquidity
        
        # Correlation and volatility regimes
        correlation_regimes = ['low', 'medium', 'high']
        corr_weights = [0.4, 0.4, 0.2] if vix_level < 20 else [0.1, 0.3, 0.6]
        correlation_regime = np.random.choice(correlation_regimes, p=corr_weights)
        
        if vix_level < 15:
            volatility_regime = 'low'
        elif vix_level < 25:
            volatility_regime = 'normal'
        elif vix_level < 40:
            volatility_regime = 'high'
        else:
            volatility_regime = 'extreme'
        
        return MarketState(
            vix_level=vix_level,
            market_sentiment=market_sentiment,
            sector_rotation=sector_rotation,
            fed_policy_stance=fed_policy_stance,
            earnings_season=earnings_season,
            geopolitical_risk=geopolitical_risk,
            liquidity_conditions=liquidity_conditions,
            correlation_regime=correlation_regime,
            volatility_regime=volatility_regime
        )
    
    def _generate_economic_cycle(self) -> EconomicCycle:
        """Generate economic cycle state"""
        
        phases = ['expansion', 'peak', 'contraction', 'trough']
        phase_transitions = {
            'expansion': {'expansion': 0.85, 'peak': 0.15, 'contraction': 0.0, 'trough': 0.0},
            'peak': {'expansion': 0.2, 'peak': 0.4, 'contraction': 0.4, 'trough': 0.0},
            'contraction': {'expansion': 0.0, 'peak': 0.0, 'contraction': 0.8, 'trough': 0.2},
            'trough': {'expansion': 0.6, 'peak': 0.0, 'contraction': 0.1, 'trough': 0.3}
        }
        
        if self.current_economic_cycle:
            current_phase = self.current_economic_cycle.phase
            phase_probs = list(phase_transitions[current_phase].values())
            phase = np.random.choice(phases, p=phase_probs)
        else:
            phase = np.random.choice(phases, p=[0.5, 0.2, 0.2, 0.1])
        
        # Economic indicators based on phase
        if phase == 'expansion':
            inflation_rate = np.random.normal(2.5, 0.8)
            interest_rates = np.random.normal(3.5, 1.0)
            unemployment_rate = np.random.normal(4.5, 0.8)
            gdp_growth = np.random.normal(3.0, 1.2)
        elif phase == 'peak':
            inflation_rate = np.random.normal(3.5, 1.0)
            interest_rates = np.random.normal(4.5, 1.2)
            unemployment_rate = np.random.normal(3.8, 0.6)
            gdp_growth = np.random.normal(2.2, 1.0)
        elif phase == 'contraction':
            inflation_rate = np.random.normal(1.5, 1.2)
            interest_rates = np.random.normal(2.5, 1.5)
            unemployment_rate = np.random.normal(6.5, 1.5)
            gdp_growth = np.random.normal(-0.5, 2.0)
        else:  # trough
            inflation_rate = np.random.normal(1.0, 0.8)
            interest_rates = np.random.normal(1.5, 1.0)
            unemployment_rate = np.random.normal(8.0, 1.8)
            gdp_growth = np.random.normal(-1.8, 1.5)
        
        return EconomicCycle(
            phase=phase,
            inflation_rate=max(0, inflation_rate),
            interest_rates=max(0.1, interest_rates),
            unemployment_rate=max(2, unemployment_rate),
            gdp_growth=gdp_growth
        )
    
    def generate_complex_scenario(self) -> Dict:
        """Generate a complex market scenario with multiple interacting factors"""
        
        # Update market state
        self.current_market_state = self._generate_market_state()
        self.current_economic_cycle = self._generate_economic_cycle()
        
        # Select symbol with sector bias
        symbol_weights = {}
        for symbol, info in self.symbols.items():
            weight = 1.0
            
            # Sector rotation effects
            if self.current_market_state.sector_rotation == 'growth' and info['sector'] == 'tech':
                weight *= 1.5
            elif self.current_market_state.sector_rotation == 'value' and info['sector'] == 'finance':
                weight *= 1.4
            elif self.current_market_state.sector_rotation == 'defensive' and info['sector'] == 'broad':
                weight *= 1.3
            
            symbol_weights[symbol] = weight
        
        # Normalize weights
        total_weight = sum(symbol_weights.values())
        symbol_probs = [symbol_weights[sym] / total_weight for sym in self.symbols.keys()]
        selected_symbol = np.random.choice(list(self.symbols.keys()), p=symbol_probs)
        
        symbol_info = self.symbols[selected_symbol]
        
        # Select market regime
        regime_probs = list(self.market_regimes.values())
        regime_names = list(self.market_regimes.keys())
        
        # Adjust regime probabilities based on market state
        if self.current_market_state.market_sentiment > 0.5:
            # More bullish regimes
            regime_probs[0]['prob'] *= 1.5  # bull_market
            regime_probs[3]['prob'] *= 1.3  # volatile_bull
        elif self.current_market_state.market_sentiment < -0.5:
            # More bearish regimes
            regime_probs[1]['prob'] *= 1.5  # bear_market
            regime_probs[4]['prob'] *= 1.3  # volatile_bear
        
        # Crisis more likely with high geopolitical risk
        if self.current_market_state.geopolitical_risk > 0.8:
            regime_probs[5]['prob'] *= 3.0  # crisis
        
        # Normalize probabilities
        total_prob = sum(r['prob'] for r in regime_probs)
        normalized_probs = [r['prob'] / total_prob for r in regime_probs]
        
        selected_regime = np.random.choice(regime_names, p=normalized_probs)
        regime_info = self.market_regimes[selected_regime]
        
        # Generate price and volatility
        base_price = symbol_info['avg_price'] * np.random.uniform(0.7, 1.3)
        
        # Volatility affected by multiple factors
        base_volatility = symbol_info['base_vol']
        vol_multiplier = regime_info['vol_multiplier']
        
        # VIX effect
        vix_effect = self.current_market_state.vix_level / 20.0
        
        # Earnings season effect
        earnings_effect = 1.2 if self.current_market_state.earnings_season else 1.0
        
        # Liquidity effect
        liquidity_effect = 1.0 / max(0.5, self.current_market_state.liquidity_conditions)
        
        final_volatility = base_volatility * vol_multiplier * vix_effect * earnings_effect * liquidity_effect
        final_volatility = max(0.1, min(2.0, final_volatility))  # Bounds
        
        # Generate returns with complex factors
        base_return = regime_info['return_drift'] / 252  # Daily return
        
        # Beta adjustment for market returns
        market_return = base_return
        adjusted_return = base_return + symbol_info['beta'] * (market_return - 0.02/252)  # Risk-free rate
        
        # Add noise and mean reversion
        return_noise = np.random.normal(0, final_volatility / np.sqrt(252))
        mean_reversion = -0.1 * (base_price - symbol_info['avg_price']) / symbol_info['avg_price'] / 252
        
        total_return = adjusted_return + return_noise + mean_reversion
        
        # Calculate technical indicators with noise
        rsi_base = 50
        if self.current_market_state.market_sentiment > 0:
            rsi_base += self.current_market_state.market_sentiment * 30
        rsi_base += total_return * 252 * 20  # Price momentum effect
        rsi_noise = np.random.normal(0, 8)
        rsi = max(10, min(90, rsi_base + rsi_noise))
        
        # Volume multiplier
        volume_base = 1.0
        if abs(total_return) > 0.02:  # Large moves increase volume
            volume_base *= (1 + abs(total_return) * 10)
        if self.current_market_state.earnings_season:
            volume_base *= 1.5
        volume_multiplier = volume_base * np.random.uniform(0.5, 2.0)
        
        return {
            'symbol': selected_symbol,
            'regime': selected_regime,
            'base_price': base_price,
            'price_change_pct': total_return * 100,
            'volatility': final_volatility,
            'rsi': rsi,
            'volume_multiplier': volume_multiplier,
            'current_price': base_price * (1 + total_return),
            
            # Market state factors
            'vix_level': self.current_market_state.vix_level,
            'market_sentiment': self.current_market_state.market_sentiment,
            'sector_rotation': self.current_market_state.sector_rotation,
            'fed_policy': self.current_market_state.fed_policy_stance,
            'earnings_season': self.current_market_state.earnings_season,
            'geopolitical_risk': self.current_market_state.geopolitical_risk,
            'liquidity_conditions': self.current_market_state.liquidity_conditions,
            'correlation_regime': self.current_market_state.correlation_regime,
            'volatility_regime': self.current_market_state.volatility_regime,
            
            # Economic factors
            'economic_phase': self.current_economic_cycle.phase,
            'inflation_rate': self.current_economic_cycle.inflation_rate,
            'interest_rates': self.current_economic_cycle.interest_rates,
            'unemployment_rate': self.current_economic_cycle.unemployment_rate,
            'gdp_growth': self.current_economic_cycle.gdp_growth,
            
            # Asset-specific
            'beta': symbol_info['beta'],
            'sector': symbol_info['sector']
        }
    
    def simulate_multiday_path(self, scenario: Dict, days: int = 21) -> List[Dict]:
        """Simulate complex multi-day price path with time-varying factors"""
        
        paths = []
        current_price = scenario['current_price']
        
        for day in range(days):
            # Time decay of initial momentum
            momentum_decay = math.exp(-day / 10.0)
            
            # Generate new daily factors
            daily_sentiment = scenario['market_sentiment'] * momentum_decay + np.random.normal(0, 0.1)
            daily_sentiment = np.clip(daily_sentiment, -1, 1)
            
            # VIX mean reversion
            vix_change = 0.05 * (20 - scenario['vix_level']) + np.random.normal(0, 2)
            daily_vix = max(10, min(60, scenario['vix_level'] + vix_change))
            
            # Daily return with multiple factors
            base_return = scenario['price_change_pct'] / 100 / 21 * momentum_decay  # Decay initial trend
            
            # Volatility clustering
            vol_clustering = np.random.normal(0, scenario['volatility'] / np.sqrt(252))
            if abs(vol_clustering) > 0.02:  # Large move increases next period volatility
                scenario['volatility'] *= 1.1
            else:
                scenario['volatility'] *= 0.99  # Mean revert volatility
            
            # Mean reversion to fair value
            fair_value = self.symbols[scenario['symbol']]['avg_price']
            mean_reversion = -0.02 * (current_price - fair_value) / fair_value / 21
            
            # Random shocks
            random_shock = np.random.normal(0, scenario['volatility'] / np.sqrt(252))
            
            # Combine all effects
            daily_return = base_return + vol_clustering + mean_reversion + random_shock
            
            # Update price
            new_price = current_price * (1 + daily_return)
            new_price = max(new_price, current_price * 0.8)  # Daily limit down
            new_price = min(new_price, current_price * 1.2)  # Daily limit up
            
            paths.append({
                'day': day,
                'price': new_price,
                'return': daily_return,
                'volatility': scenario['volatility'],
                'vix': daily_vix,
                'sentiment': daily_sentiment
            })
            
            current_price = new_price
            scenario['vix_level'] = daily_vix
            scenario['market_sentiment'] = daily_sentiment
        
        return paths

class AdvancedOptionsSimulator:
    """Advanced options P&L simulation with realistic Greeks behavior"""
    
    def __init__(self):
        self.commission_per_contract = 1.0
        
    def calculate_complex_pnl(self, strategy: str, contracts: List, 
                            entry_scenario: Dict, price_path: List[Dict],
                            exit_day: int) -> Dict:
        """Calculate P&L with complex Greeks evolution"""
        
        entry_price = entry_scenario['current_price']
        exit_price = price_path[exit_day]['price']
        
        # Strategy-specific P&L calculations
        if strategy == 'LONG_CALL':
            return self._calculate_long_call_pnl(contracts[0], entry_scenario, price_path, exit_day)
        elif strategy == 'LONG_PUT':
            return self._calculate_long_put_pnl(contracts[0], entry_scenario, price_path, exit_day)
        elif strategy == 'BULL_CALL_SPREAD':
            return self._calculate_spread_pnl(contracts, entry_scenario, price_path, exit_day, 'bull_call')
        elif strategy == 'BEAR_PUT_SPREAD':
            return self._calculate_spread_pnl(contracts, entry_scenario, price_path, exit_day, 'bear_put')
        elif strategy == 'STRADDLE':
            return self._calculate_straddle_pnl(contracts, entry_scenario, price_path, exit_day)
        else:
            return self._calculate_generic_pnl(entry_scenario, price_path, exit_day)
    
    def _calculate_long_call_pnl(self, contract: Dict, entry_scenario: Dict, 
                                price_path: List[Dict], exit_day: int) -> Dict:
        """Detailed long call P&L with Greeks evolution"""
        
        entry_price = entry_scenario['current_price']
        strike = contract['strike']
        exit_price = price_path[exit_day]['price']
        
        # Time to expiration effect
        initial_dte = 21
        exit_dte = initial_dte - exit_day
        time_decay_factor = exit_dte / initial_dte
        
        # Intrinsic value
        entry_intrinsic = max(0, entry_price - strike)
        exit_intrinsic = max(0, exit_price - strike)
        
        # Time value with volatility expansion/contraction
        entry_vol = entry_scenario['volatility']
        exit_vol = price_path[exit_day]['volatility']
        
        # Simplified Black-Scholes time value
        entry_time_value = self._estimate_time_value(entry_price, strike, initial_dte, entry_vol, 'call')
        exit_time_value = self._estimate_time_value(exit_price, strike, exit_dte, exit_vol, 'call')
        
        # Total option values
        entry_option_value = entry_intrinsic + entry_time_value
        exit_option_value = exit_intrinsic + exit_time_value
        
        # P&L calculation
        gross_pnl = (exit_option_value - entry_option_value) * 100  # Per contract
        net_pnl = gross_pnl - self.commission_per_contract * 2  # Buy and sell
        
        # Greeks effects
        delta_pnl = contract.get('delta', 0.5) * (exit_price - entry_price) * 100
        gamma_pnl = 0.5 * contract.get('gamma', 0.02) * ((exit_price - entry_price) ** 2) * 100
        theta_pnl = contract.get('theta', -0.05) * exit_day * 100
        vega_pnl = contract.get('vega', 0.2) * (exit_vol - entry_vol) * 100 * 100  # Vega per 1% vol change
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'entry_value': entry_option_value,
            'exit_value': exit_option_value,
            'entry_intrinsic': entry_intrinsic,
            'exit_intrinsic': exit_intrinsic,
            'max_risk': entry_option_value * 100 + self.commission_per_contract * 2
        }
    
    def _calculate_long_put_pnl(self, contract: Dict, entry_scenario: Dict,
                               price_path: List[Dict], exit_day: int) -> Dict:
        """Detailed long put P&L"""
        
        entry_price = entry_scenario['current_price']
        strike = contract['strike']
        exit_price = price_path[exit_day]['price']
        
        # Time factors
        initial_dte = 21
        exit_dte = initial_dte - exit_day
        
        # Intrinsic values
        entry_intrinsic = max(0, strike - entry_price)
        exit_intrinsic = max(0, strike - exit_price)
        
        # Time values
        entry_vol = entry_scenario['volatility']
        exit_vol = price_path[exit_day]['volatility']
        
        entry_time_value = self._estimate_time_value(entry_price, strike, initial_dte, entry_vol, 'put')
        exit_time_value = self._estimate_time_value(exit_price, strike, exit_dte, exit_vol, 'put')
        
        # Total values
        entry_option_value = entry_intrinsic + entry_time_value
        exit_option_value = exit_intrinsic + exit_time_value
        
        gross_pnl = (exit_option_value - entry_option_value) * 100
        net_pnl = gross_pnl - self.commission_per_contract * 2
        
        # Greeks effects
        delta_pnl = contract.get('delta', -0.5) * (exit_price - entry_price) * 100
        gamma_pnl = 0.5 * contract.get('gamma', 0.02) * ((exit_price - entry_price) ** 2) * 100
        theta_pnl = contract.get('theta', -0.05) * exit_day * 100
        vega_pnl = contract.get('vega', 0.2) * (exit_vol - entry_vol) * 100 * 100
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'entry_value': entry_option_value,
            'exit_value': exit_option_value,
            'max_risk': entry_option_value * 100 + self.commission_per_contract * 2
        }
    
    def _calculate_spread_pnl(self, contracts: List, entry_scenario: Dict,
                             price_path: List[Dict], exit_day: int, spread_type: str) -> Dict:
        """Calculate spread P&L (bull call or bear put)"""
        
        if len(contracts) < 2:
            return {'net_pnl': -100, 'max_risk': 100}
        
        long_contract = contracts[0]
        short_contract = contracts[1]
        
        # Calculate P&L for each leg
        if spread_type == 'bull_call':
            long_pnl = self._calculate_long_call_pnl(long_contract, entry_scenario, price_path, exit_day)
            short_pnl = self._calculate_long_call_pnl(short_contract, entry_scenario, price_path, exit_day)
            # Short leg has opposite P&L
            short_pnl['net_pnl'] *= -1
            short_pnl['gross_pnl'] *= -1
        else:  # bear_put
            long_pnl = self._calculate_long_put_pnl(long_contract, entry_scenario, price_path, exit_day)
            short_pnl = self._calculate_long_put_pnl(short_contract, entry_scenario, price_path, exit_day)
            # Short leg has opposite P&L
            short_pnl['net_pnl'] *= -1
            short_pnl['gross_pnl'] *= -1
        
        # Combine legs
        total_pnl = long_pnl['net_pnl'] + short_pnl['net_pnl']
        
        # Max risk for spread
        strike_diff = abs(long_contract['strike'] - short_contract['strike'])
        max_risk = strike_diff * 100 + self.commission_per_contract * 4  # 2 contracts each way
        
        return {
            'net_pnl': total_pnl,
            'gross_pnl': long_pnl['gross_pnl'] + short_pnl['gross_pnl'],
            'long_leg_pnl': long_pnl['net_pnl'],
            'short_leg_pnl': short_pnl['net_pnl'],
            'max_risk': max_risk
        }
    
    def _calculate_straddle_pnl(self, contracts: List, entry_scenario: Dict,
                               price_path: List[Dict], exit_day: int) -> Dict:
        """Calculate straddle P&L"""
        
        if len(contracts) < 2:
            return {'net_pnl': -100, 'max_risk': 100}
        
        call_contract = contracts[0] if contracts[0].get('type') == 'call' else contracts[1]
        put_contract = contracts[1] if contracts[1].get('type') == 'put' else contracts[0]
        
        call_pnl = self._calculate_long_call_pnl(call_contract, entry_scenario, price_path, exit_day)
        put_pnl = self._calculate_long_put_pnl(put_contract, entry_scenario, price_path, exit_day)
        
        total_pnl = call_pnl['net_pnl'] + put_pnl['net_pnl']
        max_risk = call_pnl.get('max_risk', 100) + put_pnl.get('max_risk', 100)
        
        return {
            'net_pnl': total_pnl,
            'call_pnl': call_pnl['net_pnl'],
            'put_pnl': put_pnl['net_pnl'],
            'max_risk': max_risk
        }
    
    def _calculate_generic_pnl(self, entry_scenario: Dict, price_path: List[Dict], exit_day: int) -> Dict:
        """Generic P&L for unknown strategies"""
        
        entry_price = entry_scenario['current_price']
        exit_price = price_path[exit_day]['price']
        
        # Simple directional estimate
        price_move = (exit_price - entry_price) / entry_price
        
        # Estimate P&L based on price movement and volatility
        base_pnl = price_move * 100 * 50  # Rough estimate
        vol_adjustment = entry_scenario['volatility'] * 20  # Volatility benefit/cost
        
        estimated_pnl = base_pnl + vol_adjustment - exit_day * 5  # Time decay
        
        return {
            'net_pnl': estimated_pnl,
            'max_risk': 100
        }
    
    def _estimate_time_value(self, price: float, strike: float, dte: int, vol: float, option_type: str) -> float:
        """Rough time value estimate using simplified Black-Scholes"""
        
        if dte <= 0:
            return 0
        
        # Very simplified time value
        time_factor = math.sqrt(dte / 365.0)
        vol_factor = vol * time_factor
        
        # ATM options have most time value
        moneyness = price / strike if option_type == 'call' else strike / price
        atm_factor = math.exp(-2 * ((moneyness - 1) ** 2))  # Peak at ATM
        
        time_value = price * vol_factor * atm_factor * 0.4
        return max(0, time_value)

async def run_advanced_monte_carlo(num_simulations: int = 2500) -> Dict:
    """Run advanced Monte Carlo simulation with complex scenarios"""
    
    print(f"ADVANCED MONTE CARLO PROFITABILITY TEST")
    print(f"=" * 70)
    print(f"Running {num_simulations:,} complex simulations...")
    print(f"Testing realistic market dynamics with:")
    print(f"  • Stochastic volatility and correlation regimes")
    print(f"  • Economic cycles and Fed policy changes")
    print(f"  • Sector rotation and earnings season effects")
    print(f"  • Geopolitical risks and liquidity conditions")
    print(f"  • Multi-day price paths with Greeks evolution")
    print(f"=" * 70)
    
    market_sim = AdvancedMarketSimulator()
    options_sim = AdvancedOptionsSimulator()
    
    results = []
    strategies_used = {
        'LONG_CALL': {'count': 0, 'total_pnl': 0, 'wins': 0},
        'LONG_PUT': {'count': 0, 'total_pnl': 0, 'wins': 0},
        'BULL_CALL_SPREAD': {'count': 0, 'total_pnl': 0, 'wins': 0},
        'BEAR_PUT_SPREAD': {'count': 0, 'total_pnl': 0, 'wins': 0},
        'STRADDLE': {'count': 0, 'total_pnl': 0, 'wins': 0},
        'NO_TRADE': {'count': 0, 'total_pnl': 0, 'wins': 0}
    }
    
    for sim_id in range(num_simulations):
        if (sim_id + 1) % 250 == 0:
            progress = (sim_id + 1) / num_simulations * 100
            print(f"Progress: {progress:.1f}% ({sim_id + 1:,}/{num_simulations:,})")
        
        try:
            # Generate complex scenario
            scenario = market_sim.generate_complex_scenario()
            
            # Determine strategy based on complex conditions
            strategy, contracts = determine_advanced_strategy(scenario)
            
            if strategy == 'NO_TRADE':
                strategies_used['NO_TRADE']['count'] += 1
                results.append({
                    'sim_id': sim_id,
                    'strategy': 'NO_TRADE',
                    'pnl': 0,
                    'scenario': scenario
                })
                continue
            
            # Generate multi-day price path
            trade_duration = random.randint(1, 21)  # 1-21 days
            price_path = market_sim.simulate_multiday_path(scenario, trade_duration)
            
            # Determine exit day (could be early due to stop/target)
            exit_day = determine_exit_day(strategy, contracts, scenario, price_path)
            exit_day = min(exit_day, len(price_path) - 1)
            
            # Calculate P&L
            pnl_result = options_sim.calculate_complex_pnl(
                strategy, contracts, scenario, price_path, exit_day
            )
            
            final_pnl = pnl_result['net_pnl']
            
            # Record results
            strategies_used[strategy]['count'] += 1
            strategies_used[strategy]['total_pnl'] += final_pnl
            if final_pnl > 0:
                strategies_used[strategy]['wins'] += 1
            
            results.append({
                'sim_id': sim_id,
                'strategy': strategy,
                'pnl': final_pnl,
                'exit_day': exit_day,
                'max_risk': pnl_result.get('max_risk', 100),
                'scenario': scenario,
                'pnl_details': pnl_result
            })
            
        except Exception as e:
            # Count errors as losses
            strategies_used['NO_TRADE']['count'] += 1
            results.append({
                'sim_id': sim_id,
                'strategy': 'ERROR',
                'pnl': -50,  # Penalty for errors
                'scenario': {'symbol': 'ERROR', 'regime': 'ERROR'}
            })
    
    return analyze_advanced_results(results, strategies_used)

def determine_advanced_strategy(scenario: Dict) -> Tuple[str, List[Dict]]:
    """Advanced strategy selection based on complex market conditions"""
    
    symbol = scenario['symbol']
    price = scenario['current_price']
    price_change = scenario['price_change_pct'] / 100
    volatility = scenario['volatility']
    vix = scenario['vix_level']
    sentiment = scenario['market_sentiment']
    earnings_season = scenario['earnings_season']
    sector = scenario['sector']
    regime = scenario['regime']
    
    # Create mock contracts
    atm_strike = round(price / 5) * 5  # Round to nearest $5
    
    call_contract = {
        'strike': atm_strike + 5,  # Slightly OTM
        'type': 'call',
        'delta': 0.4 + price_change * 2,  # Delta varies with momentum
        'gamma': 0.02 + volatility * 0.05,
        'theta': -0.05 - volatility * 0.1,
        'vega': 0.2 + volatility * 0.3
    }
    
    put_contract = {
        'strike': atm_strike - 5,  # Slightly OTM
        'type': 'put', 
        'delta': -0.4 - price_change * 2,
        'gamma': 0.02 + volatility * 0.05,
        'theta': -0.05 - volatility * 0.1,
        'vega': 0.2 + volatility * 0.3
    }
    
    # Complex strategy selection logic
    
    # 1. Strong bullish conditions - Long Call
    bullish_score = 0
    if price_change > 0.025:  # 2.5% up move
        bullish_score += 3
    if sentiment > 0.3:
        bullish_score += 2
    if regime in ['bull_market', 'volatile_bull']:
        bullish_score += 3
    if scenario['economic_phase'] in ['expansion', 'trough']:
        bullish_score += 1
    if scenario['fed_policy'] == 'dovish':
        bullish_score += 1
    if sector == 'tech' and scenario['sector_rotation'] == 'growth':
        bullish_score += 2
    
    if bullish_score >= 6 and volatility > 0.2:
        return 'LONG_CALL', [call_contract]
    
    # 2. Strong bearish conditions - Long Put
    bearish_score = 0
    if price_change < -0.025:  # 2.5% down move
        bearish_score += 3
    if sentiment < -0.3:
        bearish_score += 2
    if regime in ['bear_market', 'volatile_bear', 'crisis']:
        bearish_score += 3
    if scenario['economic_phase'] == 'contraction':
        bearish_score += 2
    if scenario['fed_policy'] == 'hawkish':
        bearish_score += 1
    if scenario['geopolitical_risk'] > 0.6:
        bearish_score += 2
    
    if bearish_score >= 6 and volatility > 0.2:
        return 'LONG_PUT', [put_contract]
    
    # 3. High volatility, uncertain direction - Straddle
    straddle_score = 0
    if vix > 25:
        straddle_score += 3
    if volatility > 0.35:
        straddle_score += 2
    if earnings_season and abs(price_change) < 0.01:
        straddle_score += 3
    if scenario['correlation_regime'] == 'high':  # Market stress
        straddle_score += 2
    if abs(sentiment) < 0.2:  # Neutral sentiment
        straddle_score += 1
    
    if straddle_score >= 6:
        atm_call = dict(call_contract)
        atm_put = dict(put_contract)
        atm_call['strike'] = atm_strike
        atm_put['strike'] = atm_strike
        return 'STRADDLE', [atm_call, atm_put]
    
    # 4. Moderate bullish - Bull Call Spread
    if 3 <= bullish_score < 6 and volatility < 0.4:
        long_call = dict(call_contract)
        short_call = dict(call_contract)
        short_call['strike'] = atm_strike + 10
        return 'BULL_CALL_SPREAD', [long_call, short_call]
    
    # 5. Moderate bearish - Bear Put Spread
    if 3 <= bearish_score < 6 and volatility < 0.4:
        long_put = dict(put_contract)
        short_put = dict(put_contract)
        long_put['strike'] = atm_strike
        short_put['strike'] = atm_strike - 10
        return 'BEAR_PUT_SPREAD', [long_put, short_put]
    
    # No suitable strategy found
    return 'NO_TRADE', []

def determine_exit_day(strategy: str, contracts: List, scenario: Dict, price_path: List[Dict]) -> int:
    """Determine when to exit the trade based on conditions"""
    
    entry_price = scenario['current_price']
    
    for day, path_data in enumerate(price_path):
        current_price = path_data['price']
        price_change = (current_price - entry_price) / entry_price
        
        # Profit target hit (20% gain)
        if abs(price_change) > 0.20:
            return day
        
        # Stop loss hit (50% option value loss, roughly)
        if strategy in ['LONG_CALL', 'LONG_PUT'] and price_change < -0.10:
            return day
        
        # Time-based exits
        if day >= 14:  # 14 days max hold
            return day
        
        # VIX spike exit for volatility plays
        if strategy == 'STRADDLE' and path_data.get('vix', 20) > 40:
            return day
    
    return len(price_path) - 1

def analyze_advanced_results(results: List[Dict], strategies_used: Dict) -> Dict:
    """Comprehensive analysis of advanced Monte Carlo results"""
    
    trading_results = [r for r in results if r['strategy'] not in ['NO_TRADE', 'ERROR']]
    
    if not trading_results:
        return {"error": "No trading results to analyze"}
    
    # Basic metrics
    total_simulations = len(results)
    total_trades = len(trading_results)
    winning_trades = [r for r in trading_results if r['pnl'] > 0]
    
    win_rate = len(winning_trades) / len(trading_results) * 100 if trading_results else 0
    avg_pnl = np.mean([r['pnl'] for r in trading_results])
    total_pnl = sum(r['pnl'] for r in trading_results)
    
    # Advanced metrics
    returns = [r['pnl'] for r in trading_results]
    volatility = np.std(returns)
    sharpe_ratio = avg_pnl / volatility if volatility > 0 else 0
    
    # Risk metrics
    max_loss = min(returns) if returns else 0
    max_gain = max(returns) if returns else 0
    
    # Drawdown analysis
    cumulative_pnl = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Strategy performance
    strategy_performance = {}
    for strategy, stats in strategies_used.items():
        if stats['count'] > 0:
            win_rate_strategy = stats['wins'] / stats['count'] * 100
            avg_pnl_strategy = stats['total_pnl'] / stats['count']
            
            strategy_performance[strategy] = {
                'trades': stats['count'],
                'win_rate': win_rate_strategy,
                'avg_pnl': avg_pnl_strategy,
                'total_pnl': stats['total_pnl']
            }
    
    # Market condition analysis
    regime_performance = {}
    for result in trading_results:
        regime = result['scenario'].get('regime', 'unknown')
        if regime not in regime_performance:
            regime_performance[regime] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        
        regime_performance[regime]['trades'] += 1
        regime_performance[regime]['total_pnl'] += result['pnl']
        if result['pnl'] > 0:
            regime_performance[regime]['wins'] += 1
    
    # Calculate regime stats
    for regime, stats in regime_performance.items():
        if stats['trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['trades'] * 100
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
    
    return {
        'summary': {
            'total_simulations': total_simulations,
            'total_trades': total_trades,
            'trading_rate': total_trades / total_simulations * 100,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        },
        'strategy_performance': strategy_performance,
        'regime_performance': regime_performance,
        'detailed_results': trading_results
    }

def print_advanced_results(analysis: Dict):
    """Print comprehensive results of advanced Monte Carlo test"""
    
    summary = analysis['summary']
    
    print(f"\nADVANCED MONTE CARLO RESULTS")
    print(f"=" * 70)
    
    print(f"EXECUTION SUMMARY:")
    print(f"  Total Simulations: {summary['total_simulations']:,}")
    print(f"  Trading Opportunities: {summary['total_trades']:,} ({summary['trading_rate']:.1f}%)")
    print(f"  Win Rate: {summary['win_rate']:.1f}%")
    print(f"  Average P&L per Trade: ${summary['avg_pnl']:.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    
    print(f"\nRISK METRICS:")
    print(f"  Best Trade: ${summary['max_gain']:.2f}")
    print(f"  Worst Trade: ${summary['max_loss']:.2f}")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: ${summary['max_drawdown']:.2f}")
    
    # Strategy performance
    print(f"\nSTRATEGY PERFORMANCE:")
    for strategy, stats in analysis['strategy_performance'].items():
        if stats['trades'] > 10:  # Only show strategies with meaningful sample size
            print(f"  {strategy:20} {stats['trades']:4d} trades | {stats['win_rate']:5.1f}% wins | ${stats['avg_pnl']:7.2f} avg")
    
    # Market regime performance
    print(f"\nMARKET REGIME PERFORMANCE:")
    for regime, stats in analysis['regime_performance'].items():
        if stats['trades'] > 10:
            print(f"  {regime:20} {stats['trades']:4d} trades | {stats['win_rate']:5.1f}% wins | ${stats['avg_pnl']:7.2f} avg")
    
    # Overall assessment
    win_rate = summary['win_rate']
    avg_pnl = summary['avg_pnl']
    sharpe = summary['sharpe_ratio']
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"=" * 40)
    
    if win_rate >= 60 and avg_pnl > 25 and sharpe > 0.5:
        rating = "A+"
        assessment = "EXCEPTIONAL - Outstanding performance across all metrics"
    elif win_rate >= 55 and avg_pnl > 20 and sharpe > 0.4:
        rating = "A"
        assessment = "EXCELLENT - Strong profitability with good risk management"
    elif win_rate >= 50 and avg_pnl > 15 and sharpe > 0.3:
        rating = "B+"
        assessment = "VERY GOOD - Solid returns with reasonable risk"
    elif win_rate >= 45 and avg_pnl > 8 and sharpe > 0.2:
        rating = "B"
        assessment = "GOOD - Profitable but could be optimized"
    elif win_rate >= 40 and avg_pnl > 0:
        rating = "C+"
        assessment = "MARGINAL - Barely profitable, needs improvement"
    else:
        rating = "D"
        assessment = "POOR - Unprofitable, major changes needed"
    
    print(f"Rating: {rating}")
    print(f"Assessment: {assessment}")
    
    # Recommendations
    monthly_return = avg_pnl * 22  # 22 trading days per month
    annual_return = monthly_return * 12
    
    print(f"\nPROJECTED RETURNS:")
    print(f"  Expected Monthly: ${monthly_return:,.2f}")
    print(f"  Expected Annual: ${annual_return:,.2f}")
    
    if rating in ['A+', 'A']:
        print(f"\n[RECOMMENDATION] Deploy with confidence!")
        print(f"  This bot shows exceptional performance across diverse conditions")
        print(f"  Consider increasing position sizes for higher returns")
    elif rating in ['B+', 'B']:
        print(f"\n[RECOMMENDATION] Deploy with monitoring")
        print(f"  Good performance but monitor for changing market conditions")
        print(f"  Consider risk management enhancements")
    else:
        print(f"\n[RECOMMENDATION] Optimize before deployment")
        print(f"  Performance needs improvement before live trading")
        print(f"  Focus on strategy selection and risk management")
    
    print(f"=" * 70)

async def main():
    """Run advanced Monte Carlo test"""
    
    num_sims = 2500  # Large sample for statistical significance
    
    print("Starting Advanced Monte Carlo Simulation...")
    print("This may take a few minutes due to complex calculations...")
    
    start_time = datetime.now()
    
    analysis = await run_advanced_monte_carlo(num_sims)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nSimulation completed in {duration:.1f} seconds")
    
    print_advanced_results(analysis)
    
    # Save results
    df_results = pd.DataFrame(analysis['detailed_results'])
    df_results.to_csv('advanced_monte_carlo_results.csv', index=False)
    print(f"\nDetailed results saved to: advanced_monte_carlo_results.csv")

if __name__ == "__main__":
    asyncio.run(main())