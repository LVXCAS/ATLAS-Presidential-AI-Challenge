"""
Options Volatility Agent - LangGraph Implementation

This agent implements a comprehensive options volatility trading strategy that:
1. Analyzes implied volatility surfaces and detects skew anomalies
2. Integrates earnings calendar for event-driven strategies
3. Calculates Greeks for risk management
4. Detects volatility regime changes
5. Generates explainable options trading signals

The agent operates autonomously within the LangGraph framework, communicating
with other agents and making real-time trading decisions based on options
market inefficiencies and volatility patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"

class OptionsStrategy(Enum):
    """Options strategy types"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"

@dataclass
class OptionsData:
    """Options chain data structure"""
    symbol: str
    expiration: datetime
    strike: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    time_to_expiration: float

@dataclass
class IVSurfacePoint:
    """Implied volatility surface data point"""
    strike: float
    expiration: datetime
    time_to_expiration: float
    moneyness: float  # strike / underlying_price
    implied_volatility: float
    delta: float
    volume: int
    open_interest: int

@dataclass
class VolatilitySkew:
    """Volatility skew analysis results"""
    symbol: str
    expiration: datetime
    skew_slope: float
    skew_convexity: float
    put_call_skew: float
    term_structure_slope: float
    skew_anomaly_score: float
    is_anomalous: bool

@dataclass
class EarningsEvent:
    """Earnings event data"""
    symbol: str
    earnings_date: datetime
    days_to_earnings: int
    expected_move: float
    iv_rank: float
    iv_percentile: float
    historical_earnings_moves: List[float]
    strategy_recommendation: OptionsStrategy

@dataclass
class GreeksRisk:
    """Greeks-based risk metrics"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    delta_neutral: bool
    gamma_risk_level: str
    vega_exposure: float
    theta_decay_daily: float

@dataclass
class OptionsSignal:
    """Options trading signal with explainability"""
    signal_type: str
    symbol: str
    strategy: OptionsStrategy
    value: float  # Signal strength [-1, 1]
    confidence: float  # [0, 1]
    top_3_reasons: List[Dict[str, Any]]
    timestamp: datetime
    model_version: str
    expiration: datetime
    strike: float
    option_type: str
    entry_price: float
    target_profit: float
    stop_loss: float
    max_risk: float
    expected_return: float
    greeks: GreeksRisk
    iv_analysis: Dict[str, Any]
    volatility_regime: VolatilityRegime

class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return max(price, 0)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def calculate_implied_volatility(market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Brent's method"""
        if T <= 0:
            return 0.0
        
        def objective(sigma):
            try:
                theoretical_price = BlackScholesCalculator.calculate_option_price(
                    S, K, T, r, sigma, option_type
                )
                return abs(theoretical_price - market_price)
            except:
                return float('inf')
        
        try:
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
            return result.x if result.success else 0.2
        except:
            return 0.2

class OptionsVolatilityAgent:
    """
    LangGraph Options Volatility Agent
    
    Implements sophisticated options trading strategies based on:
    - IV surface analysis and skew detection
    - Earnings calendar integration
    - Greeks calculation and risk management
    - Volatility regime detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model_version = "1.0.0"
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.bs_calculator = BlackScholesCalculator()
        
        # Volatility regime thresholds
        self.vol_regime_thresholds = {
            'low': 0.15,      # Below 15% IV
            'normal': 0.30,   # 15-30% IV
            'high': 0.50,     # 30-50% IV
            'extreme': 0.50   # Above 50% IV
        }
        
        # Risk management parameters
        self.max_position_size = 0.05  # 5% of portfolio per position
        self.max_vega_exposure = 1000  # Maximum vega exposure
        self.min_liquidity_threshold = 100  # Minimum open interest
        
        logger.info("Options Volatility Agent initialized")
    
    async def analyze_iv_surface(self, symbol: str, options_data: List[OptionsData]) -> Dict[str, Any]:
        """
        Analyze implied volatility surface for anomalies and opportunities
        """
        try:
            if not options_data:
                return {'error': 'No options data available'}
            
            # Convert to surface points
            surface_points = []
            underlying_price = options_data[0].underlying_price
            
            for option in options_data:
                if option.implied_volatility > 0 and option.time_to_expiration > 0:
                    surface_points.append(IVSurfacePoint(
                        strike=option.strike,
                        expiration=option.expiration,
                        time_to_expiration=option.time_to_expiration,
                        moneyness=option.strike / underlying_price,
                        implied_volatility=option.implied_volatility,
                        delta=option.delta,
                        volume=option.volume,
                        open_interest=option.open_interest
                    ))
            
            if not surface_points:
                return {'error': 'No valid surface points'}
            
            # Analyze volatility skew
            skew_analysis = await self._analyze_volatility_skew(symbol, surface_points)
            
            # Detect arbitrage opportunities
            arbitrage_opportunities = await self._detect_volatility_arbitrage(surface_points)
            
            # Calculate surface metrics
            surface_metrics = self._calculate_surface_metrics(surface_points)
            
            return {
                'symbol': symbol,
                'surface_points': len(surface_points),
                'skew_analysis': skew_analysis,
                'arbitrage_opportunities': arbitrage_opportunities,
                'surface_metrics': surface_metrics,
                'analysis_timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing IV surface for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _analyze_volatility_skew(self, symbol: str, surface_points: List[IVSurfacePoint]) -> List[VolatilitySkew]:
        """Analyze volatility skew patterns"""
        skew_results = []
        
        # Group by expiration
        expirations = {}
        for point in surface_points:
            exp_key = point.expiration.strftime('%Y-%m-%d')
            if exp_key not in expirations:
                expirations[exp_key] = []
            expirations[exp_key].append(point)
        
        for exp_date, points in expirations.items():
            if len(points) < 3:  # Need at least 3 points for skew analysis
                continue
            
            # Sort by moneyness
            points.sort(key=lambda x: x.moneyness)
            
            # Calculate skew metrics
            moneyness = [p.moneyness for p in points]
            iv_values = [p.implied_volatility for p in points]
            
            # Skew slope (linear regression)
            if len(moneyness) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(moneyness, iv_values)
                
                # Convexity (second derivative approximation)
                convexity = 0
                if len(points) >= 3:
                    mid_idx = len(points) // 2
                    if mid_idx > 0 and mid_idx < len(points) - 1:
                        convexity = (iv_values[mid_idx + 1] - 2 * iv_values[mid_idx] + iv_values[mid_idx - 1])
                
                # Put-call skew
                atm_idx = min(range(len(moneyness)), key=lambda i: abs(moneyness[i] - 1.0))
                otm_put_iv = iv_values[0] if moneyness[0] < 1.0 else iv_values[-1]
                otm_call_iv = iv_values[-1] if moneyness[-1] > 1.0 else iv_values[0]
                put_call_skew = otm_put_iv - otm_call_iv
                
                # Anomaly detection
                skew_anomaly_score = abs(slope) + abs(convexity) + abs(put_call_skew)
                is_anomalous = skew_anomaly_score > 0.5  # Threshold for anomaly
                
                skew_result = VolatilitySkew(
                    symbol=symbol,
                    expiration=datetime.strptime(exp_date, '%Y-%m-%d'),
                    skew_slope=slope,
                    skew_convexity=convexity,
                    put_call_skew=put_call_skew,
                    term_structure_slope=0,  # Will be calculated separately
                    skew_anomaly_score=skew_anomaly_score,
                    is_anomalous=is_anomalous
                )
                
                skew_results.append(skew_result)
        
        return skew_results
    
    async def _detect_volatility_arbitrage(self, surface_points: List[IVSurfacePoint]) -> List[Dict[str, Any]]:
        """Detect volatility arbitrage opportunities"""
        opportunities = []
        
        # Look for calendar spread opportunities
        for i, point1 in enumerate(surface_points):
            for j, point2 in enumerate(surface_points[i+1:], i+1):
                # Same strike, different expirations
                if (abs(point1.strike - point2.strike) < 0.01 and 
                    point1.expiration != point2.expiration):
                    
                    # Check for inverted term structure
                    if point1.time_to_expiration < point2.time_to_expiration:
                        short_term_iv = point1.implied_volatility
                        long_term_iv = point2.implied_volatility
                        
                        # Arbitrage if short-term IV > long-term IV significantly
                        if short_term_iv > long_term_iv + 0.05:  # 5% threshold
                            opportunities.append({
                                'type': 'calendar_spread_arbitrage',
                                'strike': point1.strike,
                                'short_expiration': point1.expiration,
                                'long_expiration': point2.expiration,
                                'short_iv': short_term_iv,
                                'long_iv': long_term_iv,
                                'iv_difference': short_term_iv - long_term_iv,
                                'confidence': min((short_term_iv - long_term_iv) / 0.1, 1.0)
                            })
        
        return opportunities
    
    def _calculate_surface_metrics(self, surface_points: List[IVSurfacePoint]) -> Dict[str, float]:
        """Calculate overall surface metrics"""
        if not surface_points:
            return {}
        
        iv_values = [p.implied_volatility for p in surface_points]
        
        return {
            'average_iv': np.mean(iv_values),
            'iv_std': np.std(iv_values),
            'min_iv': np.min(iv_values),
            'max_iv': np.max(iv_values),
            'iv_range': np.max(iv_values) - np.min(iv_values),
            'total_volume': sum(p.volume for p in surface_points),
            'total_open_interest': sum(p.open_interest for p in surface_points)
        }
    
    async def integrate_earnings_calendar(self, symbol: str, options_data: List[OptionsData]) -> Optional[EarningsEvent]:
        """
        Integrate earnings calendar for event-driven strategies
        """
        try:
            # Get earnings date (simplified - in production would use earnings API)
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return None
            
            # Get next earnings date
            earnings_date = pd.to_datetime(calendar.index[0])
            days_to_earnings = (earnings_date - datetime.now()).days
            
            if days_to_earnings < 0 or days_to_earnings > 60:
                return None  # Only consider earnings within 60 days
            
            # Calculate expected move from straddle prices
            underlying_price = options_data[0].underlying_price if options_data else 100
            expected_move = self._calculate_expected_earnings_move(options_data, underlying_price)
            
            # Calculate IV rank and percentile
            iv_values = [opt.implied_volatility for opt in options_data if opt.implied_volatility > 0]
            current_iv = np.mean(iv_values) if iv_values else 0.3
            
            # Simplified IV rank calculation
            iv_rank = min(current_iv / 0.5, 1.0)  # Normalize to 0-1
            iv_percentile = iv_rank * 100
            
            # Historical earnings moves (simplified)
            historical_moves = [0.05, 0.08, 0.12, 0.06, 0.09]  # Mock data
            
            # Strategy recommendation
            strategy_recommendation = self._recommend_earnings_strategy(
                days_to_earnings, expected_move, iv_rank, historical_moves
            )
            
            return EarningsEvent(
                symbol=symbol,
                earnings_date=earnings_date,
                days_to_earnings=days_to_earnings,
                expected_move=expected_move,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                historical_earnings_moves=historical_moves,
                strategy_recommendation=strategy_recommendation
            )
            
        except Exception as e:
            logger.error(f"Error integrating earnings calendar for {symbol}: {e}")
            return None
    
    def _calculate_expected_earnings_move(self, options_data: List[OptionsData], underlying_price: float) -> float:
        """Calculate expected earnings move from straddle prices"""
        if not options_data:
            return 0.05  # Default 5% move
        
        # Find ATM options closest to earnings
        atm_options = [opt for opt in options_data 
                      if abs(opt.strike - underlying_price) / underlying_price < 0.05]
        
        if not atm_options:
            return 0.05
        
        # Get closest expiration after earnings
        earnings_options = [opt for opt in atm_options 
                           if opt.time_to_expiration > 0.02]  # At least 1 week
        
        if not earnings_options:
            return 0.05
        
        # Calculate straddle price
        closest_strike = min(earnings_options, key=lambda x: abs(x.strike - underlying_price)).strike
        call_price = next((opt.last_price for opt in earnings_options 
                          if opt.strike == closest_strike and opt.option_type == 'call'), 0)
        put_price = next((opt.last_price for opt in earnings_options 
                         if opt.strike == closest_strike and opt.option_type == 'put'), 0)
        
        straddle_price = call_price + put_price
        expected_move = straddle_price / underlying_price if underlying_price > 0 else 0.05
        
        return min(expected_move, 0.3)  # Cap at 30%
    
    def _recommend_earnings_strategy(self, days_to_earnings: int, expected_move: float, 
                                   iv_rank: float, historical_moves: List[float]) -> OptionsStrategy:
        """Recommend earnings strategy based on analysis"""
        avg_historical_move = np.mean(historical_moves) if historical_moves else 0.08
        
        # High IV + Low expected move relative to historical = Sell premium
        if iv_rank > 0.7 and expected_move < avg_historical_move * 0.8:
            if days_to_earnings > 7:
                return OptionsStrategy.IRON_CONDOR
            else:
                return OptionsStrategy.SHORT_PUT
        
        # Low IV + High expected move relative to historical = Buy premium
        elif iv_rank < 0.3 and expected_move > avg_historical_move * 1.2:
            return OptionsStrategy.STRADDLE
        
        # Moderate conditions
        elif days_to_earnings > 14:
            return OptionsStrategy.CALENDAR_SPREAD
        else:
            return OptionsStrategy.STRANGLE
    
    async def calculate_greeks_risk(self, positions: List[OptionsData]) -> GreeksRisk:
        """
        Calculate portfolio Greeks and risk metrics
        """
        try:
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            total_rho = 0
            
            for position in positions:
                # Use provided Greeks or calculate them
                if all(hasattr(position, greek) for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']):
                    total_delta += position.delta
                    total_gamma += position.gamma
                    total_theta += position.theta
                    total_vega += position.vega
                    total_rho += position.rho
                else:
                    # Calculate Greeks using Black-Scholes
                    greeks = self.bs_calculator.calculate_greeks(
                        S=position.underlying_price,
                        K=position.strike,
                        T=position.time_to_expiration,
                        r=self.risk_free_rate,
                        sigma=position.implied_volatility,
                        option_type=position.option_type
                    )
                    
                    total_delta += greeks['delta']
                    total_gamma += greeks['gamma']
                    total_theta += greeks['theta']
                    total_vega += greeks['vega']
                    total_rho += greeks['rho']
            
            # Risk assessments
            delta_neutral = abs(total_delta) < 0.1
            
            if abs(total_gamma) < 0.01:
                gamma_risk_level = "low"
            elif abs(total_gamma) < 0.05:
                gamma_risk_level = "medium"
            else:
                gamma_risk_level = "high"
            
            vega_exposure = abs(total_vega)
            theta_decay_daily = total_theta
            
            return GreeksRisk(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                total_rho=total_rho,
                delta_neutral=delta_neutral,
                gamma_risk_level=gamma_risk_level,
                vega_exposure=vega_exposure,
                theta_decay_daily=theta_decay_daily
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks risk: {e}")
            return GreeksRisk(0, 0, 0, 0, 0, True, "unknown", 0, 0)
    
    async def detect_volatility_regime(self, symbol: str, lookback_days: int = 30) -> VolatilityRegime:
        """
        Detect current volatility regime
        """
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=f"{lookback_days}d")
            
            if hist_data.empty:
                return VolatilityRegime.NORMAL_VOL
            
            # Calculate realized volatility
            returns = hist_data['Close'].pct_change().dropna()
            realized_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Get current implied volatility (simplified)
            # In production, would get from options data
            current_iv = realized_vol * 1.2  # Approximate IV from RV
            
            # Classify regime
            if current_iv < self.vol_regime_thresholds['low']:
                return VolatilityRegime.LOW_VOL
            elif current_iv < self.vol_regime_thresholds['normal']:
                return VolatilityRegime.NORMAL_VOL
            elif current_iv < self.vol_regime_thresholds['high']:
                return VolatilityRegime.HIGH_VOL
            else:
                return VolatilityRegime.EXTREME_VOL
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime for {symbol}: {e}")
            return VolatilityRegime.NORMAL_VOL
    
    async def generate_options_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[OptionsSignal]:
        """
        Generate comprehensive options trading signals with explainability
        """
        try:
            signals = []
            
            # Get options data (mock for now)
            options_data = await self._get_options_data(symbol)
            if not options_data:
                return signals
            
            # Analyze IV surface
            iv_analysis = await self.analyze_iv_surface(symbol, options_data)
            
            # Check earnings calendar
            earnings_event = await self.integrate_earnings_calendar(symbol, options_data)
            
            # Detect volatility regime
            vol_regime = await self.detect_volatility_regime(symbol)
            
            # Calculate Greeks risk
            greeks_risk = await self.calculate_greeks_risk(options_data)
            
            # Generate signals based on analysis
            
            # 1. Volatility Arbitrage Signal
            if 'arbitrage_opportunities' in iv_analysis:
                for arb_opp in iv_analysis['arbitrage_opportunities']:
                    if arb_opp['confidence'] > 0.7:
                        signal = await self._create_arbitrage_signal(
                            symbol, arb_opp, iv_analysis, vol_regime, greeks_risk
                        )
                        if signal:
                            signals.append(signal)
            
            # 2. Earnings Play Signal
            if earnings_event and earnings_event.days_to_earnings <= 30:
                signal = await self._create_earnings_signal(
                    symbol, earnings_event, iv_analysis, vol_regime, greeks_risk
                )
                if signal:
                    signals.append(signal)
            
            # 3. Volatility Regime Signal
            signal = await self._create_regime_signal(
                symbol, vol_regime, iv_analysis, greeks_risk, options_data
            )
            if signal:
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating options signals for {symbol}: {e}")
            return []
    
    async def _get_options_data(self, symbol: str) -> List[OptionsData]:
        """Get options chain data (mock implementation)"""
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Mock options data
            options_data = []
            strikes = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.05)
            expirations = [datetime.now() + timedelta(days=d) for d in [7, 14, 30, 60]]
            
            for exp in expirations:
                for strike in strikes:
                    tte = (exp - datetime.now()).days / 365.0
                    
                    for option_type in ['call', 'put']:
                        # Mock implied volatility
                        moneyness = strike / current_price
                        base_iv = 0.25
                        if moneyness < 0.95:  # OTM puts
                            iv = base_iv + 0.05
                        elif moneyness > 1.05:  # OTM calls
                            iv = base_iv + 0.02
                        else:  # ATM
                            iv = base_iv
                        
                        # Calculate theoretical price and Greeks
                        theo_price = self.bs_calculator.calculate_option_price(
                            current_price, strike, tte, self.risk_free_rate, iv, option_type
                        )
                        
                        greeks = self.bs_calculator.calculate_greeks(
                            current_price, strike, tte, self.risk_free_rate, iv, option_type
                        )
                        
                        option = OptionsData(
                            symbol=symbol,
                            expiration=exp,
                            strike=strike,
                            option_type=option_type,
                            bid=theo_price * 0.98,
                            ask=theo_price * 1.02,
                            last_price=theo_price,
                            volume=np.random.randint(10, 1000),
                            open_interest=np.random.randint(100, 5000),
                            implied_volatility=iv,
                            delta=greeks['delta'],
                            gamma=greeks['gamma'],
                            theta=greeks['theta'],
                            vega=greeks['vega'],
                            rho=greeks['rho'],
                            underlying_price=current_price,
                            time_to_expiration=tte
                        )
                        
                        options_data.append(option)
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options data for {symbol}: {e}")
            return []
    
    async def _create_arbitrage_signal(self, symbol: str, arb_opp: Dict[str, Any], 
                                     iv_analysis: Dict[str, Any], vol_regime: VolatilityRegime,
                                     greeks_risk: GreeksRisk) -> Optional[OptionsSignal]:
        """Create volatility arbitrage signal"""
        try:
            confidence = arb_opp['confidence']
            signal_strength = min(arb_opp['iv_difference'] / 0.1, 1.0)
            
            top_3_reasons = [
                {
                    'rank': 1,
                    'factor': 'IV Surface Arbitrage',
                    'contribution': 0.6,
                    'explanation': f"Calendar spread arbitrage detected with {arb_opp['iv_difference']:.1%} IV difference",
                    'confidence': confidence,
                    'supporting_data': {
                        'short_iv': arb_opp['short_iv'],
                        'long_iv': arb_opp['long_iv'],
                        'strike': arb_opp['strike']
                    }
                },
                {
                    'rank': 2,
                    'factor': 'Volatility Regime',
                    'contribution': 0.25,
                    'explanation': f"Current volatility regime: {vol_regime.value}",
                    'confidence': 0.8,
                    'supporting_data': {'regime': vol_regime.value}
                },
                {
                    'rank': 3,
                    'factor': 'Greeks Risk',
                    'contribution': 0.15,
                    'explanation': f"Portfolio vega exposure: {greeks_risk.vega_exposure:.0f}",
                    'confidence': 0.7,
                    'supporting_data': {'vega_exposure': greeks_risk.vega_exposure}
                }
            ]
            
            return OptionsSignal(
                signal_type='volatility_arbitrage',
                symbol=symbol,
                strategy=OptionsStrategy.CALENDAR_SPREAD,
                value=signal_strength,
                confidence=confidence,
                top_3_reasons=top_3_reasons,
                timestamp=datetime.utcnow(),
                model_version=self.model_version,
                expiration=arb_opp['long_expiration'],
                strike=arb_opp['strike'],
                option_type='call',
                entry_price=0,  # Would be calculated based on spread
                target_profit=arb_opp['iv_difference'] * 100,
                stop_loss=arb_opp['iv_difference'] * 50,
                max_risk=arb_opp['iv_difference'] * 200,
                expected_return=arb_opp['iv_difference'] * 0.5,
                greeks=greeks_risk,
                iv_analysis=iv_analysis,
                volatility_regime=vol_regime
            )
            
        except Exception as e:
            logger.error(f"Error creating arbitrage signal: {e}")
            return None
    
    async def _create_earnings_signal(self, symbol: str, earnings_event: EarningsEvent,
                                    iv_analysis: Dict[str, Any], vol_regime: VolatilityRegime,
                                    greeks_risk: GreeksRisk) -> Optional[OptionsSignal]:
        """Create earnings-based signal"""
        try:
            # Signal strength based on IV rank and expected move
            signal_strength = min(earnings_event.iv_rank + earnings_event.expected_move, 1.0)
            confidence = 0.8 if earnings_event.days_to_earnings <= 7 else 0.6
            
            top_3_reasons = [
                {
                    'rank': 1,
                    'factor': 'Earnings Event',
                    'contribution': 0.5,
                    'explanation': f"Earnings in {earnings_event.days_to_earnings} days, expected move {earnings_event.expected_move:.1%}",
                    'confidence': confidence,
                    'supporting_data': {
                        'days_to_earnings': earnings_event.days_to_earnings,
                        'expected_move': earnings_event.expected_move
                    }
                },
                {
                    'rank': 2,
                    'factor': 'IV Rank',
                    'contribution': 0.3,
                    'explanation': f"IV rank at {earnings_event.iv_percentile:.0f}th percentile",
                    'confidence': 0.7,
                    'supporting_data': {'iv_percentile': earnings_event.iv_percentile}
                },
                {
                    'rank': 3,
                    'factor': 'Historical Pattern',
                    'contribution': 0.2,
                    'explanation': f"Average historical earnings move: {np.mean(earnings_event.historical_earnings_moves):.1%}",
                    'confidence': 0.6,
                    'supporting_data': {'historical_moves': earnings_event.historical_earnings_moves}
                }
            ]
            
            return OptionsSignal(
                signal_type='earnings_play',
                symbol=symbol,
                strategy=earnings_event.strategy_recommendation,
                value=signal_strength,
                confidence=confidence,
                top_3_reasons=top_3_reasons,
                timestamp=datetime.utcnow(),
                model_version=self.model_version,
                expiration=earnings_event.earnings_date + timedelta(days=7),
                strike=0,  # Would be determined by strategy
                option_type='both',
                entry_price=0,
                target_profit=earnings_event.expected_move * 0.5,
                stop_loss=earnings_event.expected_move * 0.25,
                max_risk=earnings_event.expected_move * 1.0,
                expected_return=earnings_event.expected_move * 0.3,
                greeks=greeks_risk,
                iv_analysis=iv_analysis,
                volatility_regime=vol_regime
            )
            
        except Exception as e:
            logger.error(f"Error creating earnings signal: {e}")
            return None
    
    async def _create_regime_signal(self, symbol: str, vol_regime: VolatilityRegime,
                                  iv_analysis: Dict[str, Any], greeks_risk: GreeksRisk,
                                  options_data: List[OptionsData]) -> Optional[OptionsSignal]:
        """Create volatility regime-based signal"""
        try:
            # Strategy selection based on regime
            if vol_regime == VolatilityRegime.LOW_VOL:
                strategy = OptionsStrategy.LONG_CALL
                signal_strength = 0.6
                explanation = "Low volatility regime favors buying premium"
            elif vol_regime == VolatilityRegime.HIGH_VOL:
                strategy = OptionsStrategy.SHORT_PUT
                signal_strength = 0.7
                explanation = "High volatility regime favors selling premium"
            elif vol_regime == VolatilityRegime.EXTREME_VOL:
                strategy = OptionsStrategy.IRON_CONDOR
                signal_strength = 0.8
                explanation = "Extreme volatility regime favors range-bound strategies"
            else:
                strategy = OptionsStrategy.STRADDLE
                signal_strength = 0.5
                explanation = "Normal volatility regime allows directional plays"
            
            confidence = 0.7
            
            # Get average IV for analysis
            avg_iv = np.mean([opt.implied_volatility for opt in options_data]) if options_data else 0.25
            
            top_3_reasons = [
                {
                    'rank': 1,
                    'factor': 'Volatility Regime',
                    'contribution': 0.6,
                    'explanation': explanation,
                    'confidence': confidence,
                    'supporting_data': {'regime': vol_regime.value, 'avg_iv': avg_iv}
                },
                {
                    'rank': 2,
                    'factor': 'IV Surface Analysis',
                    'contribution': 0.25,
                    'explanation': f"Surface shows {len(iv_analysis.get('surface_points', []))} data points",
                    'confidence': 0.6,
                    'supporting_data': iv_analysis.get('surface_metrics', {})
                },
                {
                    'rank': 3,
                    'factor': 'Risk Management',
                    'contribution': 0.15,
                    'explanation': f"Current portfolio delta: {greeks_risk.total_delta:.2f}",
                    'confidence': 0.8,
                    'supporting_data': {'total_delta': greeks_risk.total_delta}
                }
            ]
            
            return OptionsSignal(
                signal_type='volatility_regime',
                symbol=symbol,
                strategy=strategy,
                value=signal_strength,
                confidence=confidence,
                top_3_reasons=top_3_reasons,
                timestamp=datetime.utcnow(),
                model_version=self.model_version,
                expiration=datetime.now() + timedelta(days=30),
                strike=0,
                option_type='call' if strategy in [OptionsStrategy.LONG_CALL, OptionsStrategy.SHORT_CALL] else 'put',
                entry_price=0,
                target_profit=0.2,
                stop_loss=0.1,
                max_risk=0.3,
                expected_return=0.15,
                greeks=greeks_risk,
                iv_analysis=iv_analysis,
                volatility_regime=vol_regime
            )
            
        except Exception as e:
            logger.error(f"Error creating regime signal: {e}")
            return None

# LangGraph Integration Functions
async def options_volatility_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for Options Volatility Agent
    """
    try:
        agent = OptionsVolatilityAgent()
        
        # Extract relevant data from state
        market_data = state.get('market_data', {})
        symbols = list(market_data.keys()) if market_data else ['AAPL']  # Default symbol
        
        all_signals = []
        
        for symbol in symbols[:5]:  # Limit to 5 symbols for performance
            signals = await agent.generate_options_signals(symbol, market_data.get(symbol, {}))
            all_signals.extend(signals)
        
        # Update state with options signals
        current_signals = state.get('signals', {})
        current_signals['options_volatility'] = [asdict(signal) for signal in all_signals]
        
        return {
            **state,
            'signals': current_signals,
            'options_analysis': {
                'agent': 'options_volatility',
                'signals_generated': len(all_signals),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in options volatility agent node: {e}")
        return {
            **state,
            'system_alerts': state.get('system_alerts', []) + [{
                'type': 'agent_error',
                'agent': 'options_volatility',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }]
        }

# Example usage and testing
async def main():
    """Example usage of the Options Volatility Agent"""
    agent = OptionsVolatilityAgent()
    
    # Test with AAPL
    symbol = "AAPL"
    market_data = {'current_price': 150.0, 'volume': 1000000}
    
    print(f"Testing Options Volatility Agent with {symbol}")
    
    # Generate signals
    signals = await agent.generate_options_signals(symbol, market_data)
    
    print(f"\nGenerated {len(signals)} options signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.signal_type.upper()} Signal:")
        print(f"   Strategy: {signal.strategy.value}")
        print(f"   Strength: {signal.value:.2f}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Volatility Regime: {signal.volatility_regime.value}")
        print(f"   Top 3 Reasons:")
        for reason in signal.top_3_reasons:
            print(f"     {reason['rank']}. {reason['factor']}: {reason['explanation']}")

if __name__ == "__main__":
    asyncio.run(main())