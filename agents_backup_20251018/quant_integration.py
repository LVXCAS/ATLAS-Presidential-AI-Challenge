#!/usr/bin/env python3
"""
Quantitative Finance Integration Module
Integrates the quantitative finance engine with trading bots

Provides:
- Enhanced options pricing and Greeks calculation
- Portfolio risk management and optimization
- Advanced technical analysis
- Machine learning-based predictions
- Real-time quantitative signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf

from agents.quantitative_finance_engine import (
    quantitative_engine,
    OptionParameters,
    RiskMetrics,
    calculate_option_price,
    calculate_portfolio_risk,
    get_technical_indicators
)
from config.logging_config import get_logger

logger = get_logger(__name__)

class QuantitativeTradeAnalyzer:
    """Enhanced trade analysis using quantitative methods"""

    def __init__(self):
        self.engine = quantitative_engine
        self.cache = {}
        logger.info("Quantitative Trade Analyzer initialized")

    def analyze_options_opportunity(self, symbol: str, strike: float,
                                  expiry_date: str, option_type: str = 'call') -> Dict[str, float]:
        """
        Comprehensive options analysis using advanced quantitative methods

        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiry_date: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'

        Returns:
            Dictionary with pricing, Greeks, and risk metrics
        """
        try:
            # Get current market data
            ticker = yf.Ticker(symbol)

            # Get current price
            current_data = ticker.history(period='1d', interval='1m')
            if current_data.empty:
                current_data = ticker.history(period='5d')

            current_price = current_data['Close'].iloc[-1]

            # Calculate time to expiry
            expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            time_to_expiry = (expiry_dt - datetime.now()).days / 365.25

            if time_to_expiry <= 0:
                logger.warning(f"Option {symbol} {strike} {expiry_date} already expired")
                return {'expired': True}

            # Get historical data for volatility calculation
            hist_data = ticker.history(period='60d')
            returns = hist_data['Close'].pct_change().dropna()

            # Calculate realized volatility using GARCH if possible
            vol_analysis = self.engine.garch_volatility_forecast(returns)
            current_vol = vol_analysis.get('current_vol', returns.std() * np.sqrt(252))

            # Option parameters
            params = OptionParameters(
                underlying_price=current_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=0.05,  # 5% risk-free rate
                volatility=current_vol,
                option_type=option_type
            )

            # Black-Scholes pricing and Greeks
            bs_results = self.engine.black_scholes_price(params)

            # Monte Carlo pricing for comparison
            mc_results = self.engine.monte_carlo_option_price(params, num_simulations=50000)

            # Enhanced analysis
            moneyness = current_price / strike
            time_decay_risk = self._assess_time_decay_risk(time_to_expiry, current_vol)
            volatility_risk = self._assess_volatility_risk(current_vol, returns)

            # Technical analysis of underlying
            technical_data = self._get_technical_signals(symbol)

            # Comprehensive risk assessment
            risk_score = self._calculate_option_risk_score(
                moneyness, time_to_expiry, current_vol, technical_data
            )

            result = {
                # Basic info
                'symbol': symbol,
                'underlying_price': current_price,
                'strike_price': strike,
                'option_type': option_type,
                'time_to_expiry': time_to_expiry,
                'days_to_expiry': int(time_to_expiry * 365),

                # Pricing
                'bs_price': bs_results['price'],
                'mc_price': mc_results['price'],
                'price_confidence_interval': (mc_results['confidence_lower'], mc_results['confidence_upper']),

                # Greeks
                'delta': bs_results['delta'],
                'gamma': bs_results['gamma'],
                'theta': bs_results['theta'],
                'vega': bs_results['vega'],
                'rho': bs_results['rho'],

                # Volatility analysis
                'current_volatility': current_vol,
                'volatility_forecast_1d': vol_analysis.get('forecast_1d', current_vol),
                'volatility_forecast_5d': vol_analysis.get('forecast_5d', current_vol),

                # Risk metrics
                'moneyness': moneyness,
                'time_decay_risk': time_decay_risk,
                'volatility_risk': volatility_risk,
                'overall_risk_score': risk_score,

                # Technical signals
                'technical_signal': technical_data.get('signal', 'NEUTRAL'),
                'technical_strength': technical_data.get('strength', 0.5),

                # Trading recommendations
                'entry_recommendation': self._get_entry_recommendation(
                    bs_results, moneyness, time_to_expiry, technical_data
                ),
                'profit_target': self._calculate_profit_target(bs_results, current_vol),
                'stop_loss': self._calculate_stop_loss(bs_results, risk_score)
            }

            logger.info(f"Options analysis completed for {symbol} {strike} {option_type}")
            return result

        except Exception as e:
            logger.error(f"Options analysis error for {symbol}: {e}")
            return {'error': str(e)}

    def _assess_time_decay_risk(self, time_to_expiry: float, volatility: float) -> str:
        """Assess time decay risk level"""
        if time_to_expiry < 0.025:  # Less than ~9 days
            return 'HIGH'
        elif time_to_expiry < 0.08:  # Less than ~30 days
            return 'MEDIUM'
        else:
            return 'LOW'

    def _assess_volatility_risk(self, current_vol: float, returns: pd.Series) -> str:
        """Assess volatility risk level"""
        vol_percentile = np.percentile(returns.rolling(20).std() * np.sqrt(252), 80)

        if current_vol > vol_percentile * 1.5:
            return 'HIGH'
        elif current_vol > vol_percentile:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_technical_signals(self, symbol: str) -> Dict[str, Union[str, float]]:
        """Get technical analysis signals for the underlying"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='60d')

            if data.empty:
                return {'signal': 'NEUTRAL', 'strength': 0.5}

            # Calculate technical indicators
            tech_data = get_technical_indicators(data)

            # Simple signal generation based on multiple indicators
            signals = []

            # RSI signal
            rsi = tech_data['RSI'].iloc[-1]
            if rsi > 70:
                signals.append(-1)  # Overbought
            elif rsi < 30:
                signals.append(1)   # Oversold
            else:
                signals.append(0)

            # MACD signal
            macd_diff = tech_data['MACD_histogram'].iloc[-1]
            if macd_diff > 0:
                signals.append(1)   # Bullish
            else:
                signals.append(-1)  # Bearish

            # Moving average signal
            sma_10 = tech_data['SMA_10'].iloc[-1]
            sma_20 = tech_data['SMA_20'].iloc[-1]
            current_price = data['Close'].iloc[-1]

            if current_price > sma_10 > sma_20:
                signals.append(1)   # Bullish
            elif current_price < sma_10 < sma_20:
                signals.append(-1)  # Bearish
            else:
                signals.append(0)   # Neutral

            # Aggregate signals
            avg_signal = np.mean(signals)
            signal_strength = abs(avg_signal)

            if avg_signal > 0.3:
                signal = 'BULLISH'
            elif avg_signal < -0.3:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            return {
                'signal': signal,
                'strength': signal_strength,
                'rsi': rsi,
                'macd_histogram': macd_diff
            }

        except Exception as e:
            logger.warning(f"Technical analysis error for {symbol}: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0.5}

    def _calculate_option_risk_score(self, moneyness: float, time_to_expiry: float,
                                   volatility: float, technical_data: Dict) -> float:
        """Calculate overall risk score for option (0-1, higher = riskier)"""
        risk_factors = []

        # Moneyness risk (how far from ATM)
        moneyness_risk = abs(1 - moneyness) * 2  # Distance from ATM
        risk_factors.append(min(moneyness_risk, 1.0))

        # Time decay risk
        if time_to_expiry < 0.025:  # < 9 days
            time_risk = 0.8
        elif time_to_expiry < 0.08:  # < 30 days
            time_risk = 0.5
        else:
            time_risk = 0.2
        risk_factors.append(time_risk)

        # Volatility risk (high vol = high risk)
        vol_risk = min(volatility / 0.5, 1.0)  # Normalize around 50% vol
        risk_factors.append(vol_risk)

        # Technical risk (opposing signals increase risk)
        tech_strength = technical_data.get('strength', 0.5)
        if technical_data.get('signal') == 'NEUTRAL':
            tech_risk = 0.6
        else:
            tech_risk = 1 - tech_strength  # Strong signals reduce risk
        risk_factors.append(tech_risk)

        # Overall risk score (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]  # Moneyness and time most important
        risk_score = np.average(risk_factors, weights=weights)

        return min(max(risk_score, 0.0), 1.0)

    def _get_entry_recommendation(self, bs_results: Dict, moneyness: float,
                                time_to_expiry: float, technical_data: Dict) -> str:
        """Generate entry recommendation based on analysis"""
        # Factors favoring entry
        good_factors = []

        # Time to expiry
        if time_to_expiry > 0.08:  # > 30 days
            good_factors.append("Sufficient time to expiry")

        # Technical alignment
        if technical_data.get('strength', 0) > 0.6:
            good_factors.append("Strong technical signals")

        # Greeks analysis
        if abs(bs_results.get('delta', 0)) > 0.3:  # Meaningful delta
            good_factors.append("Good delta exposure")

        # Moneyness
        if 0.8 <= moneyness <= 1.2:  # Near the money
            good_factors.append("Near-the-money positioning")

        # Overall recommendation
        if len(good_factors) >= 3:
            return "STRONG_BUY"
        elif len(good_factors) >= 2:
            return "BUY"
        elif len(good_factors) >= 1:
            return "HOLD"
        else:
            return "AVOID"

    def _calculate_profit_target(self, bs_results: Dict, volatility: float) -> float:
        """Calculate profit target based on option characteristics"""
        base_price = bs_results.get('price', 0)

        # Target based on volatility and time decay
        if volatility > 0.4:  # High vol
            target_multiplier = 2.0
        elif volatility > 0.25:  # Medium vol
            target_multiplier = 1.5
        else:  # Low vol
            target_multiplier = 1.3

        return base_price * target_multiplier

    def _calculate_stop_loss(self, bs_results: Dict, risk_score: float) -> float:
        """Calculate stop loss based on risk assessment"""
        base_price = bs_results.get('price', 0)

        # Tighter stops for higher risk
        if risk_score > 0.7:
            stop_multiplier = 0.7  # 30% stop loss
        elif risk_score > 0.5:
            stop_multiplier = 0.6  # 40% stop loss
        else:
            stop_multiplier = 0.5  # 50% stop loss

        return base_price * stop_multiplier

    def analyze_portfolio_risk(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Analyze portfolio-level risk using quantitative methods

        Args:
            positions: List of position dictionaries

        Returns:
            Portfolio risk metrics
        """
        try:
            if not positions:
                return {'error': 'No positions to analyze'}

            # Extract symbols and weights
            symbols = [pos.get('symbol', '') for pos in positions]
            weights = np.array([pos.get('weight', 1.0) for pos in positions])
            weights = weights / weights.sum()  # Normalize

            # Get historical data for all symbols
            returns_data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='60d')
                    if not hist.empty:
                        returns_data[symbol] = hist['Close'].pct_change().dropna()
                except:
                    logger.warning(f"Could not get data for {symbol}")

            if not returns_data:
                return {'error': 'No valid price data available'}

            # Combine returns
            returns_df = pd.DataFrame(returns_data).dropna()

            if returns_df.empty:
                return {'error': 'No overlapping data for portfolio analysis'}

            # Calculate portfolio risk metrics
            risk_metrics = calculate_portfolio_risk(returns_df, weights)

            # Additional portfolio analysis
            correlation_matrix = returns_df.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

            # Portfolio optimization
            optimal_weights = self.engine.optimal_portfolio_weights(returns_df)

            return {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'cvar_95': risk_metrics.cvar_95,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'volatility': risk_metrics.volatility,
                'avg_correlation': avg_correlation,
                'diversification_ratio': 1 / np.sqrt(len(symbols)),  # Simple measure
                'optimal_weights': optimal_weights['weights'].tolist(),
                'current_vs_optimal_sharpe': optimal_weights['sharpe_ratio'] - risk_metrics.sharpe_ratio
            }

        except Exception as e:
            logger.error(f"Portfolio risk analysis error: {e}")
            return {'error': str(e)}

    def generate_ml_predictions(self, symbol: str, features: List[str] = None) -> Dict[str, float]:
        """
        Generate ML-based return predictions

        Args:
            symbol: Symbol to predict
            features: Custom features (if None, uses default technical indicators)

        Returns:
            Prediction results
        """
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='252d')  # 1 year

            if data.empty:
                return {'error': 'No data available'}

            # Calculate technical indicators as features
            tech_data = get_technical_indicators(data)

            # Prepare features and targets
            if features is None:
                feature_cols = [
                    'RSI', 'MACD', 'MACD_signal', 'BB_width',
                    'ATR', 'Williams_R', 'Volume_Change', 'High_Low_Ratio'
                ]
            else:
                feature_cols = features

            # Ensure we have the required columns
            available_cols = [col for col in feature_cols if col in tech_data.columns]
            if not available_cols:
                return {'error': 'No valid features available'}

            X = tech_data[available_cols].iloc[:-1]  # Features (excluding last row)
            y = tech_data['Price_Change'].shift(-1).iloc[:-1]  # Next day return

            # Remove NaN values
            data_clean = pd.concat([X, y], axis=1).dropna()
            if len(data_clean) < 50:  # Need sufficient data
                return {'error': 'Insufficient clean data for ML'}

            X_clean = data_clean[available_cols]
            y_clean = data_clean['Price_Change']

            # Train model
            performance = self.engine.train_return_prediction_model(
                X_clean, y_clean, model_type='random_forest'
            )

            # Make prediction for next period
            latest_features = tech_data[available_cols].iloc[-1:].fillna(0)
            prediction = self.engine.predict_returns(latest_features, model_type='random_forest')

            return {
                'predicted_return': prediction[0] if len(prediction) > 0 else 0.0,
                'model_performance': performance,
                'features_used': available_cols,
                'prediction_confidence': 1 - performance.get('mse', 1.0)  # Simple confidence metric
            }

        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return {'error': str(e)}

# Global instance for use across modules
quant_analyzer = QuantitativeTradeAnalyzer()

# Convenience functions for easy integration
def analyze_option(symbol: str, strike: float, expiry_date: str, option_type: str = 'call') -> Dict:
    """Convenience function for options analysis"""
    return quant_analyzer.analyze_options_opportunity(symbol, strike, expiry_date, option_type)

def analyze_portfolio(positions: List[Dict]) -> Dict:
    """Convenience function for portfolio analysis"""
    return quant_analyzer.analyze_portfolio_risk(positions)

def predict_returns(symbol: str) -> Dict:
    """Convenience function for ML predictions"""
    return quant_analyzer.generate_ml_predictions(symbol)