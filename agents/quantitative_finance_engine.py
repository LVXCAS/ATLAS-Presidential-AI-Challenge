#!/usr/bin/env python3
"""
Advanced Quantitative Finance Engine
Modern replacement for tf-quant-finance using active, maintained libraries

Capabilities:
- Options pricing models (Black-Scholes, Monte Carlo)
- Risk management and portfolio optimization
- Stochastic processes and volatility modeling
- Technical analysis and machine learning
- Performance analytics and backtesting
"""

import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from scipy.optimize import minimize_scalar, minimize, brentq
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import QuantLib as ql
import vectorbt as vbt
import ta
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class OptionParameters:
    """Option contract parameters"""
    underlying_price: float
    strike_price: float
    time_to_expiry: float  # in years
    risk_free_rate: float
    volatility: float
    option_type: str  # 'call' or 'put'
    dividend_yield: float = 0.0

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (95%)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    beta: float = None

class QuantitativeFinanceEngine:
    """Advanced quantitative finance engine for trading applications"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.ml_models = {}
        logger.info(f"Quantitative Finance Engine initialized (device: {self.device})")

    # ============== OPTIONS PRICING ==============

    def black_scholes_price(self, params: OptionParameters) -> Dict[str, float]:
        """
        Calculate Black-Scholes option price and Greeks

        Args:
            params: Option parameters

        Returns:
            Dictionary with price and Greeks
        """
        try:
            S = params.underlying_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            sigma = params.volatility
            q = params.dividend_yield

            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            # Standard normal CDF
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            N_neg_d1 = stats.norm.cdf(-d1)
            N_neg_d2 = stats.norm.cdf(-d2)

            # Standard normal PDF
            n_d1 = stats.norm.pdf(d1)

            if params.option_type.lower() == 'call':
                price = S * np.exp(-q*T) * N_d1 - K * np.exp(-r*T) * N_d2
                delta = np.exp(-q*T) * N_d1
                theta = (-S * n_d1 * sigma * np.exp(-q*T) / (2*np.sqrt(T))
                        - r * K * np.exp(-r*T) * N_d2
                        + q * S * np.exp(-q*T) * N_d1) / 365
            else:  # put
                price = K * np.exp(-r*T) * N_neg_d2 - S * np.exp(-q*T) * N_neg_d1
                delta = -np.exp(-q*T) * N_neg_d1
                theta = (-S * n_d1 * sigma * np.exp(-q*T) / (2*np.sqrt(T))
                        + r * K * np.exp(-r*T) * N_neg_d2
                        - q * S * np.exp(-q*T) * N_neg_d1) / 365

            # Greeks (common for both calls and puts)
            gamma = n_d1 * np.exp(-q*T) / (S * sigma * np.sqrt(T))
            vega = S * n_d1 * np.sqrt(T) * np.exp(-q*T) / 100

            if params.option_type.lower() == 'call':
                rho = K * T * np.exp(-r*T) * N_d2 / 100
            else:
                rho = -K * T * np.exp(-r*T) * N_neg_d2 / 100

            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'implied_volatility': sigma
            }

        except Exception as e:
            logger.error(f"Black-Scholes calculation error: {e}")
            return {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    def monte_carlo_option_price(self, params: OptionParameters,
                               num_simulations: int = 100000) -> Dict[str, float]:
        """
        Monte Carlo option pricing with GPU acceleration

        Args:
            params: Option parameters
            num_simulations: Number of simulation paths

        Returns:
            Dictionary with price and confidence intervals
        """
        try:
            S0 = params.underlying_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            sigma = params.volatility
            q = params.dividend_yield

            # Generate random numbers using PyTorch for GPU acceleration
            torch.manual_seed(42)  # For reproducibility
            Z = torch.randn(num_simulations, device=self.device)

            # Simulate final stock prices
            ST = S0 * torch.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

            # Calculate payoffs
            if params.option_type.lower() == 'call':
                payoffs = torch.maximum(ST - K, torch.tensor(0.0, device=self.device))
            else:  # put
                payoffs = torch.maximum(K - ST, torch.tensor(0.0, device=self.device))

            # Discount back to present value
            option_prices = payoffs * np.exp(-r*T)

            # Calculate statistics
            mean_price = torch.mean(option_prices).cpu().item()
            std_price = torch.std(option_prices).cpu().item()

            # 95% confidence interval
            confidence_interval = 1.96 * std_price / np.sqrt(num_simulations)

            return {
                'price': mean_price,
                'std_error': std_price / np.sqrt(num_simulations),
                'confidence_lower': mean_price - confidence_interval,
                'confidence_upper': mean_price + confidence_interval,
                'num_simulations': num_simulations
            }

        except Exception as e:
            logger.error(f"Monte Carlo pricing error: {e}")
            return {'price': 0.0, 'std_error': 0.0, 'confidence_lower': 0.0, 'confidence_upper': 0.0}

    def implied_volatility(self, market_price: float, params: OptionParameters) -> float:
        """
        Calculate implied volatility using Brent's method

        Args:
            market_price: Observed market price
            params: Option parameters (volatility will be solved)

        Returns:
            Implied volatility
        """
        try:
            def objective(vol):
                params.volatility = vol
                bs_result = self.black_scholes_price(params)
                return bs_result['price'] - market_price

            # Use Brent's method for root finding
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return max(0.001, min(5.0, iv))  # Clamp to reasonable range

        except Exception as e:
            logger.warning(f"Implied volatility calculation failed: {e}")
            return 0.20  # Default to 20% volatility

    # ============== VOLATILITY MODELING ==============

    def garch_volatility_forecast(self, returns: pd.Series, horizon: int = 1) -> Dict[str, float]:
        """
        GARCH(1,1) volatility forecasting

        Args:
            returns: Historical return series
            horizon: Forecast horizon (days)

        Returns:
            Volatility forecasts and model parameters
        """
        try:
            from arch import arch_model

            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            fitted_model = model.fit(disp='off')

            # Forecast volatility
            forecast = fitted_model.forecast(horizon=horizon, reindex=False)
            forecasted_vol = np.sqrt(forecast.variance.values[-1, :] / 100)

            return {
                'current_vol': np.sqrt(fitted_model.conditional_volatility.iloc[-1] / 100),
                'forecast_1d': forecasted_vol[0] if len(forecasted_vol) > 0 else 0.0,
                'forecast_5d': np.mean(forecasted_vol[:5]) if len(forecasted_vol) >= 5 else 0.0,
                'omega': fitted_model.params['omega'],
                'alpha': fitted_model.params['alpha[1]'],
                'beta': fitted_model.params['beta[1]'],
                'log_likelihood': fitted_model.loglikelihood
            }

        except Exception as e:
            logger.warning(f"GARCH volatility modeling failed: {e}")
            # Fallback to rolling volatility
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            return {
                'current_vol': rolling_vol.iloc[-1] if not rolling_vol.empty else 0.20,
                'forecast_1d': rolling_vol.iloc[-1] if not rolling_vol.empty else 0.20,
                'forecast_5d': rolling_vol.iloc[-5:].mean() if len(rolling_vol) >= 5 else 0.20
            }

    # ============== RISK MANAGEMENT ==============

    def calculate_portfolio_risk(self, returns: pd.DataFrame,
                               weights: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights (equal weight if None)

        Returns:
            RiskMetrics object with VaR, CVaR, drawdown, etc.
        """
        try:
            if weights is None:
                weights = np.ones(len(returns.columns)) / len(returns.columns)

            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)

            # Value at Risk calculations
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Sharpe Ratio (assuming 0% risk-free rate)
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

            # Sortino Ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                sortino_ratio = portfolio_returns.mean() / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = float('inf')

            # Volatility (annualized)
            volatility = portfolio_returns.std() * np.sqrt(252)

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                volatility=volatility
            )

        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

    def optimal_portfolio_weights(self, returns: pd.DataFrame,
                                target_return: Optional[float] = None,
                                risk_aversion: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate optimal portfolio weights using mean-variance optimization

        Args:
            returns: Historical returns DataFrame
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion: Risk aversion parameter

        Returns:
            Dictionary with optimal weights and portfolio statistics
        """
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized

            n_assets = len(expected_returns)

            # Objective function for mean-variance optimization
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

                if target_return is None:
                    # Maximize Sharpe ratio (minimize negative Sharpe)
                    return -portfolio_return / np.sqrt(portfolio_variance)
                else:
                    # Minimize variance subject to return target
                    return portfolio_variance + risk_aversion * (portfolio_return - target_return)**2

            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only portfolio

            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize_scalar(objective) if n_assets == 1 else \
                     minimize(objective, x0, bounds=bounds, constraints=constraints)

            if hasattr(result, 'x'):
                optimal_weights = result.x
            else:
                optimal_weights = x0  # Fallback to equal weights

            # Calculate portfolio statistics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'success': result.success if hasattr(result, 'success') else True
            }

        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            # Return equal weights as fallback
            n_assets = len(returns.columns)
            equal_weights = np.ones(n_assets) / n_assets
            return {
                'weights': equal_weights,
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'success': False
            }

    # ============== MACHINE LEARNING MODELS ==============

    def train_return_prediction_model(self, features: pd.DataFrame,
                                    returns: pd.Series,
                                    model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train machine learning model for return prediction

        Args:
            features: Feature matrix
            returns: Target returns
            model_type: Type of model ('random_forest', 'neural_network')

        Returns:
            Model performance metrics
        """
        try:
            # Prepare data
            X = features.fillna(0)
            y = returns.fillna(0)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Feature importance
                feature_importance = dict(zip(features.columns, model.feature_importances_))

            else:
                # Neural network using PyTorch
                model = self._create_neural_network(X_train.shape[1])
                y_pred = self._train_neural_network(model, X_train, y_train, X_test)
                feature_importance = {}

            # Calculate performance metrics
            mse = np.mean((y_test - y_pred)**2)
            mae = np.mean(np.abs(y_test - y_pred))
            correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0

            # Store model
            self.ml_models[model_type] = {
                'model': model,
                'scaler': self.scaler,
                'feature_names': features.columns.tolist(),
                'performance': {'mse': mse, 'mae': mae, 'correlation': correlation}
            }

            logger.info(f"Trained {model_type} model - MSE: {mse:.6f}, Correlation: {correlation:.3f}")

            return {
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'feature_importance': feature_importance
            }

        except Exception as e:
            logger.error(f"ML model training error: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'correlation': 0.0}

    def _create_neural_network(self, input_size: int) -> torch.nn.Module:
        """Create a simple neural network for return prediction"""
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(self.device)
        return model

    def _train_neural_network(self, model: torch.nn.Module,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray) -> np.ndarray:
        """Train neural network and return predictions"""
        try:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)

            # Training setup
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            for epoch in range(100):  # Quick training
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

            # Predictions
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy().flatten()

            return predictions

        except Exception as e:
            logger.error(f"Neural network training error: {e}")
            return np.zeros(len(X_test))

    def predict_returns(self, features: pd.DataFrame,
                       model_type: str = 'random_forest') -> np.ndarray:
        """
        Predict returns using trained model

        Args:
            features: Feature matrix for prediction
            model_type: Type of model to use

        Returns:
            Predicted returns
        """
        try:
            if model_type not in self.ml_models:
                logger.warning(f"Model {model_type} not trained")
                return np.zeros(len(features))

            model_info = self.ml_models[model_type]
            model = model_info['model']
            scaler = model_info['scaler']

            # Prepare features
            X = features[model_info['feature_names']].fillna(0)
            X_scaled = scaler.transform(X)

            # Make predictions
            if model_type == 'random_forest':
                predictions = model.predict(X_scaled)
            else:  # neural network
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                model.eval()
                with torch.no_grad():
                    predictions = model(X_tensor).cpu().numpy().flatten()

            return predictions

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(len(features))

    # ============== TECHNICAL ANALYSIS ==============

    def enhanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        try:
            df = data.copy()

            # Price-based indicators
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

            # MACD
            df['MACD'] = ta.trend.macd(df['Close'])
            df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
            df['MACD_histogram'] = ta.trend.macd_diff(df['Close'])

            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            # Bollinger Bands
            df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
            df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

            # Volume indicators
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()  # Simple volume SMA

            # Volatility indicators
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['Keltner_upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
            df['Keltner_lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])

            # Momentum indicators
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

            # Custom indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']

            return df.fillna(0)

        except Exception as e:
            logger.error(f"Technical indicators calculation error: {e}")
            return data

    # ============== PERFORMANCE ANALYTICS ==============

    def backtest_strategy(self, signals: pd.Series, returns: pd.Series,
                         transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Backtest a trading strategy

        Args:
            signals: Trading signals (-1, 0, 1)
            returns: Asset returns
            transaction_cost: Transaction cost per trade

        Returns:
            Performance metrics
        """
        try:
            # Align signals and returns
            aligned_data = pd.DataFrame({'signals': signals, 'returns': returns}).dropna()
            signals = aligned_data['signals']
            returns = aligned_data['returns']

            # Calculate strategy returns
            position_changes = signals.diff().abs()
            transaction_costs = position_changes * transaction_cost
            strategy_returns = signals.shift(1) * returns - transaction_costs

            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Win rate
            winning_trades = strategy_returns[strategy_returns > 0]
            win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0

            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'num_trades': int(position_changes.sum()),
                'avg_trade_return': strategy_returns[strategy_returns != 0].mean() if len(strategy_returns[strategy_returns != 0]) > 0 else 0
            }

        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return {
                'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0,
                'win_rate': 0.0, 'num_trades': 0, 'avg_trade_return': 0.0
            }

# Global instance for use across modules
quantitative_engine = QuantitativeFinanceEngine()

# Convenience functions for easy access
def calculate_option_price(underlying_price: float, strike_price: float,
                         time_to_expiry: float, volatility: float,
                         option_type: str = 'call', risk_free_rate: float = 0.05) -> Dict[str, float]:
    """Convenience function for option pricing"""
    params = OptionParameters(
        underlying_price=underlying_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type
    )
    return quantitative_engine.black_scholes_price(params)

def calculate_portfolio_risk(returns: pd.DataFrame, weights: Optional[np.ndarray] = None) -> RiskMetrics:
    """Convenience function for portfolio risk calculation"""
    return quantitative_engine.calculate_portfolio_risk(returns, weights)

def get_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for technical analysis"""
    return quantitative_engine.enhanced_technical_indicators(data)