"""
GPU RISK MANAGEMENT BEAST
10,000+ Monte Carlo simulations powered by GTX 1660 Super
Institutional-grade risk analysis in real-time
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure GPU for Monte Carlo simulations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_risk_management.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    stress_test_results: Dict[str, float]
    monte_carlo_scenarios: int
    confidence_interval: Tuple[float, float]

@dataclass
class PortfolioPosition:
    """Portfolio position representation"""
    symbol: str
    quantity: float
    current_price: float
    market_value: float
    weight: float
    beta: float
    volatility: float
    correlation_matrix: Dict[str, float]

class MonteCarloEngine:
    """GPU-accelerated Monte Carlo simulation engine"""

    def __init__(self, device, num_simulations=10000):
        self.device = device
        self.num_simulations = num_simulations
        self.logger = logging.getLogger('MonteCarloEngine')

        if self.device.type == 'cuda':
            # Optimize for GTX 1660 Super
            self.batch_size = 2048  # Process 2048 scenarios simultaneously
            torch.cuda.empty_cache()
            self.logger.info(f">> Monte Carlo Engine: {torch.cuda.get_device_name(0)}")
        else:
            self.batch_size = 512
            self.logger.info(">> CPU Monte Carlo Engine")

        self.logger.info(f">> Batch size: {self.batch_size} scenarios")
        self.logger.info(f">> Total capacity: {num_simulations:,} simulations")

    def simulate_price_paths(self, initial_prices: torch.Tensor, returns: torch.Tensor,
                           correlations: torch.Tensor, time_horizon: int = 252) -> torch.Tensor:
        """Simulate price paths using geometric Brownian motion"""

        num_assets = len(initial_prices)

        # Generate correlated random shocks on GPU
        random_shocks = torch.randn(self.num_simulations, time_horizon, num_assets, device=self.device)

        # Apply correlation structure
        L = torch.linalg.cholesky(correlations)
        correlated_shocks = torch.matmul(random_shocks, L.T)

        # Calculate means and volatilities
        mu = returns.mean(dim=0)
        sigma = returns.std(dim=0)

        # Geometric Brownian Motion
        dt = 1/252  # Daily time step
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * torch.sqrt(torch.tensor(dt, device=self.device))

        # Generate price paths
        log_returns = drift.unsqueeze(0).unsqueeze(0) + diffusion.unsqueeze(0).unsqueeze(0) * correlated_shocks
        cumulative_log_returns = torch.cumsum(log_returns, dim=1)

        # Convert to price paths
        price_paths = initial_prices.unsqueeze(0).unsqueeze(0) * torch.exp(cumulative_log_returns)

        return price_paths

    def calculate_portfolio_scenarios(self, price_paths: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Calculate portfolio value scenarios"""

        # Portfolio values for each scenario and time step
        portfolio_values = torch.sum(price_paths * weights.unsqueeze(0).unsqueeze(0), dim=2)

        return portfolio_values

    def run_stress_tests(self, portfolio_values: torch.Tensor, initial_value: float) -> Dict[str, float]:
        """Run comprehensive stress tests"""

        # Convert to returns
        portfolio_returns = (portfolio_values[:, -1] - initial_value) / initial_value

        stress_results = {}

        # Market crash scenarios
        crash_scenarios = portfolio_returns < -0.20  # 20% loss
        stress_results['market_crash_probability'] = crash_scenarios.float().mean().item()

        # Severe loss scenarios
        severe_loss = portfolio_returns < -0.10  # 10% loss
        stress_results['severe_loss_probability'] = severe_loss.float().mean().item()

        # Black swan events (3+ sigma moves)
        returns_mean = portfolio_returns.mean()
        returns_std = portfolio_returns.std()
        black_swan = torch.abs(portfolio_returns - returns_mean) > 3 * returns_std
        stress_results['black_swan_probability'] = black_swan.float().mean().item()

        # Maximum consecutive losses
        daily_returns = torch.diff(portfolio_values, dim=1) / portfolio_values[:, :-1]
        consecutive_losses = self._calculate_max_consecutive_losses(daily_returns)
        stress_results['max_consecutive_loss_days'] = consecutive_losses.float().mean().item()

        # Liquidity stress (extreme volume scenarios)
        extreme_moves = torch.abs(portfolio_returns) > 2 * returns_std
        stress_results['extreme_move_probability'] = extreme_moves.float().mean().item()

        return stress_results

    def _calculate_max_consecutive_losses(self, daily_returns: torch.Tensor) -> torch.Tensor:
        """Calculate maximum consecutive loss days for each scenario"""

        # Convert to loss indicators (1 if loss, 0 if gain)
        loss_indicators = (daily_returns < 0).float()

        # Calculate consecutive losses using convolution-like approach
        max_consecutive = torch.zeros(loss_indicators.shape[0], device=self.device)

        for i in range(loss_indicators.shape[0]):
            # Find consecutive loss streaks
            losses = loss_indicators[i]
            consecutive_count = 0
            max_count = 0

            for loss in losses:
                if loss == 1:
                    consecutive_count += 1
                    max_count = max(max_count, consecutive_count)
                else:
                    consecutive_count = 0

            max_consecutive[i] = max_count

        return max_consecutive

class GPURiskManagementBeast:
    """The ultimate GPU-powered risk management system"""

    def __init__(self):
        self.device = device
        self.logger = logging.getLogger('RiskManagementBeast')

        # Initialize Monte Carlo engine
        self.mc_engine = MonteCarloEngine(self.device, num_simulations=10000)

        # Risk parameters
        self.confidence_levels = [0.95, 0.99]
        self.time_horizons = [1, 5, 10, 22, 63, 252]  # 1D, 1W, 2W, 1M, 3M, 1Y

        # Portfolio tracking
        self.portfolio_data = {}
        self.correlation_matrix = None
        self.risk_history = []

        if self.device.type == 'cuda':
            self.logger.info(f">> Risk Management Beast: {torch.cuda.get_device_name(0)}")
            self.logger.info(f">> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.info(">> CPU Risk Management")

        self.logger.info(f">> Monte Carlo capacity: {self.mc_engine.num_simulations:,} scenarios")
        self.logger.info(f">> Institutional-grade risk analysis ready")

    def fetch_portfolio_data(self, symbols: List[str], period: str = '2y') -> Dict[str, pd.DataFrame]:
        """Fetch historical data for portfolio symbols"""

        def fetch_symbol(symbol):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    data['Returns'] = data['Close'].pct_change().dropna()
                    return symbol, data
                return symbol, None
            except Exception as e:
                self.logger.debug(f"Error fetching {symbol}: {e}")
                return symbol, None

        portfolio_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_symbol, symbols)

            for symbol, data in results:
                if data is not None:
                    portfolio_data[symbol] = data

        self.logger.info(f">> Fetched data for {len(portfolio_data)} symbols")
        return portfolio_data

    def calculate_correlation_matrix(self, portfolio_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Calculate correlation matrix on GPU"""

        # Align data and extract returns
        symbols = list(portfolio_data.keys())
        returns_data = []

        # Find common date range
        min_length = min(len(data) for data in portfolio_data.values())

        for symbol in symbols:
            returns = portfolio_data[symbol]['Returns'].iloc[-min_length:].values
            returns_data.append(returns)

        # Convert to tensor
        returns_tensor = torch.tensor(np.array(returns_data).T, dtype=torch.float32, device=self.device)

        # Calculate correlation matrix on GPU
        correlation_matrix = torch.corrcoef(returns_tensor.T)

        # Ensure positive definite (add small diagonal if needed)
        eigenvals = torch.linalg.eigvals(correlation_matrix)
        if torch.min(eigenvals.real) < 1e-8:
            correlation_matrix += torch.eye(len(symbols), device=self.device) * 1e-6

        self.correlation_matrix = correlation_matrix
        self.logger.info(f">> Calculated {len(symbols)}x{len(symbols)} correlation matrix on GPU")

        return correlation_matrix

    def calculate_var_cvar(self, portfolio_returns: torch.Tensor, confidence_levels: List[float]) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR on GPU"""

        var_cvar_results = {}

        for confidence in confidence_levels:
            # Calculate VaR (percentile)
            alpha = 1 - confidence
            var_percentile = torch.quantile(portfolio_returns, alpha)
            var_cvar_results[f'var_{int(confidence*100)}'] = var_percentile.item()

            # Calculate CVaR (expected value beyond VaR)
            tail_losses = portfolio_returns[portfolio_returns <= var_percentile]
            if len(tail_losses) > 0:
                cvar = tail_losses.mean()
                var_cvar_results[f'cvar_{int(confidence*100)}'] = cvar.item()
            else:
                var_cvar_results[f'cvar_{int(confidence*100)}'] = var_percentile.item()

        return var_cvar_results

    def run_comprehensive_risk_analysis(self, portfolio: Dict[str, float],
                                      benchmark_symbol: str = 'SPY') -> Dict[str, Any]:
        """Run comprehensive GPU-accelerated risk analysis"""

        self.logger.info(f">> Starting comprehensive risk analysis...")
        start_time = datetime.now()

        # Fetch portfolio data
        symbols = list(portfolio.keys())
        portfolio_data = self.fetch_portfolio_data(symbols + [benchmark_symbol])

        if len(portfolio_data) < len(symbols):
            self.logger.warning(f">> Missing data for some symbols")
            return {}

        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(portfolio_data)

        # Prepare data for Monte Carlo
        current_prices = []
        returns_data = []
        weights = []

        total_value = sum(portfolio.values())

        for symbol in symbols:
            if symbol in portfolio_data:
                current_price = portfolio_data[symbol]['Close'].iloc[-1]
                returns = portfolio_data[symbol]['Returns'].dropna()

                current_prices.append(current_price)
                returns_data.append(returns.values)
                weights.append(portfolio[symbol] / total_value)

        # Convert to tensors
        current_prices_tensor = torch.tensor(current_prices, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Align returns data
        min_length = min(len(returns) for returns in returns_data)
        aligned_returns = np.array([returns[-min_length:] for returns in returns_data]).T
        returns_tensor = torch.tensor(aligned_returns, dtype=torch.float32, device=self.device)

        # Run Monte Carlo simulations
        self.logger.info(f">> Running {self.mc_engine.num_simulations:,} Monte Carlo scenarios on GPU...")
        mc_start = datetime.now()

        # Simulate price paths
        price_paths = self.mc_engine.simulate_price_paths(
            current_prices_tensor, returns_tensor, correlation_matrix, time_horizon=252
        )

        # Calculate portfolio scenarios
        portfolio_scenarios = self.mc_engine.calculate_portfolio_scenarios(price_paths, weights_tensor)
        initial_portfolio_value = torch.sum(current_prices_tensor * weights_tensor)

        # Convert to returns
        final_values = portfolio_scenarios[:, -1]
        portfolio_returns = (final_values - initial_portfolio_value) / initial_portfolio_value

        mc_time = (datetime.now() - mc_start).total_seconds()

        # Calculate risk metrics
        self.logger.info(f">> Calculating comprehensive risk metrics...")

        # VaR and CVaR
        var_cvar_results = self.calculate_var_cvar(portfolio_returns, self.confidence_levels)

        # Drawdown analysis
        cumulative_values = torch.cumprod(1 + torch.diff(portfolio_scenarios, dim=1) / portfolio_scenarios[:, :-1], dim=1)
        drawdowns = (torch.cummax(cumulative_values, dim=1)[0] - cumulative_values) / torch.cummax(cumulative_values, dim=1)[0]
        max_drawdown = torch.max(drawdowns, dim=1)[0].mean()

        # Volatility and Sharpe ratio
        portfolio_returns_std = portfolio_returns.std()
        portfolio_returns_mean = portfolio_returns.mean()
        sharpe_ratio = portfolio_returns_mean / portfolio_returns_std if portfolio_returns_std > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else portfolio_returns_std
        sortino_ratio = portfolio_returns_mean / downside_deviation if downside_deviation > 0 else 0

        # Beta calculation (if benchmark available)
        beta = 0.0
        if benchmark_symbol in portfolio_data:
            benchmark_returns = torch.tensor(
                portfolio_data[benchmark_symbol]['Returns'].iloc[-min_length:].values,
                dtype=torch.float32, device=self.device
            )
            portfolio_historical_returns = torch.sum(returns_tensor * weights_tensor.unsqueeze(0), dim=1)

            # Calculate beta on GPU
            covariance = torch.mean((portfolio_historical_returns - portfolio_historical_returns.mean()) *
                                  (benchmark_returns - benchmark_returns.mean()))
            benchmark_variance = torch.var(benchmark_returns)
            beta = (covariance / benchmark_variance).item() if benchmark_variance > 0 else 0

        # Stress tests
        self.logger.info(f">> Running stress tests...")
        stress_results = self.mc_engine.run_stress_tests(portfolio_scenarios, initial_portfolio_value.item())

        # Concentration and correlation risk
        concentration_risk = torch.max(weights_tensor).item()  # Largest position weight
        avg_correlation = torch.mean(correlation_matrix[torch.triu(torch.ones_like(correlation_matrix), diagonal=1) == 1])

        # Confidence intervals
        ci_lower = torch.quantile(portfolio_returns, 0.025).item()
        ci_upper = torch.quantile(portfolio_returns, 0.975).item()

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Compile results
        risk_analysis = {
            'analysis_timestamp': start_time,
            'completion_timestamp': end_time,
            'processing_time_seconds': total_time,
            'monte_carlo_time_seconds': mc_time,
            'portfolio_summary': {
                'total_value': total_value,
                'num_positions': len(symbols),
                'largest_position_weight': concentration_risk,
                'average_correlation': avg_correlation.item()
            },
            'risk_metrics': {
                **var_cvar_results,
                'max_drawdown': max_drawdown.item(),
                'volatility_annualized': portfolio_returns_std.item() * np.sqrt(252),
                'sharpe_ratio': sharpe_ratio.item(),
                'sortino_ratio': sortino_ratio.item(),
                'calmar_ratio': (portfolio_returns_mean.item() * 252) / max_drawdown.item() if max_drawdown.item() > 0 else 0,
                'beta': beta,
                'concentration_risk': concentration_risk,
                'correlation_risk': avg_correlation.item()
            },
            'stress_test_results': stress_results,
            'monte_carlo_metrics': {
                'num_simulations': self.mc_engine.num_simulations,
                'scenarios_per_second': self.mc_engine.num_simulations / mc_time if mc_time > 0 else 0,
                'gpu_accelerated': self.device.type == 'cuda',
                'confidence_interval_95': (ci_lower, ci_upper),
                'worst_case_scenario': torch.min(portfolio_returns).item(),
                'best_case_scenario': torch.max(portfolio_returns).item()
            },
            'performance_metrics': {
                'total_processing_rate': self.mc_engine.num_simulations / total_time if total_time > 0 else 0,
                'gpu_memory_used_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            },
            'risk_warnings': self._generate_risk_warnings(var_cvar_results, stress_results, concentration_risk)
        }

        # Save analysis
        self.risk_history.append(risk_analysis)

        # Log results
        self.logger.info(f">> COMPREHENSIVE RISK ANALYSIS COMPLETE!")
        self.logger.info(f">> Total time: {total_time:.1f}s")
        self.logger.info(f">> Monte Carlo: {mc_time:.1f}s ({self.mc_engine.num_simulations/mc_time:.0f} scenarios/sec)")
        self.logger.info(f">> VaR 95%: {var_cvar_results['var_95']:.2%}")
        self.logger.info(f">> Max Drawdown: {max_drawdown.item():.2%}")
        self.logger.info(f">> Sharpe Ratio: {sharpe_ratio.item():.3f}")

        if torch.cuda.is_available():
            self.logger.info(f">> GPU memory used: {risk_analysis['performance_metrics']['gpu_memory_used_gb']:.2f} GB")

        return risk_analysis

    def _generate_risk_warnings(self, var_results: Dict, stress_results: Dict, concentration: float) -> List[str]:
        """Generate risk warnings based on analysis results"""
        warnings = []

        if var_results.get('var_95', 0) < -0.10:
            warnings.append("HIGH RISK: 95% VaR exceeds 10% - consider position sizing")

        if stress_results.get('market_crash_probability', 0) > 0.05:
            warnings.append("WARNING: High probability of severe losses in stress scenarios")

        if concentration > 0.3:
            warnings.append("CONCENTRATION RISK: Single position exceeds 30% of portfolio")

        if stress_results.get('black_swan_probability', 0) > 0.02:
            warnings.append("TAIL RISK: Elevated probability of extreme events")

        if not warnings:
            warnings.append("Portfolio risk profile appears within acceptable parameters")

        return warnings

def run_risk_demo():
    """Run risk management demonstration"""

    # Initialize risk management system
    risk_beast = GPURiskManagementBeast()

    # Demo portfolio
    demo_portfolio = {
        'AAPL': 25000,  # $25k in Apple
        'MSFT': 20000,  # $20k in Microsoft
        'GOOGL': 15000, # $15k in Google
        'TSLA': 10000,  # $10k in Tesla
        'SPY': 30000    # $30k in S&P 500 ETF
    }

    print(f"\n>> RISK MANAGEMENT BEAST DEMONSTRATION")
    print(f">> Portfolio value: ${sum(demo_portfolio.values()):,}")
    print(f">> Positions: {len(demo_portfolio)}")

    # Run comprehensive analysis
    results = risk_beast.run_comprehensive_risk_analysis(demo_portfolio)

    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'risk_analysis_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n>> RISK ANALYSIS COMPLETE!")
        print(f">> {results['monte_carlo_metrics']['num_simulations']:,} scenarios in {results['monte_carlo_time_seconds']:.1f}s")
        print(f">> Performance: {results['monte_carlo_metrics']['scenarios_per_second']:,.0f} scenarios/second")
        print(f"\n>> KEY RISK METRICS:")
        print(f"   VaR 95%: {results['risk_metrics']['var_95']:.2%}")
        print(f"   VaR 99%: {results['risk_metrics']['var_99']:.2%}")
        print(f"   Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.3f}")
        print(f"   Beta: {results['risk_metrics']['beta']:.3f}")

        print(f"\n>> STRESS TEST RESULTS:")
        for test, probability in results['stress_test_results'].items():
            print(f"   {test}: {probability:.1%}")

        if results['risk_warnings']:
            print(f"\n>> RISK WARNINGS:")
            for warning in results['risk_warnings']:
                print(f"   ! {warning}")

        print(f"\n>> GPU memory used: {results['performance_metrics']['gpu_memory_used_gb']:.2f} GB")
        print(f">> Risk Beast ready for institutional-grade analysis! 🛡️")

    else:
        print(">> Risk analysis failed - check configuration")

if __name__ == "__main__":
    run_risk_demo()