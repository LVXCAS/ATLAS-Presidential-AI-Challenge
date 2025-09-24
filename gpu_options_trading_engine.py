"""
GPU OPTIONS TRADING ENGINE
Ultra-fast options pricing and Greeks calculations using GTX 1660 Super
Professional-grade options analysis with massive parallel processing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@dataclass
class OptionContract:
    """Option contract specification"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    underlying_price: float
    volatility: float
    risk_free_rate: float = 0.05

class GPUOptionsEngine:
    """GPU-accelerated options pricing and Greeks calculation engine"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('OptionsEngine')

        # Initialize constants for Black-Scholes calculations
        self.setup_constants()

        # Performance metrics
        self.calculations_per_second = 0
        self.total_calculations = 0

        # GPU memory management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU Options Engine initialized on {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Available GPU memory: {self.gpu_memory_total:.1f} GB")
        else:
            self.logger.warning("GPU not available, using CPU")

    def setup_constants(self):
        """Setup mathematical constants for options calculations"""
        self.sqrt_2pi = math.sqrt(2 * math.pi)
        self.inv_sqrt_2pi = 1.0 / self.sqrt_2pi

    def batch_black_scholes(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                           r: torch.Tensor, sigma: torch.Tensor,
                           option_type: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Black-Scholes pricing for massive batch processing

        Args:
            S: Underlying prices (batch_size,)
            K: Strike prices (batch_size,)
            T: Time to expiry in years (batch_size,)
            r: Risk-free rates (batch_size,)
            sigma: Volatilities (batch_size,)
            option_type: 1 for call, -1 for put (batch_size,)

        Returns:
            Option prices (batch_size,)
        """
        # Avoid division by zero
        T = torch.clamp(T, min=1e-8)
        sigma = torch.clamp(sigma, min=1e-8)

        # Black-Scholes formula components
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Cumulative normal distribution
        N_d1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))
        N_d2 = 0.5 * (1 + torch.erf(d2 / math.sqrt(2)))

        # Call option price
        call_price = S * N_d1 - K * torch.exp(-r * T) * N_d2

        # Put option price (put-call parity)
        put_price = K * torch.exp(-r * T) * (1 - N_d2) - S * (1 - N_d1)

        # Select call or put based on option_type
        prices = torch.where(option_type > 0, call_price, put_price)

        return prices

    def batch_greeks(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                    r: torch.Tensor, sigma: torch.Tensor,
                    option_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate all Greeks in parallel for massive batch processing

        Returns:
            Dictionary containing Delta, Gamma, Theta, Vega, Rho tensors
        """
        # Avoid division by zero
        T = torch.clamp(T, min=1e-8)
        sigma = torch.clamp(sigma, min=1e-8)

        # Common calculations
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Normal PDF and CDF
        pdf_d1 = torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        N_d1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))
        N_d2 = 0.5 * (1 + torch.erf(d2 / math.sqrt(2)))

        # Delta
        delta_call = N_d1
        delta_put = N_d1 - 1
        delta = torch.where(option_type > 0, delta_call, delta_put)

        # Gamma (same for calls and puts)
        gamma = pdf_d1 / (S * sigma * sqrt_T)

        # Theta
        theta_common = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * torch.exp(-r * T)
        theta_call = (theta_common * N_d2) / 365  # Convert to per day
        theta_put = (theta_common * (N_d2 - 1)) / 365
        theta = torch.where(option_type > 0, theta_call, theta_put)

        # Vega (same for calls and puts)
        vega = S * pdf_d1 * sqrt_T / 100  # Convert to per 1% volatility change

        # Rho
        rho_call = K * T * torch.exp(-r * T) * N_d2 / 100
        rho_put = -K * T * torch.exp(-r * T) * (1 - N_d2) / 100
        rho = torch.where(option_type > 0, rho_call, rho_put)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def price_options_batch(self, contracts: List[OptionContract]) -> pd.DataFrame:
        """
        Price multiple option contracts using GPU batch processing

        Args:
            contracts: List of option contracts

        Returns:
            DataFrame with pricing and Greeks
        """
        if not contracts:
            return pd.DataFrame()

        start_time = time.time()

        # Convert contracts to tensors
        batch_size = len(contracts)

        S = torch.tensor([c.underlying_price for c in contracts], device=self.device, dtype=torch.float32)
        K = torch.tensor([c.strike for c in contracts], device=self.device, dtype=torch.float32)
        T = torch.tensor([(c.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                         for c in contracts], device=self.device, dtype=torch.float32)
        r = torch.tensor([c.risk_free_rate for c in contracts], device=self.device, dtype=torch.float32)
        sigma = torch.tensor([c.volatility for c in contracts], device=self.device, dtype=torch.float32)
        option_type = torch.tensor([1 if c.option_type.lower() == 'call' else -1
                                  for c in contracts], device=self.device, dtype=torch.float32)

        # Calculate prices and Greeks
        prices = self.batch_black_scholes(S, K, T, r, sigma, option_type)
        greeks = self.batch_greeks(S, K, T, r, sigma, option_type)

        # Convert back to CPU for DataFrame creation
        results = {
            'symbol': [c.symbol for c in contracts],
            'strike': K.cpu().numpy(),
            'option_type': [c.option_type for c in contracts],
            'underlying_price': S.cpu().numpy(),
            'option_price': prices.cpu().numpy(),
            'delta': greeks['delta'].cpu().numpy(),
            'gamma': greeks['gamma'].cpu().numpy(),
            'theta': greeks['theta'].cpu().numpy(),
            'vega': greeks['vega'].cpu().numpy(),
            'rho': greeks['rho'].cpu().numpy(),
            'time_to_expiry': T.cpu().numpy(),
            'volatility': sigma.cpu().numpy()
        }

        # Performance tracking
        calc_time = time.time() - start_time
        self.total_calculations += batch_size
        self.calculations_per_second = batch_size / calc_time if calc_time > 0 else 0

        self.logger.info(f"Processed {batch_size} options in {calc_time:.4f}s "
                        f"({self.calculations_per_second:.0f} options/second)")

        return pd.DataFrame(results)

    def create_option_chain(self, symbol: str, underlying_price: float,
                           expiry_date: datetime, volatility: float = 0.25,
                           strike_range: float = 0.2, strike_step: float = 5.0) -> pd.DataFrame:
        """
        Generate complete options chain with pricing and Greeks

        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price
            expiry_date: Option expiry date
            volatility: Implied volatility
            strike_range: Strike range as percentage of underlying price
            strike_step: Strike price step

        Returns:
            Complete options chain DataFrame
        """
        # Generate strike prices
        min_strike = underlying_price * (1 - strike_range)
        max_strike = underlying_price * (1 + strike_range)
        strikes = np.arange(min_strike, max_strike + strike_step, strike_step)
        strikes = np.round(strikes / strike_step) * strike_step  # Round to nearest step

        # Create option contracts
        contracts = []
        for strike in strikes:
            # Call option
            contracts.append(OptionContract(
                symbol=f"{symbol}_C_{strike}_{expiry_date.strftime('%Y%m%d')}",
                strike=strike,
                expiry=expiry_date,
                option_type='call',
                underlying_price=underlying_price,
                volatility=volatility
            ))

            # Put option
            contracts.append(OptionContract(
                symbol=f"{symbol}_P_{strike}_{expiry_date.strftime('%Y%m%d')}",
                strike=strike,
                expiry=expiry_date,
                option_type='put',
                underlying_price=underlying_price,
                volatility=volatility
            ))

        return self.price_options_batch(contracts)

    def monte_carlo_option_pricing(self, contract: OptionContract,
                                  num_simulations: int = 100000) -> Dict[str, float]:
        """
        Monte Carlo option pricing with GPU acceleration

        Args:
            contract: Option contract to price
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with price and confidence intervals
        """
        start_time = time.time()

        # Time to expiry
        T = (contract.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
        T = max(T, 1e-8)  # Avoid division by zero

        # Monte Carlo simulation parameters
        dt = T
        sqrt_dt = math.sqrt(dt)

        # Generate random paths on GPU
        batch_size = min(num_simulations, 10000)  # Process in batches for memory efficiency
        total_payoffs = []

        for i in range(0, num_simulations, batch_size):
            current_batch_size = min(batch_size, num_simulations - i)

            # Generate random numbers
            randn = torch.randn(current_batch_size, device=self.device)

            # Simulate final prices
            drift = (contract.risk_free_rate - 0.5 * contract.volatility**2) * T
            diffusion = contract.volatility * sqrt_dt * randn
            final_prices = contract.underlying_price * torch.exp(drift + diffusion)

            # Calculate payoffs
            if contract.option_type.lower() == 'call':
                payoffs = torch.clamp(final_prices - contract.strike, min=0)
            else:
                payoffs = torch.clamp(contract.strike - final_prices, min=0)

            total_payoffs.append(payoffs)

        # Combine all payoffs
        all_payoffs = torch.cat(total_payoffs)

        # Discount to present value
        discount_factor = torch.exp(torch.tensor(-contract.risk_free_rate * T, device=self.device))
        option_price = discount_factor * torch.mean(all_payoffs)

        # Calculate confidence intervals
        payoffs_std = torch.std(all_payoffs)
        std_error = payoffs_std / math.sqrt(num_simulations)
        confidence_95 = 1.96 * std_error * math.exp(-contract.risk_free_rate * T)

        calc_time = time.time() - start_time
        simulations_per_second = num_simulations / calc_time if calc_time > 0 else 0

        self.logger.info(f"Monte Carlo: {num_simulations} simulations in {calc_time:.4f}s "
                        f"({simulations_per_second:.0f} sims/second)")

        return {
            'option_price': float(option_price.cpu()),
            'confidence_95_lower': float((option_price - confidence_95).cpu()),
            'confidence_95_upper': float((option_price + confidence_95).cpu()),
            'standard_error': float(std_error.cpu()),
            'simulations_per_second': simulations_per_second
        }

    def implied_volatility_batch(self, contracts: List[OptionContract],
                                market_prices: List[float]) -> List[float]:
        """
        Calculate implied volatility using GPU-accelerated Newton-Raphson method

        Args:
            contracts: List of option contracts
            market_prices: Corresponding market prices

        Returns:
            List of implied volatilities
        """
        if len(contracts) != len(market_prices):
            raise ValueError("Contracts and market prices must have same length")

        start_time = time.time()
        batch_size = len(contracts)

        # Convert to tensors
        S = torch.tensor([c.underlying_price for c in contracts], device=self.device, dtype=torch.float32)
        K = torch.tensor([c.strike for c in contracts], device=self.device, dtype=torch.float32)
        T = torch.tensor([(c.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                         for c in contracts], device=self.device, dtype=torch.float32)
        r = torch.tensor([c.risk_free_rate for c in contracts], device=self.device, dtype=torch.float32)
        option_type = torch.tensor([1 if c.option_type.lower() == 'call' else -1
                                  for c in contracts], device=self.device, dtype=torch.float32)
        market_prices_tensor = torch.tensor(market_prices, device=self.device, dtype=torch.float32)

        # Initial volatility guess
        sigma = torch.full((batch_size,), 0.25, device=self.device)

        # Newton-Raphson iteration
        for iteration in range(50):  # Max 50 iterations
            # Calculate theoretical prices and vega
            theoretical_prices = self.batch_black_scholes(S, K, T, r, sigma, option_type)
            greeks = self.batch_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back from percentage

            # Price difference
            price_diff = theoretical_prices - market_prices_tensor

            # Update volatility using Newton-Raphson
            # Avoid division by zero
            vega = torch.clamp(vega, min=1e-8)
            sigma_new = sigma - price_diff / vega

            # Clamp volatility to reasonable bounds
            sigma_new = torch.clamp(sigma_new, min=0.01, max=5.0)

            # Check convergence
            max_change = torch.max(torch.abs(sigma_new - sigma))
            sigma = sigma_new

            if max_change < 1e-6:
                break

        calc_time = time.time() - start_time
        self.logger.info(f"Implied volatility calculation: {batch_size} options in {calc_time:.4f}s")

        return sigma.cpu().numpy().tolist()

    def risk_analytics(self, portfolio_contracts: List[OptionContract]) -> Dict[str, float]:
        """
        Calculate portfolio-level risk analytics

        Args:
            portfolio_contracts: List of portfolio option contracts

        Returns:
            Risk metrics dictionary
        """
        if not portfolio_contracts:
            return {}

        # Price all contracts
        portfolio_df = self.price_options_batch(portfolio_contracts)

        # Portfolio Greeks
        total_delta = portfolio_df['delta'].sum()
        total_gamma = portfolio_df['gamma'].sum()
        total_theta = portfolio_df['theta'].sum()
        total_vega = portfolio_df['vega'].sum()
        total_rho = portfolio_df['rho'].sum()

        # Portfolio value
        total_value = portfolio_df['option_price'].sum()

        # Risk metrics
        risk_metrics = {
            'portfolio_value': total_value,
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'total_rho': total_rho,
            'delta_dollars': total_delta * portfolio_contracts[0].underlying_price,
            'gamma_dollars': total_gamma * portfolio_contracts[0].underlying_price**2 * 0.01,
            'theta_dollars': total_theta,
            'vega_dollars': total_vega,
            'option_count': len(portfolio_contracts)
        }

        self.logger.info(f"Portfolio risk analytics calculated for {len(portfolio_contracts)} contracts")

        return risk_metrics

    def performance_benchmark(self, num_options: int = 10000) -> Dict[str, float]:
        """
        Run performance benchmark for GPU options engine

        Args:
            num_options: Number of options to price for benchmark

        Returns:
            Performance metrics
        """
        self.logger.info(f"Running performance benchmark with {num_options} options...")

        # Generate random option contracts
        contracts = []
        for i in range(num_options):
            contracts.append(OptionContract(
                symbol=f"TEST_{i}",
                strike=100 + np.random.uniform(-20, 20),
                expiry=datetime.now() + timedelta(days=np.random.randint(1, 365)),
                option_type=np.random.choice(['call', 'put']),
                underlying_price=100 + np.random.uniform(-10, 10),
                volatility=np.random.uniform(0.15, 0.4)
            ))

        # Benchmark pricing
        start_time = time.time()
        results_df = self.price_options_batch(contracts)
        pricing_time = time.time() - start_time

        # Benchmark Monte Carlo
        start_time = time.time()
        mc_result = self.monte_carlo_option_pricing(contracts[0], num_simulations=100000)
        mc_time = time.time() - start_time

        benchmark_results = {
            'options_priced': len(contracts),
            'pricing_time_seconds': pricing_time,
            'options_per_second': len(contracts) / pricing_time if pricing_time > 0 else 0,
            'monte_carlo_time_seconds': mc_time,
            'monte_carlo_sims_per_second': mc_result['simulations_per_second'],
            'gpu_available': self.device.type == 'cuda',
            'device_name': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'
        }

        self.logger.info(f"Benchmark completed: {benchmark_results['options_per_second']:.0f} options/second")

        return benchmark_results

def demo_gpu_options_engine():
    """Demonstration of GPU Options Trading Engine capabilities"""
    print("\n" + "="*80)
    print("GPU OPTIONS TRADING ENGINE DEMONSTRATION")
    print("="*80)

    # Initialize engine
    engine = GPUOptionsEngine()

    # Demo 1: Single option pricing
    print("\n1. SINGLE OPTION PRICING:")
    spy_call = OptionContract(
        symbol="SPY_C_450_20241220",
        strike=450,
        expiry=datetime(2024, 12, 20),
        option_type='call',
        underlying_price=445.50,
        volatility=0.18
    )

    single_result = engine.price_options_batch([spy_call])
    print(f"   SPY Call Option:")
    print(f"   Price: ${single_result['option_price'].iloc[0]:.2f}")
    print(f"   Delta: {single_result['delta'].iloc[0]:.4f}")
    print(f"   Gamma: {single_result['gamma'].iloc[0]:.4f}")
    print(f"   Theta: ${single_result['theta'].iloc[0]:.2f}")
    print(f"   Vega: ${single_result['vega'].iloc[0]:.2f}")

    # Demo 2: Options chain
    print("\n2. COMPLETE OPTIONS CHAIN:")
    expiry = datetime.now() + timedelta(days=30)
    chain = engine.create_option_chain("AAPL", 175.0, expiry, volatility=0.22)
    print(f"   Generated {len(chain)} options in chain")
    print(f"   Processing rate: {engine.calculations_per_second:.0f} options/second")

    # Demo 3: Monte Carlo pricing
    print("\n3. MONTE CARLO PRICING:")
    mc_result = engine.monte_carlo_option_pricing(spy_call, num_simulations=100000)
    print(f"   Monte Carlo Price: ${mc_result['option_price']:.2f}")
    print(f"   95% Confidence: ${mc_result['confidence_95_lower']:.2f} - ${mc_result['confidence_95_upper']:.2f}")
    print(f"   Simulations/second: {mc_result['simulations_per_second']:.0f}")

    # Demo 4: Performance benchmark
    print("\n4. PERFORMANCE BENCHMARK:")
    benchmark = engine.performance_benchmark(num_options=5000)
    print(f"   Options priced: {benchmark['options_priced']}")
    print(f"   Options/second: {benchmark['options_per_second']:.0f}")
    print(f"   Device: {benchmark['device_name']}")
    print(f"   GPU acceleration: {'YES' if benchmark['gpu_available'] else 'NO'}")

    print("\n" + "="*80)
    print("GPU OPTIONS ENGINE READY FOR MARKET DOMINATION!")
    print("="*80)

if __name__ == "__main__":
    demo_gpu_options_engine()