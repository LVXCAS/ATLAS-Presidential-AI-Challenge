"""
Monte Carlo Agent

Runs real-time Monte Carlo simulations BEFORE each trade to assess risk.

This is next-level risk management: Instead of hoping a trade works out,
we simulate 1000+ possible outcomes and only proceed if the probability
of success meets our threshold.

Specialization: Pre-trade risk simulation, probabilistic decision making.
"""

from typing import Dict, Tuple, List
from .base_agent import BaseAgent
import numpy as np
try:
    from arch import arch_model
except ImportError:  # optional dependency
    arch_model = None

try:
    from scipy import stats
except ImportError:  # optional dependency
    stats = None


class MonteCarloAgent(BaseAgent):
    """
    Real-time Monte Carlo simulation agent.

    Runs before each trade to answer:
    - "What's the probability this trade succeeds?"
    - "What's the worst-case drawdown if I take this trade?"
    - "What's the expected value of this trade?"

    Uses historical trade statistics to simulate possible outcomes.

    VETO Capability: Can block trades with unfavorable probability.
    """

    def __init__(self, initial_weight: float = 2.0, is_veto: bool = False):
        super().__init__(name="MonteCarloAgent", initial_weight=initial_weight)

        self.advanced_stats_available = (arch_model is not None) and (stats is not None)

        # Monte Carlo parameters
        self.num_simulations = 1000  # Run 1000 simulations per trade
        self.min_win_probability = 0.55  # Minimum 55% win probability
        self.max_acceptable_dd_risk = 0.02  # Max 2% DD risk from single trade
        self.is_veto = is_veto

        # Historical statistics (updated as we trade)
        self.historical_win_rate = 0.50  # Start with 50% assumption
        self.historical_avg_win = 1500  # Default: $1500 avg win
        self.historical_avg_loss = 800  # Default: $800 avg loss
        self.historical_win_variance = 0.3  # 30% variance in wins
        self.historical_loss_variance = 0.3  # 30% variance in losses
        
        # GARCH volatility forecasting
        self.price_returns = {}  # Store returns by pair for GARCH
        self.volatility_forecasts = {}  # Cached GARCH forecasts

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Run Monte Carlo simulation on proposed trade.

        Returns:
            (vote, confidence, reasoning)
        """
        pair = market_data.get("pair", "UNKNOWN")
        current_balance = market_data.get("current_balance", 200000)
        proposed_position_size = market_data.get("position_size", 3.0)
        stop_loss_pips = market_data.get("stop_loss_pips", 15)
        take_profit_pips = market_data.get("take_profit_pips", 30)

        # Run Monte Carlo simulation
        simulation_results = self._run_trade_simulations(
            current_balance=current_balance,
            position_size=proposed_position_size,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips
        )

        # Calculate probabilities
        win_probability = simulation_results['win_probability']
        expected_value = simulation_results['expected_value']
        worst_case_dd = simulation_results['worst_case_dd']
        median_outcome = simulation_results['median_outcome']

        # Make decision
        vote, confidence = self._make_decision(
            win_probability, expected_value, worst_case_dd, median_outcome
        )

        reasoning = {
            "agent": self.name,
            "vote": vote,
            "win_probability": round(win_probability, 3),
            "expected_value": round(expected_value, 2),
            "worst_case_dd": round(worst_case_dd, 4),
            "median_outcome": round(median_outcome, 2),
            "num_simulations": self.num_simulations,
            "veto": self.is_veto and vote == "BLOCK"
        }

        return (vote, confidence, reasoning)

    def _run_trade_simulations(self, current_balance: float, position_size: float,
                                stop_loss_pips: float, take_profit_pips: float) -> Dict:
        """
        Run Monte Carlo simulations for this specific trade.

        Simulates 1000+ outcomes based on historical statistics.
        """
        outcomes = []
        drawdowns = []

        for _ in range(self.num_simulations):
            # Simulate win/loss based on historical win rate
            is_win = np.random.random() < self.historical_win_rate

            if is_win:
                # Winning trade - use take profit
                # Add variance based on historical distribution
                variance_factor = np.random.normal(1.0, self.historical_win_variance)
                pnl_per_lot = take_profit_pips * 10  # $10 per pip for major pairs
                pnl = position_size * pnl_per_lot * variance_factor
            else:
                # Losing trade - use stop loss
                variance_factor = np.random.normal(1.0, self.historical_loss_variance)
                pnl_per_lot = stop_loss_pips * 10
                pnl = -position_size * pnl_per_lot * variance_factor

            outcomes.append(pnl)

            # Calculate potential drawdown from this trade
            if pnl < 0:
                dd_from_trade = abs(pnl) / current_balance
                drawdowns.append(dd_from_trade)

        # Calculate statistics
        wins = sum(1 for o in outcomes if o > 0)
        win_probability = wins / self.num_simulations
        expected_value = np.mean(outcomes)
        median_outcome = np.median(outcomes)
        worst_case_dd = max(drawdowns) if drawdowns else 0.0

        return {
            'win_probability': win_probability,
            'expected_value': expected_value,
            'median_outcome': median_outcome,
            'worst_case_dd': worst_case_dd,
            'outcomes_distribution': outcomes
        }

    def _make_decision(self, win_probability: float, expected_value: float,
                       worst_case_dd: float, median_outcome: float) -> Tuple[str, float]:
        """
        Make trading decision based on Monte Carlo results.

        Decision rules:
        1. If win probability < 55%, BLOCK (unfavorable odds)
        2. If expected value < 0, BLOCK (negative expectancy)
        3. If worst case DD > 2%, BLOCK (too risky)
        4. If median outcome < 0, CAUTION (more likely to lose)
        5. Otherwise ALLOW
        """
        # Rule 1: Win probability check
        if win_probability < self.min_win_probability:
            return ("BLOCK", 0.95)

        # Rule 2: Expected value check (must be positive)
        if expected_value < 0:
            return ("BLOCK", 0.90)

        # Rule 3: Worst-case drawdown check
        if worst_case_dd > self.max_acceptable_dd_risk:
            return ("BLOCK", 0.85)

        # Rule 4: Median outcome check
        if median_outcome < 0:
            return ("CAUTION", 0.70)

        # All checks passed - favorable trade
        confidence = min(win_probability, 0.95)
        return ("ALLOW", confidence)

    def update_statistics(self, trade_result: Dict):
        """
        Update historical statistics based on completed trade.

        This makes the agent smarter over time as it learns
        the actual distribution of outcomes.
        """
        pnl = trade_result.get("pnl", 0)
        outcome = trade_result.get("outcome", "UNKNOWN")

        # Update win rate (exponential moving average)
        alpha = 0.05  # Learning rate
        if outcome == "WIN":
            self.historical_win_rate = (1 - alpha) * self.historical_win_rate + alpha * 1.0

            # Update average win size
            self.historical_avg_win = (1 - alpha) * self.historical_avg_win + alpha * abs(pnl)
        elif outcome == "LOSS":
            self.historical_win_rate = (1 - alpha) * self.historical_win_rate + alpha * 0.0

            # Update average loss size
            self.historical_avg_loss = (1 - alpha) * self.historical_avg_loss + alpha * abs(pnl)

    def get_historical_stats(self) -> Dict:
        """Get current historical statistics"""
        return {
            "win_rate": self.historical_win_rate,
            "avg_win": self.historical_avg_win,
            "avg_loss": self.historical_avg_loss,
            "expectancy": (
                self.historical_win_rate * self.historical_avg_win -
                (1 - self.historical_win_rate) * self.historical_avg_loss
            )
        }

    def set_risk_parameters(self, min_win_prob: float = None,
                           max_dd_risk: float = None,
                           num_sims: int = None):
        """
        Adjust risk parameters dynamically.

        Args:
            min_win_prob: Minimum win probability (0-1)
            max_dd_risk: Maximum acceptable DD risk per trade (0-1)
            num_sims: Number of Monte Carlo simulations
        """
        if min_win_prob is not None:
            self.min_win_probability = min_win_prob

        if max_dd_risk is not None:
            self.max_acceptable_dd_risk = max_dd_risk

        if num_sims is not None:
            self.num_simulations = num_sims

    def run_bulk_simulation(self, trade_scenarios: List[Dict]) -> List[Dict]:
        """
        Run Monte Carlo on multiple trade scenarios.

        Used for strategy optimization - test multiple setups
        and rank them by expected value.

        Args:
            trade_scenarios: List of trade setups to simulate

        Returns:
            List of scenarios ranked by expected value
        """
        results = []

        for scenario in trade_scenarios:
            sim_results = self._run_trade_simulations(
                current_balance=scenario.get('balance', 200000),
                position_size=scenario.get('position_size', 3.0),
                stop_loss_pips=scenario.get('stop_loss_pips', 15),
                take_profit_pips=scenario.get('take_profit_pips', 30)
            )

            results.append({
                'scenario': scenario,
                'win_probability': sim_results['win_probability'],
                'expected_value': sim_results['expected_value'],
                'worst_case_dd': sim_results['worst_case_dd']
            })

        # Rank by expected value
        ranked = sorted(results, key=lambda x: x['expected_value'], reverse=True)

        return ranked

    def stress_test_position(self, position_size: float, num_trades: int = 50) -> Dict:
        """
        Stress test a position size across multiple trades.

        Simulates taking this position size N times to see if it's safe
        for E8 challenge (6% trailing DD limit).

        Args:
            position_size: Position size in lots
            num_trades: Number of sequential trades to simulate

        Returns:
            Stress test results with max DD observed
        """
        starting_balance = 200000
        balance = starting_balance
        peak_balance = starting_balance
        max_drawdown = 0.0

        for _ in range(num_trades):
            # Run single trade simulation
            result = self._run_trade_simulations(
                current_balance=balance,
                position_size=position_size,
                stop_loss_pips=15,
                take_profit_pips=30
            )

            # Use median outcome (more realistic than EV)
            pnl = result['median_outcome']
            balance += pnl

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance

            current_dd = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_dd)

            # Check if we violated E8 limits
            if current_dd >= 0.06:
                return {
                    'position_size': position_size,
                    'max_drawdown': max_drawdown,
                    'final_balance': balance,
                    'trades_survived': _ + 1,
                    'verdict': 'UNSAFE - Exceeds 6% DD limit',
                    'recommendation': 'Reduce position size'
                }

        return {
            'position_size': position_size,
            'max_drawdown': max_drawdown,
            'final_balance': balance,
            'trades_survived': num_trades,
            'verdict': 'SAFE - Within E8 limits' if max_drawdown < 0.06 else 'RISKY',
            'recommendation': 'Acceptable' if max_drawdown < 0.05 else 'Consider reducing'
        }


class MonteCarloAgentAdvanced(MonteCarloAgent):
    """
    Advanced Monte Carlo agent with correlation awareness.

    Takes into account existing positions and their correlations
    when simulating new trade outcomes.
    """

    def __init__(self, initial_weight: float = 2.5, is_veto: bool = True):
        super().__init__(initial_weight=initial_weight, is_veto=is_veto)
        self.name = "MonteCarloAgentAdvanced"

        # Correlation matrix for forex pairs
        self.correlation_matrix = {
            ('EUR_USD', 'GBP_USD'): 0.65,
            ('EUR_USD', 'USD_JPY'): -0.45,
            ('GBP_USD', 'USD_JPY'): -0.40,
            ('EUR_USD', 'AUD_USD'): 0.70,
            ('GBP_USD', 'AUD_USD'): 0.60,
        }

    def analyze_with_portfolio_context(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Run Monte Carlo considering existing positions.

        This accounts for correlation - if you already have EUR/USD long,
        adding GBP/USD long increases risk due to 0.65 correlation.
        """
        existing_positions = market_data.get("existing_positions", [])
        proposed_pair = market_data.get("pair", "UNKNOWN")

        # Calculate correlation-adjusted risk
        correlation_risk = self._calculate_correlation_risk(
            existing_positions, proposed_pair
        )

        # Run base Monte Carlo
        vote, confidence, reasoning = self.analyze(market_data)

        # Adjust decision based on correlation
        if correlation_risk > 0.7:
            # High correlation with existing positions
            vote = "BLOCK"
            confidence = 0.90
            reasoning["correlation_risk"] = correlation_risk
            reasoning["message"] = "High correlation with existing positions"

        return (vote, confidence, reasoning)

    def _calculate_correlation_risk(self, existing_positions: List[Dict],
                                    proposed_pair: str) -> float:
        """Calculate correlation risk score (0-1)"""
        if not existing_positions:
            return 0.0

        max_correlation = 0.0

        for pos in existing_positions:
            pos_pair = pos.get("pair", "")

            # Check both directions
            corr_key1 = (proposed_pair, pos_pair)
            corr_key2 = (pos_pair, proposed_pair)

            corr = self.correlation_matrix.get(
                corr_key1,
                self.correlation_matrix.get(corr_key2, 0.0)
            )

            max_correlation = max(max_correlation, abs(corr))

        return max_correlation

    def forecast_volatility_garch(self, pair: str, price_data: np.ndarray) -> float:
        """
        Forecast next-period volatility using GARCH(1,1) model.
        
        GARCH captures volatility clustering - high volatility periods
        tend to be followed by high volatility periods.
        
        Args:
            pair: Currency pair
            price_data: Array of recent prices (at least 100 observations)
            
        Returns:
            Forecasted volatility (annualized standard deviation)
        """
        try:
            # Calculate returns
            returns = np.diff(np.log(price_data)) * 100  # Convert to percentage
            
            # Need at least 100 observations for GARCH
            if len(returns) < 100:
                # Fall back to simple standard deviation
                return np.std(returns) * np.sqrt(252)  # Annualized
            
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fitted = model.fit(disp='off')
            
            # Forecast next period volatility
            forecast = model_fitted.forecast(horizon=1)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :][0])
            
            # Cache forecast
            self.volatility_forecasts[pair] = {
                'volatility': volatility_forecast,
                'timestamp': np.datetime64('now')
            }
            
            return volatility_forecast
            
        except Exception as e:
            # Fallback to simple volatility if GARCH fails
            return np.std(returns) * np.sqrt(252) if len(returns) > 0 else 15.0
    
    def _run_trade_simulations_advanced(self, current_balance: float, position_size: float,
                                       stop_loss_pips: float, take_profit_pips: float,
                                       pair: str = None, price_data: np.ndarray = None) -> Dict:
        """
        Enhanced Monte Carlo simulation with GARCH volatility and scipy distributions.
        
        Improvements over basic version:
        1. GARCH volatility forecasting (captures volatility clustering)
        2. Scipy t-distribution for fat tails (more realistic than normal)
        3. Volatility-adjusted position sizing
        4. Skewness and kurtosis from actual return distributions
        """
        if stats is None:
            # Fall back to the basic simulation if scipy isn't installed.
            return self._run_trade_simulations(
                current_balance=current_balance,
                position_size=position_size,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
            )

        # Forecast volatility if price data provided
        if price_data is not None and len(price_data) >= 100:
            forecasted_vol = self.forecast_volatility_garch(pair, price_data)
            vol_adjustment = forecasted_vol / 15.0  # Normalize to typical 15% vol
        else:
            vol_adjustment = 1.0
        
        outcomes = []
        drawdowns = []
        
        for _ in range(self.num_simulations):
            # Simulate win/loss based on historical win rate
            is_win = np.random.random() < self.historical_win_rate
            
            if is_win:
                # Use t-distribution for fat tails (more realistic than normal)
                # df=5 gives heavier tails than normal distribution
                variance_factor = stats.t.rvs(df=5, loc=1.0, scale=self.historical_win_variance / 2.5)
                variance_factor = max(0.2, min(2.5, variance_factor))  # Clip extremes
                
                pnl_per_lot = take_profit_pips * 10 * vol_adjustment
                pnl = position_size * pnl_per_lot * variance_factor
            else:
                # Losses also use t-distribution
                variance_factor = stats.t.rvs(df=5, loc=1.0, scale=self.historical_loss_variance / 2.5)
                variance_factor = max(0.2, min(2.5, variance_factor))
                
                pnl_per_lot = stop_loss_pips * 10 * vol_adjustment
                pnl = -position_size * pnl_per_lot * variance_factor
            
            outcomes.append(pnl)
            
            # Calculate potential drawdown
            if pnl < 0:
                dd_from_trade = abs(pnl) / current_balance
                drawdowns.append(dd_from_trade)
        
        # Calculate statistics using scipy
        wins = sum(1 for o in outcomes if o > 0)
        win_probability = wins / self.num_simulations
        expected_value = np.mean(outcomes)
        median_outcome = np.median(outcomes)
        worst_case_dd = max(drawdowns) if drawdowns else 0.0
        
        # Additional statistics with scipy
        skewness = stats.skew(outcomes)
        kurtosis = stats.kurtosis(outcomes)
        
        # Calculate Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(outcomes, 5)  # 5th percentile (95% VaR)
        
        # Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        cvar_95 = np.mean([o for o in outcomes if o <= var_95])
        
        return {
            'win_probability': win_probability,
            'expected_value': expected_value,
            'median_outcome': median_outcome,
            'worst_case_dd': worst_case_dd,
            'outcomes_distribution': outcomes,
            'volatility_adjustment': vol_adjustment,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def update_price_history(self, pair: str, price: float):
        """
        Update price history for GARCH modeling.
        
        Args:
            pair: Currency pair
            price: Current price
        """
        if pair not in self.price_returns:
            self.price_returns[pair] = []
        
        self.price_returns[pair].append(price)
        
        # Keep last 500 prices (enough for GARCH)
        if len(self.price_returns[pair]) > 500:
            self.price_returns[pair] = self.price_returns[pair][-500:]
