import sys, time, random, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')

class MonteCarloSimulation:
    def __init__(self):
        self.num_simulations = 100000
        self.initial_capital = 10000
        self.trading_days = 252
        self.risk_free_rate = 0.05
        self.win_rate_range = (0.45, 0.65)
        self.avg_win_range = (0.15, 0.35)
        self.avg_loss_range = (-0.08, -0.25)
        self.trades_per_day_range = (0.5, 2.5)
        self.commission_per_trade = 1.50
        self.bull_market_prob = 0.35
        self.bear_market_prob = 0.25

    def get_market_multipliers(self, market_condition):
        if market_condition == 'bull':
            return {'win_rate_multiplier': 1.15, 'avg_win_multiplier': 1.25, 'avg_loss_multiplier': 0.85}
        elif market_condition == 'bear':
            return {'win_rate_multiplier': 0.80, 'avg_win_multiplier': 0.85, 'avg_loss_multiplier': 1.30}
        else:
            return {'win_rate_multiplier': 0.95, 'avg_win_multiplier': 0.90, 'avg_loss_multiplier': 1.10}

    def simulate_single_day(self, capital, market_condition, trading_params):
        multipliers = self.get_market_multipliers(market_condition)
        adjusted_win_rate = min(0.95, trading_params['win_rate'] * multipliers['win_rate_multiplier'])
        adjusted_avg_win = trading_params['avg_win'] * multipliers['avg_win_multiplier']
        adjusted_avg_loss = trading_params['avg_loss'] * multipliers['avg_loss_multiplier']
        num_trades = max(0, int(np.random.poisson(trading_params['trades_per_day'])))
        daily_return = 0.0
        for _ in range(num_trades):
            if random.random() < adjusted_win_rate:
                trade_return = random.uniform(adjusted_avg_win * 0.5, adjusted_avg_win * 1.5)
            else:
                trade_return = random.uniform(adjusted_avg_loss * 1.5, adjusted_avg_loss * 0.5)
            position_size = min(0.05, abs(trade_return) * adjusted_win_rate)
            daily_return += trade_return * position_size
        commission_impact = (num_trades * self.commission_per_trade) / capital
        final_return = daily_return - commission_impact
        return capital * (1 + final_return), num_trades, final_return

    def run_single_simulation(self, sim_id):
        capital = self.initial_capital
        daily_returns = []
        total_trades = 0
        max_drawdown = 0.0
        peak_capital = capital
        trading_params = {
            'win_rate': random.uniform(*self.win_rate_range),
            'avg_win': random.uniform(*self.avg_win_range),
            'avg_loss': random.uniform(*self.avg_loss_range),
            'trades_per_day': random.uniform(*self.trades_per_day_range)
        }
        for day in range(self.trading_days):
            rand = random.random()
            if rand < self.bull_market_prob:
                market_condition = 'bull'
            elif rand < self.bull_market_prob + self.bear_market_prob:
                market_condition = 'bear'
            else:
                market_condition = 'sideways'
            new_capital, day_trades, day_return = self.simulate_single_day(capital, market_condition, trading_params)
            capital = new_capital
            daily_returns.append(day_return)
            total_trades += day_trades
            if capital > peak_capital:
                peak_capital = capital
            else:
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)
        total_return = (capital - self.initial_capital) / self.initial_capital
        if len(daily_returns) > 0:
            daily_returns_array = np.array(daily_returns)
            avg_daily_return = np.mean(daily_returns_array)
            daily_volatility = np.std(daily_returns_array)
            annual_return = avg_daily_return * self.trading_days
            annual_volatility = daily_volatility * np.sqrt(self.trading_days)
            excess_return = annual_return - self.risk_free_rate
            sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        else:
            annual_return = 0
            annual_volatility = 0
            sharpe_ratio = 0
        return {
            'final_capital': capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }

    def run_monte_carlo_simulation(self):
        print(f'Starting Monte Carlo simulation with {self.num_simulations:,} iterations...')
        start_time = time.time()
        results = []
        batch_size = 10000
        for batch_start in range(0, self.num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, self.num_simulations)
            print(f'Batch {batch_start//batch_size + 1}/{(self.num_simulations-1)//batch_size + 1}')
            for i in range(batch_start, batch_end):
                result = self.run_single_simulation(i + 1)
                results.append(result)
        end_time = time.time()
        simulation_time = end_time - start_time
        print(f'Simulation completed in {simulation_time:.1f} seconds')
        return results, simulation_time

print('=' * 60)
print('OPTIONS_BOT MONTE CARLO SIMULATION - 100,000 ITERATIONS')
print('=' * 60)
simulation = MonteCarloSimulation()
results, simulation_time = simulation.run_monte_carlo_simulation()
df = pd.DataFrame(results)
