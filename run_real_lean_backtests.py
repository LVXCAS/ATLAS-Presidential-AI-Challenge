"""
RUN REAL LEAN BACKTESTS
======================
Execute real LEAN backtests with actual historical data
"""

import subprocess
import json
import os
import time
from datetime import datetime

class RealLEANBacktestRunner:
    """Run real LEAN backtests with actual data"""

    def __init__(self):
        self.lean_config_template = {
            "algorithm-type-name": "",
            "algorithm-language": "Python",
            "algorithm-location": "",
            "data-folder": "./lean_engine/Data",
            "results-destination-folder": "./lean_engine/Results",
            "live-mode": False,
            "data-queue-handler": "LocalDataQueueHandler",
            "setup-handler": "BasicSetupHandler",
            "result-handler": "BacktestingResultHandler",
            "history-provider": "LocalHistoryProvider",
            "cash": 100000,
            "start-date": "2020-01-01",
            "end-date": "2024-09-18",
            "python-additional-paths": ["."],
            "debug-mode": True,
            "log-level": "Debug"
        }

        self.strategies = [
            {
                'name': 'RealMomentumStrategy',
                'file': 'real_momentum_strategy.py',
                'description': 'Real momentum strategy using 12-1 month momentum'
            },
            {
                'name': 'RealMeanReversionStrategy',
                'file': 'real_mean_reversion_strategy.py',
                'description': 'Real mean reversion strategy using RSI and Bollinger Bands'
            },
            {
                'name': 'RealVolatilityStrategy',
                'file': 'real_volatility_strategy.py',
                'description': 'Real volatility strategy using volatility regimes'
            }
        ]

        self.results = {}

    def create_lean_config(self, strategy_name, strategy_file):
        """Create LEAN config for a specific strategy"""
        config = self.lean_config_template.copy()
        config['algorithm-type-name'] = strategy_name
        config['algorithm-location'] = strategy_file

        config_file = f"lean_config_{strategy_name.lower()}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config_file

    def run_single_backtest(self, strategy):
        """Run a single LEAN backtest"""
        print(f"\\n{'='*60}")
        print(f"RUNNING: {strategy['name']}")
        print(f"DESCRIPTION: {strategy['description']}")
        print(f"{'='*60}")

        try:
            # Create config file
            config_file = self.create_lean_config(strategy['name'], strategy['file'])

            # Copy strategy file to lean_engine directory for easier access
            strategy_destination = f"lean_engine/{strategy['file']}"
            with open(strategy['file'], 'r') as src, open(strategy_destination, 'w') as dst:
                dst.write(src.read())

            # Update config to point to lean_engine location
            with open(config_file, 'r') as f:
                config = json.load(f)
            config['algorithm-location'] = strategy_destination
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Configuration: {config_file}")
            print(f"Algorithm: {strategy_destination}")

            # Run LEAN backtest
            start_time = time.time()

            # Use the config file we created
            result = subprocess.run([
                'lean', 'backtest',
                '--config', config_file,
                '--verbose'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

            end_time = time.time()
            duration = end_time - start_time

            print(f"\\nBacktest completed in {duration:.1f} seconds")
            print(f"Return code: {result.returncode}")

            # Parse results
            if result.returncode == 0:
                print("[SUCCESS] Backtest completed successfully")

                # Try to extract results from output
                output_lines = result.stdout.split('\\n')
                debug_lines = [line for line in output_lines if 'DEBUG' in line or 'RESULTS' in line]

                results_data = self.parse_lean_output(result.stdout, strategy['name'])
                self.results[strategy['name']] = {
                    'status': 'SUCCESS',
                    'duration': duration,
                    'results': results_data,
                    'output': result.stdout[-2000:],  # Last 2000 chars
                    'debug_lines': debug_lines[-20:]  # Last 20 debug lines
                }

                print("\\nKEY RESULTS:")
                for key, value in results_data.items():
                    print(f"  {key}: {value}")

            else:
                print(f"[ERROR] Backtest failed")
                print(f"Error output: {result.stderr}")

                self.results[strategy['name']] = {
                    'status': 'FAILED',
                    'duration': duration,
                    'error': result.stderr,
                    'output': result.stdout
                }

            # Show some output regardless
            if result.stdout:
                print("\\nSample output:")
                output_lines = result.stdout.split('\\n')
                relevant_lines = [line for line in output_lines if any(keyword in line.upper()
                                                                     for keyword in ['RETURN', 'SHARPE', 'DRAWDOWN', 'RESULTS', 'TOTAL'])]
                for line in relevant_lines[-10:]:  # Last 10 relevant lines
                    print(f"  {line}")

        except subprocess.TimeoutExpired:
            print("[TIMEOUT] Backtest timed out after 5 minutes")
            self.results[strategy['name']] = {
                'status': 'TIMEOUT',
                'duration': 300,
                'error': 'Backtest timed out'
            }

        except Exception as e:
            print(f"[ERROR] Exception during backtest: {e}")
            self.results[strategy['name']] = {
                'status': 'EXCEPTION',
                'error': str(e)
            }

    def parse_lean_output(self, output, strategy_name):
        """Parse LEAN output to extract performance metrics"""
        results = {
            'total_return': 'N/A',
            'annual_return': 'N/A',
            'sharpe_ratio': 'N/A',
            'max_drawdown': 'N/A',
            'win_rate': 'N/A',
            'total_trades': 'N/A'
        }

        try:
            lines = output.split('\\n')

            for line in lines:
                line_upper = line.upper()

                if 'TOTAL RETURN:' in line_upper:
                    results['total_return'] = line.split(':')[-1].strip()
                elif 'ANNUAL RETURN:' in line_upper:
                    results['annual_return'] = line.split(':')[-1].strip()
                elif 'SHARPE RATIO:' in line_upper:
                    results['sharpe_ratio'] = line.split(':')[-1].strip()
                elif 'MAX DRAWDOWN:' in line_upper:
                    results['max_drawdown'] = line.split(':')[-1].strip()
                elif 'WIN RATE:' in line_upper:
                    results['win_rate'] = line.split(':')[-1].strip()
                elif 'TOTAL TRADES:' in line_upper:
                    results['total_trades'] = line.split(':')[-1].strip()

        except Exception as e:
            print(f"Error parsing output: {e}")

        return results

    def run_all_backtests(self):
        """Run all strategy backtests"""
        print("=" * 80)
        print("REAL LEAN BACKTESTING WITH ACTUAL HISTORICAL DATA")
        print("=" * 80)
        print(f"Start time: {datetime.now()}")
        print(f"Strategies to test: {len(self.strategies)}")

        # Check if we have data
        if not os.path.exists("./lean_engine/Data/equity/usa/daily/spy.csv"):
            print("\\n[ERROR] Historical data not found!")
            print("Please run setup_real_lean_data.py first")
            return

        print("\\n[OK] Historical data found")

        # Run each strategy
        for i, strategy in enumerate(self.strategies):
            print(f"\\nRunning strategy {i+1}/{len(self.strategies)}")
            self.run_single_backtest(strategy)

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate summary of all backtest results"""
        print("\\n" + "=" * 80)
        print("REAL BACKTEST RESULTS SUMMARY")
        print("=" * 80)

        successful_backtests = 0

        for strategy_name, result in self.results.items():
            status = result['status']
            duration = result.get('duration', 0)

            print(f"\\n{strategy_name}:")
            print(f"  Status: {status}")
            print(f"  Duration: {duration:.1f}s")

            if status == 'SUCCESS':
                successful_backtests += 1
                results_data = result['results']
                print(f"  Total Return: {results_data.get('total_return', 'N/A')}")
                print(f"  Sharpe Ratio: {results_data.get('sharpe_ratio', 'N/A')}")
                print(f"  Max Drawdown: {results_data.get('max_drawdown', 'N/A')}")
                print(f"  Win Rate: {results_data.get('win_rate', 'N/A')}")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  Error: {error[:100]}...")

        print(f"\\nSUMMARY:")
        print(f"Total strategies: {len(self.strategies)}")
        print(f"Successful backtests: {successful_backtests}")
        print(f"Failed backtests: {len(self.strategies) - successful_backtests}")

        # Save results
        results_file = f"real_lean_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\\nDetailed results saved to: {results_file}")

        if successful_backtests > 0:
            print("\\n[SUCCESS] Real LEAN backtesting completed!")
            print("You now have genuine performance results from actual historical data!")
        else:
            print("\\n[WARNING] No successful backtests - check logs for issues")

def main():
    """Run real LEAN backtests"""
    runner = RealLEANBacktestRunner()
    runner.run_all_backtests()

if __name__ == "__main__":
    main()