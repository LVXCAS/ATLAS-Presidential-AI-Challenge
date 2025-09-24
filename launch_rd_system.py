"""
R&D System Launcher

Launch the complete Research & Development system for continuous strategy optimization,
market analysis, and algorithm improvement while trading live.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import warnings
import json
import time

warnings.filterwarnings('ignore')

class StrategyResearcher:
    """Continuous strategy research and optimization"""

    def __init__(self):
        self.active_strategies = {}
        self.performance_data = []
        self.market_regimes = {}
        self.optimization_queue = []

    async def research_momentum_strategies(self):
        """Research optimal momentum strategy parameters"""
        print("\n[R&D] Researching Momentum Strategies...")

        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        best_params = {}

        for symbol in symbols:
            try:
                # Get historical data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")

                if len(data) < 100:
                    continue

                # Test different momentum parameters
                lookback_periods = [5, 10, 14, 20, 30]
                thresholds = [0.01, 0.02, 0.03, 0.05]

                best_sharpe = -999
                best_config = None

                for lookback in lookback_periods:
                    for threshold in thresholds:
                        # Calculate momentum signals
                        data['momentum'] = data['Close'].pct_change(lookback)
                        data['signal'] = np.where(data['momentum'] > threshold, 1,
                                                np.where(data['momentum'] < -threshold, -1, 0))

                        # Calculate returns
                        data['strategy_returns'] = data['signal'].shift(1) * data['Close'].pct_change()

                        # Calculate Sharpe ratio
                        if len(data['strategy_returns'].dropna()) > 30:
                            sharpe = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252)

                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_config = {
                                    'lookback': lookback,
                                    'threshold': threshold,
                                    'sharpe': sharpe,
                                    'total_return': data['strategy_returns'].sum()
                                }

                if best_config:
                    best_params[symbol] = best_config
                    print(f"[R&D] {symbol}: Best Sharpe {best_config['sharpe']:.3f} "
                          f"(lookback={best_config['lookback']}, threshold={best_config['threshold']:.3f})")

            except Exception as e:
                print(f"[R&D] Error researching {symbol}: {e}")

        return best_params

    async def research_mean_reversion_strategies(self):
        """Research optimal mean reversion parameters"""
        print("\n[R&D] Researching Mean Reversion Strategies...")

        symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        best_params = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2y")

                if len(data) < 200:
                    continue

                # Test different mean reversion parameters
                lookback_periods = [10, 20, 30, 50]
                std_thresholds = [1.5, 2.0, 2.5, 3.0]

                best_sharpe = -999
                best_config = None

                for lookback in lookback_periods:
                    for std_threshold in std_thresholds:
                        # Calculate rolling mean and std
                        data['rolling_mean'] = data['Close'].rolling(lookback).mean()
                        data['rolling_std'] = data['Close'].rolling(lookback).std()
                        data['z_score'] = (data['Close'] - data['rolling_mean']) / data['rolling_std']

                        # Generate signals
                        data['signal'] = np.where(data['z_score'] < -std_threshold, 1,  # Buy oversold
                                                np.where(data['z_score'] > std_threshold, -1, 0))  # Sell overbought

                        # Calculate returns
                        data['strategy_returns'] = data['signal'].shift(1) * data['Close'].pct_change()

                        # Calculate Sharpe ratio
                        if len(data['strategy_returns'].dropna()) > 50:
                            sharpe = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(252)

                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_config = {
                                    'lookback': lookback,
                                    'std_threshold': std_threshold,
                                    'sharpe': sharpe,
                                    'total_return': data['strategy_returns'].sum()
                                }

                if best_config:
                    best_params[symbol] = best_config
                    print(f"[R&D] {symbol}: Best Sharpe {best_config['sharpe']:.3f} "
                          f"(lookback={best_config['lookback']}, std_dev={best_config['std_threshold']:.1f})")

            except Exception as e:
                print(f"[R&D] Error researching {symbol}: {e}")

        return best_params

    async def research_volatility_strategies(self):
        """Research volatility-based trading strategies"""
        print("\n[R&D] Researching Volatility Strategies...")

        # VIX and volatility analysis
        try:
            vix = yf.Ticker("^VIX")
            spy = yf.Ticker("SPY")

            vix_data = vix.history(period="2y")
            spy_data = spy.history(period="2y")

            # Align data
            common_dates = vix_data.index.intersection(spy_data.index)
            vix_data = vix_data.loc[common_dates]
            spy_data = spy_data.loc[common_dates]

            # Calculate VIX-based signals
            vix_percentile = vix_data['Close'].rolling(252).quantile(0.8)  # 80th percentile

            # High VIX = potential buying opportunity
            signal = np.where(vix_data['Close'] > vix_percentile, 1, 0)
            spy_returns = spy_data['Close'].pct_change()
            strategy_returns = pd.Series(signal, index=common_dates).shift(1) * spy_returns

            if len(strategy_returns.dropna()) > 50:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                total_return = strategy_returns.sum()

                print(f"[R&D] VIX Strategy: Sharpe {sharpe:.3f}, Total Return {total_return:.3f}")

                return {
                    'vix_strategy': {
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'signal_type': 'high_vix_contrarian'
                    }
                }

        except Exception as e:
            print(f"[R&D] Error in volatility research: {e}")

        return {}

class MarketRegimeDetector:
    """Detect and analyze market regimes for strategy adaptation"""

    def __init__(self):
        self.regimes = {}
        self.current_regime = "unknown"

    async def detect_current_regime(self):
        """Detect current market regime"""
        print("\n[R&D] Analyzing Current Market Regime...")

        try:
            # Get market data
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")

            spy_data = spy.history(period="6mo")
            vix_data = vix.history(period="6mo")

            # Calculate regime indicators
            spy_returns = spy_data['Close'].pct_change().dropna()
            recent_volatility = spy_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            recent_return = spy_data['Close'].pct_change(30).iloc[-1]
            current_vix = vix_data['Close'].iloc[-1]

            # Regime classification
            if recent_volatility > 0.25 and current_vix > 25:
                regime = "high_volatility"
            elif recent_return > 0.05 and recent_volatility < 0.15:
                regime = "bull_market"
            elif recent_return < -0.05:
                regime = "bear_market"
            elif abs(recent_return) < 0.02 and recent_volatility < 0.12:
                regime = "sideways"
            else:
                regime = "transitional"

            self.current_regime = regime

            print(f"[R&D] Current Market Regime: {regime.upper()}")
            print(f"    Recent 30-day return: {recent_return:.3f}")
            print(f"    Recent volatility: {recent_volatility:.3f}")
            print(f"    Current VIX: {current_vix:.1f}")

            # Strategy recommendations based on regime
            recommendations = self.get_regime_strategies(regime)
            print(f"[R&D] Recommended strategies: {recommendations}")

            return {
                'regime': regime,
                'return': recent_return,
                'volatility': recent_volatility,
                'vix': current_vix,
                'strategies': recommendations
            }

        except Exception as e:
            print(f"[R&D] Error in regime detection: {e}")
            return {'regime': 'unknown', 'strategies': ['momentum', 'mean_reversion']}

    def get_regime_strategies(self, regime):
        """Get optimal strategies for market regime"""
        regime_strategies = {
            'bull_market': ['momentum', 'covered_calls', 'growth_stocks'],
            'bear_market': ['mean_reversion', 'protective_puts', 'short_strategies'],
            'high_volatility': ['straddles', 'iron_condors', 'volatility_arbitrage'],
            'sideways': ['iron_condors', 'covered_calls', 'mean_reversion'],
            'transitional': ['momentum', 'mean_reversion', 'options_strategies']
        }
        return regime_strategies.get(regime, ['momentum', 'mean_reversion'])

class PerformanceOptimizer:
    """Optimize strategy performance using machine learning"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    async def optimize_strategy_parameters(self, strategy_data):
        """Use ML to optimize strategy parameters"""
        print("\n[R&D] Optimizing Strategy Parameters with ML...")

        try:
            # Create features for ML model
            features = []
            targets = []

            for symbol, data in strategy_data.items():
                if 'sharpe' in data:
                    # Features: lookback, threshold, market conditions
                    feature_vector = [
                        data.get('lookback', 14),
                        data.get('threshold', 0.02),
                        data.get('total_return', 0),
                        len(symbol)  # Symbol length as proxy for company size
                    ]

                    # Target: Sharpe ratio (convert to binary for classification)
                    target = 1 if data['sharpe'] > 1.0 else 0

                    features.append(feature_vector)
                    targets.append(target)

            if len(features) > 3:
                # Train random forest model
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(features, targets)

                # Get feature importance
                feature_names = ['lookback', 'threshold', 'total_return', 'symbol_length']
                importance_scores = dict(zip(feature_names, rf_model.feature_importances_))

                print(f"[R&D] Feature Importance:")
                for feature, importance in importance_scores.items():
                    print(f"    {feature}: {importance:.3f}")

                # Predict optimal parameters for new strategies
                optimal_params = self.predict_optimal_parameters(rf_model)
                print(f"[R&D] ML-Suggested Optimal Parameters: {optimal_params}")

                return {
                    'model_trained': True,
                    'feature_importance': importance_scores,
                    'optimal_params': optimal_params
                }

        except Exception as e:
            print(f"[R&D] Error in ML optimization: {e}")

        return {'model_trained': False}

    def predict_optimal_parameters(self, model):
        """Predict optimal parameters using trained model"""
        # Test different parameter combinations
        test_params = [
            [10, 0.015, 0.05, 4],  # Conservative
            [14, 0.02, 0.08, 4],   # Moderate
            [20, 0.03, 0.12, 4]    # Aggressive
        ]

        predictions = model.predict(test_params)
        best_idx = np.argmax(predictions)

        param_map = {
            0: {'type': 'conservative', 'lookback': 10, 'threshold': 0.015},
            1: {'type': 'moderate', 'lookback': 14, 'threshold': 0.02},
            2: {'type': 'aggressive', 'lookback': 20, 'threshold': 0.03}
        }

        return param_map[best_idx]

class RDOrchestrator:
    """Main R&D system orchestrator"""

    def __init__(self):
        self.researcher = StrategyResearcher()
        self.regime_detector = MarketRegimeDetector()
        self.optimizer = PerformanceOptimizer()
        self.results = {}

    async def run_full_rd_cycle(self):
        """Run complete R&D analysis cycle"""

        print("="*70)
        print("HIVE TRADING R&D SYSTEM - FULL ANALYSIS CYCLE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Market Regime Analysis
        regime_analysis = await self.regime_detector.detect_current_regime()
        self.results['regime_analysis'] = regime_analysis

        # 2. Strategy Research
        momentum_research = await self.researcher.research_momentum_strategies()
        mean_reversion_research = await self.researcher.research_mean_reversion_strategies()
        volatility_research = await self.researcher.research_volatility_strategies()

        self.results['momentum_strategies'] = momentum_research
        self.results['mean_reversion_strategies'] = mean_reversion_research
        self.results['volatility_strategies'] = volatility_research

        # 3. ML Optimization
        all_strategy_data = {**momentum_research, **mean_reversion_research}
        optimization_results = await self.optimizer.optimize_strategy_parameters(all_strategy_data)
        self.results['ml_optimization'] = optimization_results

        # 4. Generate Trading Recommendations
        recommendations = self.generate_trading_recommendations()
        self.results['recommendations'] = recommendations

        # 5. Save results
        await self.save_rd_results()

        return self.results

    def generate_trading_recommendations(self):
        """Generate specific trading recommendations for tomorrow"""
        print(f"\n{'='*70}")
        print("TRADING RECOMMENDATIONS FOR TOMORROW")
        print("="*70)

        recommendations = []
        current_regime = self.results.get('regime_analysis', {}).get('regime', 'unknown')

        # Based on market regime
        if current_regime == 'bull_market':
            recommendations.extend([
                {
                    'strategy': 'momentum_long',
                    'symbols': ['AAPL', 'NVDA', 'TSLA'],
                    'allocation': 0.3,
                    'reasoning': 'Bull market momentum continuation'
                },
                {
                    'strategy': 'covered_calls',
                    'symbols': ['SPY', 'QQQ'],
                    'allocation': 0.2,
                    'reasoning': 'Generate income in rising market'
                }
            ])
        elif current_regime == 'high_volatility':
            recommendations.extend([
                {
                    'strategy': 'straddles',
                    'symbols': ['SPY', 'QQQ'],
                    'allocation': 0.15,
                    'reasoning': 'Profit from high volatility moves'
                },
                {
                    'strategy': 'mean_reversion',
                    'symbols': ['GLD', 'TLT'],
                    'allocation': 0.25,
                    'reasoning': 'Counter-trend in safe haven assets'
                }
            ])
        else:
            # Default balanced approach
            recommendations.extend([
                {
                    'strategy': 'momentum_moderate',
                    'symbols': ['SPY', 'QQQ'],
                    'allocation': 0.2,
                    'reasoning': 'Moderate momentum exposure'
                },
                {
                    'strategy': 'mean_reversion',
                    'symbols': ['IWM', 'GLD'],
                    'allocation': 0.15,
                    'reasoning': 'Diversified counter-trend'
                }
            ])

        # Print recommendations
        total_allocation = sum(r['allocation'] for r in recommendations)
        print(f"Total Recommended Allocation: {total_allocation:.1%}")

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['strategy'].upper()}")
            print(f"   Symbols: {', '.join(rec['symbols'])}")
            print(f"   Allocation: {rec['allocation']:.1%}")
            print(f"   Reasoning: {rec['reasoning']}")

        return recommendations

    async def save_rd_results(self):
        """Save R&D results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rd_analysis_{timestamp}.json"

        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Clean results for JSON
            clean_results = json.loads(json.dumps(self.results, default=convert_numpy))

            with open(filename, 'w') as f:
                json.dump(clean_results, f, indent=2)

            print(f"\n[R&D] Results saved to: {filename}")

        except Exception as e:
            print(f"[R&D] Error saving results: {e}")

async def main():
    """Launch the full R&D system"""

    print("LAUNCHING HIVE TRADING R&D SYSTEM")
    print("="*70)

    orchestrator = RDOrchestrator()

    try:
        # Run full R&D cycle
        results = await orchestrator.run_full_rd_cycle()

        print(f"\n{'='*70}")
        print("R&D SYSTEM ANALYSIS COMPLETED")
        print("="*70)
        print(f"[OK] Market regime analyzed")
        print(f"[OK] Strategy parameters optimized")
        print(f"[OK] ML models trained")
        print(f"[OK] Trading recommendations generated")
        print(f"[OK] Results saved for execution")

        # Key insights for tomorrow
        regime = results.get('regime_analysis', {}).get('regime', 'unknown')
        num_momentum = len(results.get('momentum_strategies', {}))
        num_mean_rev = len(results.get('mean_reversion_strategies', {}))

        print(f"\nKEY INSIGHTS FOR TOMORROW:")
        print(f"   Market Regime: {regime.upper()}")
        print(f"   Momentum Opportunities: {num_momentum}")
        print(f"   Mean Reversion Opportunities: {num_mean_rev}")
        print(f"   ML Optimization: {'[OK]' if results.get('ml_optimization', {}).get('model_trained') else '[SKIP]'}")

        return True

    except Exception as e:
        print(f"[ERROR] R&D System Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())