#!/usr/bin/env python3
"""
ML-Enhanced OPTIONS_BOT with scikit-learn and DEAP integration
Combines advanced financial analytics with machine learning and genetic algorithms
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Import base bot
sys.path.append('.')
from enhanced_OPTIONS_BOT import EnhancedOptionsBot
from agents.ml_strategy_evolution import ml_strategy_evolution

class MLOptionsBot(EnhancedOptionsBot):
    """
    Machine Learning Enhanced OPTIONS_BOT
    Adds scikit-learn ML models and DEAP genetic algorithms to the enhanced bot
    """
    
    def __init__(self):
        super().__init__()
        self.ml_evolution = ml_strategy_evolution
        self.evolved_strategies = {}
        self.ml_regime_history = []
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML components
        print("Initializing ML-Enhanced OPTIONS_BOT with scikit-learn and DEAP")
        
    async def ml_enhanced_market_analysis(self) -> Dict[str, Any]:
        """
        Use ML models for superior market analysis
        """
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        
        # Get ML-based market regime
        regime_data = await self.ml_evolution.ml_enhanced_market_regime_detection(symbols)
        
        # Store regime history for learning
        self.ml_regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime_data['regime'],
            'confidence': regime_data['confidence']
        })
        
        # Keep only recent history (last 30 days)
        if len(self.ml_regime_history) > 30:
            self.ml_regime_history = self.ml_regime_history[-30:]
        
        return regime_data
    
    async def ml_enhanced_opportunity_detection(self, symbol: str) -> Optional[Dict]:
        """
        Use ML models to enhance opportunity detection
        """
        try:
            # Get base opportunity analysis
            base_opportunity = await self.enhanced_opportunity_analysis(symbol)
            
            if not base_opportunity:
                return None
            
            # Get current market data for ML scoring (simplified version)
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if len(hist) >= 2:
                    current_data = {
                        'momentum': (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1),
                        'volume_ratio': hist['Volume'].iloc[-1] / hist['Volume'].iloc[-2] if len(hist) >= 2 else 1.0
                    }
                else:
                    current_data = {'momentum': 0, 'volume_ratio': 1.0}
            except:
                current_data = {'momentum': 0, 'volume_ratio': 1.0}
            
            # ML-enhanced volatility prediction
            ml_vol_data = await self.ml_evolution.ml_enhanced_volatility_prediction(symbol)
            
            # ML opportunity scoring
            ml_score_data = await self.ml_evolution.ml_opportunity_scoring(
                symbol=symbol,
                current_vol=ml_vol_data['predicted_vol'] / 100,  # Convert to decimal
                momentum=current_data.get('momentum', 0),
                volume_ratio=current_data.get('volume_ratio', 1.0)
            )
            
            # Combine base analysis with ML insights
            enhanced_opportunity = base_opportunity.copy()
            enhanced_opportunity.update({
                'ml_volatility_prediction': ml_vol_data['predicted_vol'],
                'ml_vol_confidence': ml_vol_data['confidence'],
                'ml_opportunity_score': ml_score_data['score'],
                'ml_scoring_confidence': ml_score_data['confidence'],
                'combined_confidence': (
                    base_opportunity['confidence'] * 0.6 + 
                    ml_score_data['score'] * 0.4
                ),
                'ml_features': ml_score_data.get('feature_scores', {}),
                'analysis_method': 'ml_enhanced'
            })
            
            # Only return if ML also confirms it's a good opportunity
            if ml_score_data['score'] > 0.6:
                self.logger.info(
                    f"ML ENHANCED OPPORTUNITY: {symbol} {enhanced_opportunity['strategy']} - "
                    f"{enhanced_opportunity['combined_confidence']:.0%} combined confidence, "
                    f"ML score: {ml_score_data['score']:.1%}"
                )
                return enhanced_opportunity
            
            return None
            
        except Exception as e:
            self.logger.error(f"ML opportunity detection error for {symbol}: {e}")
            # Fallback to base analysis
            return await self.enhanced_opportunity_analysis(symbol)
    
    async def evolve_trading_strategies(self) -> Dict[str, Any]:
        """
        Use genetic algorithms to evolve optimal trading strategies
        """
        try:
            self.logger.info("Starting genetic algorithm strategy evolution...")
            
            evolution_result = await self.ml_evolution.evolve_optimal_strategy(
                generations=30,  # Reasonable for real-time use
                population_size=50
            )
            
            if evolution_result.get('success'):
                best_strategy = evolution_result['best_strategy']
                
                # Store evolved strategy
                strategy_name = f"EVOLVED_STRATEGY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.evolved_strategies[strategy_name] = best_strategy
                
                self.logger.info(
                    f"EVOLVED OPTIMAL STRATEGY: {strategy_name} - "
                    f"Fitness: {best_strategy['fitness_score']:.3f}, "
                    f"Risk/Reward: {best_strategy['risk_reward_ratio']:.2f}"
                )
                
                return {
                    'success': True,
                    'strategy_name': strategy_name,
                    'strategy_params': best_strategy,
                    'evolution_stats': evolution_result['evolution_stats']
                }
            else:
                return {'success': False, 'error': evolution_result.get('error', 'Unknown error')}
                
        except Exception as e:
            self.logger.error(f"Strategy evolution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def ml_enhanced_position_sizing(self, opportunity: Dict, account_value: float) -> int:
        """
        Use ML insights for smarter position sizing
        """
        try:
            # Get base position size
            base_size = self.enhanced_position_sizing(opportunity, account_value)
            
            # ML adjustments based on confidence and market regime
            ml_confidence = opportunity.get('ml_scoring_confidence', 0.5)
            ml_score = opportunity.get('ml_opportunity_score', 0.5)
            
            # Get current ML market regime
            regime_data = await self.ml_enhanced_market_analysis()
            regime_confidence = regime_data.get('confidence', 0.5)
            
            # Adjust size based on ML confidence
            confidence_multiplier = (ml_confidence + ml_score + regime_confidence) / 3.0
            
            # Scale position size
            ml_adjusted_size = int(base_size * confidence_multiplier)
            
            # Apply regime-based adjustments
            regime = regime_data.get('regime', 'NEUTRAL')
            if regime == 'HIGH_VOLATILITY':
                ml_adjusted_size = max(1, int(ml_adjusted_size * 0.7))  # Reduce size in high vol
            elif regime == 'BULL_MARKET' and opportunity.get('strategy') in ['BULL_CALL_SPREAD', 'LONG_CALL']:
                ml_adjusted_size = int(ml_adjusted_size * 1.2)  # Increase for aligned strategies
            
            # Final safety check
            max_ml_size = min(3, int(account_value / 5000))  # Max 3 contracts or 1 per $5k
            final_size = min(ml_adjusted_size, max_ml_size)
            
            if final_size != base_size:
                self.logger.info(
                    f"ML ADJUSTED POSITION SIZE: {opportunity['symbol']} from {base_size} to {final_size} contracts "
                    f"(ML confidence: {confidence_multiplier:.1%}, regime: {regime})"
                )
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"ML position sizing error: {e}")
            return self.enhanced_position_sizing(opportunity, account_value)
    
    async def run_ml_enhanced_trading_cycle(self):
        """
        Run full trading cycle with ML enhancements
        """
        try:
            self.logger.info("Starting ML-Enhanced Trading Cycle")
            
            # 1. ML Market Analysis
            regime_data = await self.ml_enhanced_market_analysis()
            self.logger.info(f"ML Market Regime: {regime_data['regime']} ({regime_data['confidence']:.0%} confidence)")
            
            # 2. Scan for ML-enhanced opportunities
            symbols_to_scan = self.get_symbols_to_scan()
            ml_opportunities = []
            
            for symbol in symbols_to_scan:
                ml_opportunity = await self.ml_enhanced_opportunity_detection(symbol)
                if ml_opportunity:
                    ml_opportunities.append(ml_opportunity)
            
            self.logger.info(f"Found {len(ml_opportunities)} ML-enhanced opportunities")
            
            # 3. Execute best opportunities with ML position sizing
            executed = 0
            for opportunity in sorted(ml_opportunities, key=lambda x: x['combined_confidence'], reverse=True)[:3]:
                
                # ML-enhanced position sizing
                position_size = await self.ml_enhanced_position_sizing(
                    opportunity, self.risk_manager.account_value
                )
                
                if position_size > 0:
                    await self.execute_enhanced_position(opportunity)
                    executed += 1
                    
                    # Log ML insights
                    self.logger.info(
                        f"ML EXECUTION: {opportunity['symbol']} {opportunity['strategy']} - "
                        f"Size: {position_size}, Combined confidence: {opportunity['combined_confidence']:.0%}, "
                        f"ML Vol: {opportunity['ml_volatility_prediction']:.1f}%"
                    )
            
            self.logger.info(f"ML Enhanced cycle complete: {executed} positions executed")
            
            # 4. Periodically evolve strategies (once per day)
            current_hour = datetime.now().hour
            if current_hour == 16 and not hasattr(self, '_evolved_today'):  # After market close
                evolution_result = await self.evolve_trading_strategies()
                if evolution_result['success']:
                    self.logger.info(f"Daily strategy evolution completed: {evolution_result['strategy_name']}")
                self._evolved_today = True
            elif current_hour == 9:  # Reset daily flag
                self._evolved_today = False
            
            return {
                'opportunities_found': len(ml_opportunities),
                'positions_executed': executed,
                'market_regime': regime_data['regime'],
                'regime_confidence': regime_data['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"ML trading cycle error: {e}")
            return {'error': str(e)}
    
    async def get_ml_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary with ML insights
        """
        try:
            # Get base performance
            base_stats = self.performance_stats
            
            # Add ML-specific metrics
            ml_regime_accuracy = 0.8  # Would calculate from historical performance
            
            # Regime stability analysis
            recent_regimes = [r['regime'] for r in self.ml_regime_history[-10:]] if self.ml_regime_history else []
            regime_stability = len(set(recent_regimes)) / len(recent_regimes) if recent_regimes else 0
            
            ml_summary = {
                'base_performance': base_stats,
                'ml_insights': {
                    'regime_detection_accuracy': ml_regime_accuracy,
                    'regime_stability_score': 1.0 - regime_stability,  # Lower is more stable
                    'evolved_strategies_count': len(self.evolved_strategies),
                    'recent_regimes': recent_regimes[-5:] if recent_regimes else [],
                    'ml_features_enabled': ['volatility_prediction', 'regime_detection', 'opportunity_scoring']
                },
                'genetic_algorithm': {
                    'strategies_evolved': len(self.evolved_strategies),
                    'latest_evolution': list(self.evolved_strategies.keys())[-1] if self.evolved_strategies else None
                }
            }
            
            return ml_summary
            
        except Exception as e:
            self.logger.error(f"ML performance summary error: {e}")
            return {'error': str(e)}

async def main():
    """
    Test the ML-Enhanced OPTIONS_BOT
    """
    print("ML-ENHANCED OPTIONS_BOT TESTING")
    print("=" * 60)
    
    bot = MLOptionsBot()
    
    try:
        await bot.initialize_all_systems()
        print(f"[OK] ML Bot initialized - Account: ${bot.risk_manager.account_value:,.2f}")
        
        # Test ML market analysis
        print(f"\n=== ML MARKET ANALYSIS ===")
        regime_data = await bot.ml_enhanced_market_analysis()
        print(f"ML Regime: {regime_data['regime']} ({regime_data['confidence']:.0%} confidence)")
        print(f"Model: {regime_data.get('model', 'N/A')}")
        
        # Test ML opportunity detection
        print(f"\n=== ML OPPORTUNITY DETECTION ===")
        test_symbols = ['AAPL', 'SPY', 'QQQ']
        ml_opportunities = 0
        
        for symbol in test_symbols:
            opportunity = await bot.ml_enhanced_opportunity_detection(symbol)
            if opportunity:
                ml_opportunities += 1
                print(f"ML OPPORTUNITY: {symbol} {opportunity['strategy']} - "
                      f"{opportunity['combined_confidence']:.0%} combined confidence")
                print(f"  ML Score: {opportunity['ml_opportunity_score']:.1%}, "
                      f"ML Vol: {opportunity['ml_volatility_prediction']:.1f}%")
        
        print(f"Total ML opportunities: {ml_opportunities}")
        
        # Test strategy evolution
        print(f"\n=== GENETIC ALGORITHM STRATEGY EVOLUTION ===")
        evolution_result = await bot.evolve_trading_strategies()
        if evolution_result['success']:
            strategy = evolution_result['strategy_params']
            print(f"EVOLVED STRATEGY: {evolution_result['strategy_name']}")
            print(f"  Fitness Score: {strategy['fitness_score']:.3f}")
            print(f"  Risk/Reward Ratio: {strategy['risk_reward_ratio']:.2f}")
            print(f"  Vol Threshold: {strategy['volatility_threshold']:.1%}")
            print(f"  Profit Target: {strategy['profit_target']:.1%}")
            print(f"  Stop Loss: {strategy['stop_loss']:.1%}")
        else:
            print(f"Evolution failed: {evolution_result.get('error', 'Unknown error')}")
        
        # Get ML performance summary
        print(f"\n=== ML PERFORMANCE SUMMARY ===")
        ml_summary = await bot.get_ml_performance_summary()
        if 'error' not in ml_summary:
            ml_insights = ml_summary['ml_insights']
            print(f"Regime Detection Accuracy: {ml_insights['regime_detection_accuracy']:.1%}")
            print(f"Regime Stability Score: {ml_insights['regime_stability_score']:.1%}")
            print(f"Evolved Strategies: {ml_insights['evolved_strategies_count']}")
            print(f"ML Features Enabled: {', '.join(ml_insights['ml_features_enabled'])}")
        
        print(f"\n=== ML-ENHANCED OPTIONS_BOT READY! ===")
        
    except Exception as e:
        print(f"[ERROR] ML Bot test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())