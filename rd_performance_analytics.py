#!/usr/bin/env python3
"""
R&D PERFORMANCE ANALYTICS & REPORTING SYSTEM
===========================================

Advanced analytics system for monitoring and analyzing the performance
of the R&D engine, including strategy generation quality, deployment
success rates, and comprehensive performance attribution.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

class RDPerformanceAnalyzer:
    """Comprehensive performance analysis for R&D system"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.performance_metrics = {}
        self.benchmarks = {
            'spy_returns': None,
            'risk_free_rate': 0.03,  # 3% risk-free rate
            'market_volatility': 0.16  # Historical S&P 500 vol
        }
        
        # Load historical data if available
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load historical performance data"""
        try:
            # Load R&D session history
            with open('rd_session_history.json', 'r') as f:
                self.session_history = json.load(f)
            
            # Load strategy repository
            with open('validated_strategies.json', 'r') as f:
                self.strategy_data = json.load(f)
            
            # Load Hive trading strategies
            with open('hive_trading_strategies.json', 'r') as f:
                self.hive_strategies = json.load(f)
            
            logger.info("Performance data loaded successfully")
            
        except FileNotFoundError as e:
            logger.warning(f"Performance data file not found: {e}")
            self.session_history = []
            self.strategy_data = []
            self.hive_strategies = {'active_strategies': []}
    
    def analyze_rd_generation_performance(self) -> Dict:
        """Analyze R&D strategy generation performance"""
        
        if not self.session_history:
            return {'error': 'No session history available'}
        
        # Convert to DataFrame for analysis
        sessions_df = pd.DataFrame(self.session_history)
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
        sessions_df['duration_hours'] = sessions_df['duration_minutes'] / 60
        
        # Generation metrics
        total_sessions = len(sessions_df)
        successful_sessions = len(sessions_df[sessions_df['status'] == 'completed'])
        total_strategies_generated = sessions_df['strategies_generated'].sum()
        total_strategies_deployed = sessions_df['strategies_deployed'].sum()
        
        # Success rates
        session_success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
        deployment_rate = total_strategies_deployed / total_strategies_generated if total_strategies_generated > 0 else 0
        
        # Efficiency metrics
        avg_strategies_per_session = sessions_df['strategies_generated'].mean()
        avg_session_duration = sessions_df['duration_hours'].mean()
        strategies_per_hour = avg_strategies_per_session / avg_session_duration if avg_session_duration > 0 else 0
        
        # Time series analysis
        sessions_df['date'] = sessions_df['start_time'].dt.date
        daily_generation = sessions_df.groupby('date')['strategies_generated'].sum()
        
        generation_analysis = {
            'summary': {
                'total_sessions': total_sessions,
                'successful_sessions': successful_sessions,
                'session_success_rate': session_success_rate,
                'total_strategies_generated': int(total_strategies_generated),
                'total_strategies_deployed': int(total_strategies_deployed),
                'deployment_rate': deployment_rate
            },
            'efficiency': {
                'avg_strategies_per_session': avg_strategies_per_session,
                'avg_session_duration_hours': avg_session_duration,
                'strategies_per_hour': strategies_per_hour,
                'peak_generation_day': daily_generation.idxmax() if len(daily_generation) > 0 else None,
                'peak_generation_count': daily_generation.max() if len(daily_generation) > 0 else 0
            },
            'trends': {
                'generation_trend': self._calculate_trend(daily_generation.values) if len(daily_generation) > 1 else 0,
                'recent_7d_avg': daily_generation.tail(7).mean() if len(daily_generation) >= 7 else daily_generation.mean(),
                'volatility': daily_generation.std() if len(daily_generation) > 1 else 0
            }
        }
        
        return generation_analysis
    
    def analyze_strategy_quality_evolution(self) -> Dict:
        """Analyze how strategy quality evolves over time"""
        
        if not self.strategy_data:
            return {'error': 'No strategy data available'}
        
        # Extract quality metrics over time
        quality_data = []
        for strategy in self.strategy_data:
            backtest = strategy.get('backtest_results', {})
            quality_data.append({
                'validation_date': strategy.get('validation_date'),
                'sharpe_ratio': backtest.get('sharpe_ratio', 0),
                'max_drawdown': abs(backtest.get('max_drawdown', 0)),
                'total_return': backtest.get('total_return', 0),
                'win_rate': backtest.get('win_rate', 0),
                'deployment_ready': strategy.get('deployment_ready', False)
            })
        
        quality_df = pd.DataFrame(quality_data)
        if quality_df.empty:
            return {'error': 'No quality data available'}
        
        quality_df['validation_date'] = pd.to_datetime(quality_df['validation_date'])
        quality_df = quality_df.sort_values('validation_date')
        
        # Quality evolution metrics
        quality_evolution = {
            'overall_statistics': {
                'avg_sharpe_ratio': float(quality_df['sharpe_ratio'].mean()),
                'avg_max_drawdown': float(quality_df['max_drawdown'].mean()),
                'avg_total_return': float(quality_df['total_return'].mean()),
                'avg_win_rate': float(quality_df['win_rate'].mean()),
                'deployment_ready_rate': float(quality_df['deployment_ready'].mean())
            },
            'quality_distribution': {
                'sharpe_ratio_quartiles': {
                    'q25': float(quality_df['sharpe_ratio'].quantile(0.25)),
                    'q50': float(quality_df['sharpe_ratio'].quantile(0.5)),
                    'q75': float(quality_df['sharpe_ratio'].quantile(0.75))
                },
                'high_quality_strategies': len(quality_df[quality_df['sharpe_ratio'] > 1.5]),
                'low_drawdown_strategies': len(quality_df[quality_df['max_drawdown'] < 0.1]),
                'consistent_strategies': len(quality_df[quality_df['win_rate'] > 0.6])
            },
            'temporal_trends': self._analyze_quality_trends(quality_df)
        }
        
        return quality_evolution
    
    def analyze_deployment_success(self) -> Dict:
        """Analyze deployment success and performance"""
        
        hive_strategies = self.hive_strategies.get('active_strategies', [])
        if not hive_strategies:
            return {'error': 'No deployed strategies available'}
        
        # Deployment metrics
        deployed_count = len([s for s in hive_strategies if s.get('active', False)])
        total_allocation = sum([s.get('allocation', 0) for s in hive_strategies])
        
        # Quality analysis of deployed strategies
        deployed_strategies = [s for s in hive_strategies if s.get('active', False)]
        quality_scores = [s.get('quality_assessment', {}).get('overall_score', 0) for s in deployed_strategies]
        
        # Risk analysis
        risk_levels = [s.get('quality_assessment', {}).get('risk_level', 'UNKNOWN') for s in hive_strategies]
        risk_distribution = pd.Series(risk_levels).value_counts().to_dict()
        
        deployment_analysis = {
            'deployment_summary': {
                'total_strategies_in_hive': len(hive_strategies),
                'deployed_strategies': deployed_count,
                'deployment_rate': deployed_count / len(hive_strategies) if hive_strategies else 0,
                'total_allocation': total_allocation,
                'avg_allocation_per_strategy': total_allocation / deployed_count if deployed_count > 0 else 0
            },
            'quality_analysis': {
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
                'min_quality_score': np.min(quality_scores) if quality_scores else 0,
                'max_quality_score': np.max(quality_scores) if quality_scores else 0,
                'quality_std': np.std(quality_scores) if quality_scores else 0
            },
            'risk_distribution': risk_distribution,
            'recommendations': self._generate_deployment_recommendations(hive_strategies)
        }
        
        return deployment_analysis
    
    def analyze_factor_performance(self) -> Dict:
        """Analyze performance of different factor categories"""
        
        # Group strategies by factor types
        factor_performance = {}
        
        for strategy in self.strategy_data:
            config = strategy.get('config', {})
            factors = config.get('factors', [])
            strategy_type = config.get('type', 'unknown')
            backtest = strategy.get('backtest_results', {})
            
            if not factors:
                factors = [strategy_type]
            
            for factor in factors:
                if factor not in factor_performance:
                    factor_performance[factor] = {
                        'strategies': [],
                        'sharpe_ratios': [],
                        'returns': [],
                        'drawdowns': []
                    }
                
                factor_performance[factor]['strategies'].append(strategy)
                factor_performance[factor]['sharpe_ratios'].append(backtest.get('sharpe_ratio', 0))
                factor_performance[factor]['returns'].append(backtest.get('total_return', 0))
                factor_performance[factor]['drawdowns'].append(abs(backtest.get('max_drawdown', 0)))
        
        # Calculate factor statistics
        factor_analysis = {}
        for factor, data in factor_performance.items():
            if data['sharpe_ratios']:
                factor_analysis[factor] = {
                    'strategy_count': len(data['strategies']),
                    'avg_sharpe_ratio': np.mean(data['sharpe_ratios']),
                    'avg_return': np.mean(data['returns']),
                    'avg_drawdown': np.mean(data['drawdowns']),
                    'success_rate': len([s for s in data['sharpe_ratios'] if s > 1.0]) / len(data['sharpe_ratios']),
                    'consistency': 1 / (1 + np.std(data['sharpe_ratios'])) if np.std(data['sharpe_ratios']) > 0 else 1
                }
        
        # Rank factors by performance
        if factor_analysis:
            factor_rankings = sorted(
                factor_analysis.items(),
                key=lambda x: x[1]['avg_sharpe_ratio'] * x[1]['consistency'],
                reverse=True
            )
            
            best_factors = [f[0] for f in factor_rankings[:3]]
            worst_factors = [f[0] for f in factor_rankings[-3:]]
        else:
            best_factors = []
            worst_factors = []
        
        return {
            'factor_performance': factor_analysis,
            'best_performing_factors': best_factors,
            'worst_performing_factors': worst_factors,
            'factor_diversification': len(factor_analysis),
            'total_factor_strategies': sum([data['strategy_count'] for data in factor_analysis.values()])
        }
    
    def generate_performance_attribution(self) -> Dict:
        """Generate comprehensive performance attribution analysis"""
        
        attribution = {
            'rd_engine_attribution': self._analyze_rd_engine_contribution(),
            'strategy_type_attribution': self._analyze_strategy_type_contribution(),
            'time_period_attribution': self._analyze_time_period_contribution(),
            'risk_factor_attribution': self._analyze_risk_factor_contribution()
        }
        
        return attribution
    
    def _analyze_rd_engine_contribution(self) -> Dict:
        """Analyze contribution of different R&D engine components"""
        
        component_contributions = {
            'qlib_strategies': 0,
            'monte_carlo_validation': 0,
            'gs_quant_risk_analysis': 0,
            'lean_backtesting': 0
        }
        
        # Count strategies by R&D component
        for strategy in self.strategy_data:
            config = strategy.get('config', {})
            if 'Qlib' in config.get('name', ''):
                component_contributions['qlib_strategies'] += 1
            
            if strategy.get('monte_carlo_results'):
                component_contributions['monte_carlo_validation'] += 1
            
            if strategy.get('config', {}).get('risk_analysis'):
                component_contributions['gs_quant_risk_analysis'] += 1
            
            component_contributions['lean_backtesting'] += 1  # All strategies use LEAN
        
        return component_contributions
    
    def _analyze_quality_trends(self, quality_df: pd.DataFrame) -> Dict:
        """Analyze trends in strategy quality over time"""
        
        if len(quality_df) < 2:
            return {'insufficient_data': True}
        
        # Calculate rolling averages
        quality_df['sharpe_ma_7'] = quality_df['sharpe_ratio'].rolling(7, min_periods=1).mean()
        quality_df['drawdown_ma_7'] = quality_df['max_drawdown'].rolling(7, min_periods=1).mean()
        
        # Trend analysis
        x = np.arange(len(quality_df))
        sharpe_trend = LinearRegression().fit(x.reshape(-1, 1), quality_df['sharpe_ratio']).coef_[0]
        drawdown_trend = LinearRegression().fit(x.reshape(-1, 1), quality_df['max_drawdown']).coef_[0]
        
        return {
            'sharpe_trend_slope': sharpe_trend,
            'drawdown_trend_slope': drawdown_trend,
            'quality_improvement': sharpe_trend > 0 and drawdown_trend < 0,
            'recent_avg_sharpe': quality_df['sharpe_ratio'].tail(10).mean(),
            'recent_avg_drawdown': quality_df['max_drawdown'].tail(10).mean(),
            'volatility_sharpe': quality_df['sharpe_ratio'].std(),
            'consistency_score': 1 / (1 + quality_df['sharpe_ratio'].std()) if quality_df['sharpe_ratio'].std() > 0 else 1
        }
    
    def _generate_deployment_recommendations(self, strategies: List[Dict]) -> List[str]:
        """Generate deployment recommendations based on analysis"""
        
        recommendations = []
        
        # Analyze current allocations
        total_allocation = sum([s.get('allocation', 0) for s in strategies])
        active_strategies = [s for s in strategies if s.get('active', False)]
        
        if total_allocation < 0.05:
            recommendations.append("Consider increasing total strategy allocation (currently < 5%)")
        
        if len(active_strategies) < 3:
            recommendations.append("Consider deploying more strategies for diversification")
        
        # Quality-based recommendations
        high_quality_undeployed = [
            s for s in strategies 
            if not s.get('active', False) and 
            s.get('quality_assessment', {}).get('overall_score', 0) > 75
        ]
        
        if high_quality_undeployed:
            recommendations.append(f"Consider deploying {len(high_quality_undeployed)} high-quality strategies")
        
        # Risk-based recommendations
        high_risk_deployed = [
            s for s in active_strategies
            if s.get('quality_assessment', {}).get('risk_level') == 'HIGH'
        ]
        
        if high_risk_deployed:
            recommendations.append(f"Review {len(high_risk_deployed)} high-risk deployed strategies")
        
        return recommendations
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive performance report"""
        
        # Run all analyses
        generation_perf = self.analyze_rd_generation_performance()
        quality_evolution = self.analyze_strategy_quality_evolution()
        deployment_success = self.analyze_deployment_success()
        factor_performance = self.analyze_factor_performance()
        
        report = f"""
HIVE TRADING R&D SYSTEM - PERFORMANCE ANALYTICS REPORT
======================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
- Total Strategies Generated: {generation_perf.get('summary', {}).get('total_strategies_generated', 0)}
- Deployment Rate: {generation_perf.get('summary', {}).get('deployment_rate', 0):.1%}
- Average Strategy Quality: {quality_evolution.get('overall_statistics', {}).get('avg_sharpe_ratio', 0):.2f} Sharpe
- Active Strategies: {deployment_success.get('deployment_summary', {}).get('deployed_strategies', 0)}
- Total Allocation: {deployment_success.get('deployment_summary', {}).get('total_allocation', 0):.1%}

STRATEGY GENERATION PERFORMANCE:
- Session Success Rate: {generation_perf.get('summary', {}).get('session_success_rate', 0):.1%}
- Strategies per Hour: {generation_perf.get('efficiency', {}).get('strategies_per_hour', 0):.1f}
- Generation Trend: {'Increasing' if generation_perf.get('trends', {}).get('generation_trend', 0) > 0 else 'Stable/Decreasing'}
- Recent 7d Average: {generation_perf.get('trends', {}).get('recent_7d_avg', 0):.1f} strategies/day

STRATEGY QUALITY ANALYSIS:
- Average Sharpe Ratio: {quality_evolution.get('overall_statistics', {}).get('avg_sharpe_ratio', 0):.2f}
- Average Max Drawdown: {quality_evolution.get('overall_statistics', {}).get('avg_max_drawdown', 0):.1%}
- Deployment Ready Rate: {quality_evolution.get('overall_statistics', {}).get('deployment_ready_rate', 0):.1%}
- High Quality Strategies (Sharpe > 1.5): {quality_evolution.get('quality_distribution', {}).get('high_quality_strategies', 0)}

FACTOR PERFORMANCE:
- Best Performing Factors: {', '.join(factor_performance.get('best_performing_factors', [])[:3])}
- Factor Diversification: {factor_performance.get('factor_diversification', 0)} unique factors
- Total Factor-Based Strategies: {factor_performance.get('total_factor_strategies', 0)}

DEPLOYMENT SUCCESS:
- Deployment Rate: {deployment_success.get('deployment_summary', {}).get('deployment_rate', 0):.1%}
- Average Quality Score: {deployment_success.get('quality_analysis', {}).get('avg_quality_score', 0):.1f}
- Risk Distribution: {deployment_success.get('risk_distribution', {})}

RECOMMENDATIONS:
"""
        
        recommendations = deployment_success.get('recommendations', [])
        for rec in recommendations:
            report += f"- {rec}\n"
        
        if not recommendations:
            report += "- System operating optimally, no immediate recommendations\n"
        
        report += f"""
SYSTEM HEALTH INDICATORS:
- Strategy Generation: {'[HEALTHY]' if generation_perf.get('summary', {}).get('session_success_rate', 0) > 0.8 else '[NEEDS ATTENTION]'}
- Quality Trend: {'[IMPROVING]' if quality_evolution.get('temporal_trends', {}).get('quality_improvement', False) else '[STABLE]'}
- Deployment Pipeline: {'[ACTIVE]' if deployment_success.get('deployment_summary', {}).get('deployed_strategies', 0) > 0 else '[INACTIVE]'}
- Factor Diversification: {'[GOOD]' if factor_performance.get('factor_diversification', 0) > 5 else '[LIMITED]'}

NEXT STEPS:
1. Monitor strategy performance in live trading
2. Analyze factor performance attribution
3. Optimize underperforming components
4. Consider increasing allocation to high-quality strategies
5. Regular system health checks and parameter tuning
"""
        
        return report
    
    def save_analytics_results(self):
        """Save analytics results to file"""
        
        analytics_results = {
            'generation_analysis': self.analyze_rd_generation_performance(),
            'quality_evolution': self.analyze_strategy_quality_evolution(),
            'deployment_analysis': self.analyze_deployment_success(),
            'factor_analysis': self.analyze_factor_performance(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open('rd_analytics_results.json', 'w') as f:
            json.dump(analytics_results, f, indent=2)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        with open('rd_performance_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Analytics results saved to files")

def main():
    """Run comprehensive performance analysis"""
    
    print("HIVE TRADING R&D PERFORMANCE ANALYTICS")
    print("=" * 50)
    
    analyzer = RDPerformanceAnalyzer()
    
    # Generate and display comprehensive report
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # Save results
    analyzer.save_analytics_results()
    print("\nAnalytics complete! Results saved to:")
    print("- rd_analytics_results.json")
    print("- rd_performance_report.txt")

if __name__ == "__main__":
    main()