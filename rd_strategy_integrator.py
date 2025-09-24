#!/usr/bin/env python3
"""
R&D STRATEGY INTEGRATOR - CONNECT R&D ENGINE TO HIVE TRADING
============================================================

This script integrates the After Hours R&D Engine with your existing
Hive Trading system, allowing validated strategies from R&D sessions
to be automatically added to your live trading strategy pool.

Features:
- Import validated strategies from R&D engine
- Convert strategies to Hive Trading format
- Add to existing strategy repositories
- Monitor R&D strategy performance
- Automatic deployment when criteria are met
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import os
import logging

# Import your existing Hive Trading components
try:
    from after_hours_rd_engine import AfterHoursRDEngine, StrategyRepository
    RD_ENGINE_AVAILABLE = True
except ImportError:
    RD_ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiveTradingStrategyAdapter:
    """Convert R&D strategies to Hive Trading format"""
    
    def __init__(self):
        self.hive_strategies_file = "hive_trading_strategies.json"
        self.deployment_log_file = "strategy_deployment_log.json"
        
        # Load existing Hive strategies
        self.load_hive_strategies()
        self.load_deployment_log()
    
    def load_hive_strategies(self):
        """Load existing Hive Trading strategies"""
        try:
            with open(self.hive_strategies_file, 'r') as f:
                self.hive_strategies = json.load(f)
        except FileNotFoundError:
            self.hive_strategies = {
                'active_strategies': [],
                'archived_strategies': [],
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'total_strategies': 0,
                    'rd_strategies_count': 0
                }
            }
    
    def load_deployment_log(self):
        """Load deployment history"""
        try:
            with open(self.deployment_log_file, 'r') as f:
                self.deployment_log = json.load(f)
        except FileNotFoundError:
            self.deployment_log = []
    
    def convert_rd_strategy_to_hive_format(self, rd_strategy: Dict) -> Dict:
        """Convert R&D strategy format to Hive Trading format"""
        
        strategy = rd_strategy['config']
        backtest = rd_strategy['backtest_results']
        risk_analysis = rd_strategy.get('risk_analysis', {})
        
        # Convert to Hive Trading format
        hive_strategy = {
            'strategy_id': f"RD_{strategy['name']}_{datetime.now().strftime('%Y%m%d')}",
            'name': strategy['name'],
            'source': 'R&D_ENGINE',
            'type': strategy.get('type', 'factor_based'),
            'symbols': strategy.get('symbols', []),
            'parameters': {
                'rebalance_frequency': strategy.get('rebalance_frequency', 'weekly'),
                'position_sizing': strategy.get('position_sizing', 'equal_weight'),
                'factors': strategy.get('factors', [])
            },
            'performance_metrics': {
                'sharpe_ratio': backtest.get('sharpe_ratio', 0),
                'annual_return': backtest.get('annual_return', 0),
                'max_drawdown': backtest.get('max_drawdown', 0),
                'win_rate': backtest.get('win_rate', 0),
                'profit_factor': backtest.get('profit_factor', 0),
                'num_trades': backtest.get('num_trades', 0)
            },
            'risk_metrics': {
                'annual_volatility': backtest.get('annual_volatility', 0),
                'var_95': risk_analysis.get('var_95', 0),
                'beta': risk_analysis.get('factor_exposures', {}).get('market_beta', 1.0),
                'predicted_volatility': risk_analysis.get('predicted_volatility', 0.2)
            },
            'deployment_status': 'validated',
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'deployment_ready': rd_strategy.get('deployment_ready', False),
            'active': False,  # Starts inactive until manually approved
            'allocation': 0.0  # No allocation until approved
        }
        
        return hive_strategy
    
    def assess_strategy_quality(self, hive_strategy: Dict) -> Dict:
        """Assess strategy quality and provide deployment recommendations"""
        
        metrics = hive_strategy['performance_metrics']
        risk_metrics = hive_strategy['risk_metrics']
        
        # Quality assessment criteria
        quality_scores = {
            'sharpe_score': min(100, max(0, metrics.get('sharpe_ratio', 0) * 50)),  # 2.0 Sharpe = 100
            'return_score': min(100, max(0, metrics.get('annual_return', 0) * 500)),  # 20% return = 100
            'drawdown_score': min(100, max(0, (0.3 - abs(metrics.get('max_drawdown', -0.3))) * 333)),  # <30% DD
            'win_rate_score': min(100, max(0, (metrics.get('win_rate', 0) - 0.3) * 142))  # >30% win rate
        }
        
        overall_score = np.mean(list(quality_scores.values()))
        
        # Deployment recommendation
        if overall_score >= 75:
            recommendation = "RECOMMENDED"
            suggested_allocation = min(0.1, overall_score / 1000)  # Max 10%
        elif overall_score >= 60:
            recommendation = "CONDITIONAL"
            suggested_allocation = min(0.05, overall_score / 2000)  # Max 5%
        elif overall_score >= 40:
            recommendation = "MONITOR"
            suggested_allocation = 0.01  # 1% for monitoring
        else:
            recommendation = "REJECT"
            suggested_allocation = 0.0
        
        assessment = {
            'overall_score': overall_score,
            'quality_scores': quality_scores,
            'recommendation': recommendation,
            'suggested_allocation': suggested_allocation,
            'risk_level': 'LOW' if risk_metrics.get('annual_volatility', 0.2) < 0.15 else 'MEDIUM' if risk_metrics.get('annual_volatility', 0.2) < 0.25 else 'HIGH',
            'assessment_date': datetime.now().isoformat()
        }
        
        return assessment
    
    def add_rd_strategy_to_hive(self, rd_strategy: Dict) -> str:
        """Add R&D strategy to Hive Trading system"""
        
        # Convert to Hive format
        hive_strategy = self.convert_rd_strategy_to_hive_format(rd_strategy)
        
        # Assess quality
        assessment = self.assess_strategy_quality(hive_strategy)
        hive_strategy['quality_assessment'] = assessment
        
        # Add to strategies
        self.hive_strategies['active_strategies'].append(hive_strategy)
        self.hive_strategies['metadata']['total_strategies'] += 1
        self.hive_strategies['metadata']['rd_strategies_count'] += 1
        self.hive_strategies['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Log deployment
        deployment_entry = {
            'strategy_id': hive_strategy['strategy_id'],
            'strategy_name': hive_strategy['name'],
            'deployment_date': datetime.now().isoformat(),
            'source': 'R&D_ENGINE',
            'recommendation': assessment['recommendation'],
            'suggested_allocation': assessment['suggested_allocation'],
            'overall_score': assessment['overall_score']
        }
        self.deployment_log.append(deployment_entry)
        
        # Save files
        self.save_hive_strategies()
        self.save_deployment_log()
        
        logger.info(f"Added R&D strategy '{hive_strategy['name']}' to Hive Trading system")
        logger.info(f"Quality score: {assessment['overall_score']:.1f}, Recommendation: {assessment['recommendation']}")
        
        return hive_strategy['strategy_id']
    
    def save_hive_strategies(self):
        """Save Hive strategies to file"""
        with open(self.hive_strategies_file, 'w') as f:
            json.dump(self.hive_strategies, f, indent=2)
    
    def save_deployment_log(self):
        """Save deployment log"""
        with open(self.deployment_log_file, 'w') as f:
            json.dump(self.deployment_log, f, indent=2)
    
    def get_recommended_strategies(self) -> List[Dict]:
        """Get strategies recommended for deployment"""
        recommended = []
        for strategy in self.hive_strategies['active_strategies']:
            if (strategy.get('quality_assessment', {}).get('recommendation') == 'RECOMMENDED' 
                and not strategy.get('active', False)):
                recommended.append(strategy)
        return recommended
    
    def activate_strategy(self, strategy_id: str, allocation: float) -> bool:
        """Activate a strategy with specified allocation"""
        for strategy in self.hive_strategies['active_strategies']:
            if strategy['strategy_id'] == strategy_id:
                strategy['active'] = True
                strategy['allocation'] = allocation
                strategy['activation_date'] = datetime.now().isoformat()
                
                self.save_hive_strategies()
                logger.info(f"Activated strategy {strategy_id} with {allocation:.1%} allocation")
                return True
        
        logger.error(f"Strategy {strategy_id} not found")
        return False

class RDStrategyMonitor:
    """Monitor R&D strategy performance and integration"""
    
    def __init__(self):
        self.adapter = HiveTradingStrategyAdapter()
        if RD_ENGINE_AVAILABLE:
            self.rd_repo = StrategyRepository()
        else:
            self.rd_repo = None
    
    def sync_rd_strategies(self) -> Dict:
        """Sync new R&D strategies to Hive Trading system"""
        
        if not self.rd_repo:
            logger.warning("R&D engine not available")
            return {'synced': 0, 'error': 'R&D engine not available'}
        
        # Get deployable R&D strategies
        deployable_rd_strategies = self.rd_repo.get_deployable_strategies()
        
        # Get existing Hive strategy IDs to avoid duplicates
        existing_ids = set()
        for strategy in self.adapter.hive_strategies['active_strategies']:
            if strategy.get('source') == 'R&D_ENGINE':
                existing_ids.add(strategy['name'])  # Use name as identifier
        
        synced_count = 0
        sync_results = []
        
        for rd_strategy in deployable_rd_strategies:
            strategy_name = rd_strategy['config']['name']
            
            # Skip if already synced
            if strategy_name in existing_ids:
                continue
            
            try:
                strategy_id = self.adapter.add_rd_strategy_to_hive(rd_strategy)
                synced_count += 1
                
                sync_results.append({
                    'strategy_name': strategy_name,
                    'strategy_id': strategy_id,
                    'status': 'synced'
                })
                
            except Exception as e:
                logger.error(f"Failed to sync strategy {strategy_name}: {e}")
                sync_results.append({
                    'strategy_name': strategy_name,
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"Synced {synced_count} new R&D strategies to Hive Trading")
        
        return {
            'synced': synced_count,
            'total_rd_strategies': len(deployable_rd_strategies),
            'results': sync_results
        }
    
    def generate_integration_report(self) -> Dict:
        """Generate comprehensive integration report"""
        
        report = {
            'report_date': datetime.now().isoformat(),
            'rd_engine_status': RD_ENGINE_AVAILABLE,
            'total_hive_strategies': len(self.adapter.hive_strategies['active_strategies']),
            'rd_strategies_in_hive': len([s for s in self.adapter.hive_strategies['active_strategies'] 
                                         if s.get('source') == 'R&D_ENGINE']),
            'recommended_strategies': len(self.adapter.get_recommended_strategies()),
            'active_strategies': len([s for s in self.adapter.hive_strategies['active_strategies'] 
                                     if s.get('active', False)]),
            'total_allocation': sum([s.get('allocation', 0) for s in self.adapter.hive_strategies['active_strategies']]),
        }
        
        # Strategy quality distribution
        quality_scores = []
        for strategy in self.adapter.hive_strategies['active_strategies']:
            if strategy.get('source') == 'R&D_ENGINE':
                score = strategy.get('quality_assessment', {}).get('overall_score', 0)
                quality_scores.append(score)
        
        if quality_scores:
            report['quality_statistics'] = {
                'average_quality_score': np.mean(quality_scores),
                'min_quality_score': np.min(quality_scores),
                'max_quality_score': np.max(quality_scores),
                'quality_std': np.std(quality_scores)
            }
        
        # Recent deployments
        recent_deployments = [d for d in self.adapter.deployment_log 
                             if d.get('deployment_date', '') > 
                             (datetime.now() - pd.Timedelta(days=7)).isoformat()]
        report['recent_deployments'] = len(recent_deployments)
        
        return report

def main():
    """Main integration workflow"""
    
    print("""
R&D STRATEGY INTEGRATOR - CONNECTING R&D TO HIVE TRADING
========================================================

This system integrates validated strategies from the After Hours R&D Engine
into your existing Hive Trading infrastructure.

Starting integration process...
    """)
    
    monitor = RDStrategyMonitor()
    
    # Step 1: Sync new R&D strategies
    print("Step 1: Syncing R&D strategies...")
    sync_results = monitor.sync_rd_strategies()
    print(f"  Synced: {sync_results['synced']} new strategies")
    
    # Step 2: Generate report
    print("Step 2: Generating integration report...")
    report = monitor.generate_integration_report()
    
    print("\n" + "="*60)
    print("INTEGRATION REPORT")
    print("="*60)
    print(f"R&D Engine Available: {'YES' if report['rd_engine_status'] else 'NO'}")
    print(f"Total Hive Strategies: {report['total_hive_strategies']}")
    print(f"R&D Strategies in Hive: {report['rd_strategies_in_hive']}")
    print(f"Recommended for Deployment: {report['recommended_strategies']}")
    print(f"Currently Active: {report['active_strategies']}")
    print(f"Total Allocation: {report['total_allocation']:.1%}")
    print(f"Recent Deployments (7 days): {report['recent_deployments']}")
    
    if 'quality_statistics' in report:
        print(f"\nR&D Strategy Quality Statistics:")
        print(f"  Average Score: {report['quality_statistics']['average_quality_score']:.1f}")
        print(f"  Best Strategy: {report['quality_statistics']['max_quality_score']:.1f}")
        print(f"  Quality Range: {report['quality_statistics']['quality_std']:.1f}")
    
    # Step 3: Show recommendations
    recommended = monitor.adapter.get_recommended_strategies()
    if recommended:
        print(f"\n DEPLOYMENT RECOMMENDATIONS:")
        print("-" * 40)
        for strategy in recommended[:5]:  # Show top 5
            assessment = strategy['quality_assessment']
            print(f"  {strategy['name']}")
            print(f"    Score: {assessment['overall_score']:.1f}")
            print(f"    Suggested Allocation: {assessment['suggested_allocation']:.1%}")
            print(f"    Sharpe Ratio: {strategy['performance_metrics']['sharpe_ratio']:.2f}")
            print()
    
    print(f"Integration complete! Check 'hive_trading_strategies.json' for full strategy list.")

if __name__ == "__main__":
    main()