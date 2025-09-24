#!/usr/bin/env python3
"""
HIVE TRADING R&D ORCHESTRATOR - COMPLETE AUTOMATION
===================================================

This is the master orchestrator that manages your complete R&D workflow:
- Detects market hours automatically
- Runs R&D sessions when markets are closed
- Integrates validated strategies into Hive Trading system
- Monitors performance and manages strategy lifecycle
- Provides comprehensive reporting and alerts

This is the "set it and forget it" solution for continuous strategy development.
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List
import argparse

# Import our R&D components
from after_hours_rd_engine import AfterHoursRDEngine, MarketHoursDetector
from rd_strategy_integrator import RDStrategyMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hive_rd_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HiveRDOrchestrator:
    """Master orchestrator for complete R&D automation"""
    
    def __init__(self):
        self.rd_engine = AfterHoursRDEngine()
        self.market_detector = MarketHoursDetector()
        self.strategy_monitor = RDStrategyMonitor()
        
        self.orchestration_config = {
            'rd_session_frequency': 'every_4_hours',  # How often to run R&D when market closed
            'auto_sync_enabled': True,  # Automatically sync validated strategies
            'auto_deployment_threshold': 70,  # Quality score threshold for auto-deployment
            'max_auto_allocation': 0.02,  # Max allocation for auto-deployed strategies (2%)
            'monitoring_interval': 3600,  # Check system status every hour
            'session_history_retention': 30  # Keep 30 days of session history
        }
        
        self.session_history = self.load_session_history()
        
        logger.info("Hive R&D Orchestrator initialized")
    
    def load_session_history(self) -> List[Dict]:
        """Load R&D session history"""
        try:
            with open('rd_session_history.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_session_history(self):
        """Save session history"""
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.orchestration_config['session_history_retention'])
        
        self.session_history = [
            session for session in self.session_history
            if datetime.fromisoformat(session['start_time']) > cutoff_date
        ]
        
        with open('rd_session_history.json', 'w') as f:
            json.dump(self.session_history, f, indent=2)
    
    async def orchestrated_rd_session(self) -> Dict:
        """Run complete orchestrated R&D session"""
        
        session_start = datetime.now()
        session_id = f"rd_session_{session_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"=== ORCHESTRATED R&D SESSION STARTED: {session_id} ===")
        
        session_results = {
            'session_id': session_id,
            'start_time': session_start.isoformat(),
            'market_status': self.market_detector.get_market_status(),
            'components_run': [],
            'strategies_generated': 0,
            'strategies_validated': 0,
            'strategies_synced': 0,
            'strategies_deployed': 0,
            'errors': [],
            'status': 'running'
        }
        
        try:
            # Step 1: Run R&D Engine
            logger.info("Step 1: Running R&D Engine...")
            rd_status_before = self.rd_engine.get_system_status()
            await self.rd_engine.run_rd_session()
            rd_status_after = self.rd_engine.get_system_status()
            
            strategies_generated = rd_status_after['total_strategies'] - rd_status_before['total_strategies']
            session_results['strategies_generated'] = strategies_generated
            session_results['strategies_validated'] = strategies_generated  # All generated are validated
            session_results['components_run'].append('rd_engine')
            
            # Step 2: Sync strategies to Hive Trading
            if self.orchestration_config['auto_sync_enabled']:
                logger.info("Step 2: Syncing strategies to Hive Trading...")
                sync_results = self.strategy_monitor.sync_rd_strategies()
                session_results['strategies_synced'] = sync_results['synced']
                session_results['components_run'].append('strategy_sync')
                
                # Step 3: Auto-deploy high-quality strategies
                deployed_count = await self.auto_deploy_strategies()
                session_results['strategies_deployed'] = deployed_count
                session_results['components_run'].append('auto_deployment')
            
            session_results['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"R&D session failed: {e}")
            session_results['errors'].append(str(e))
            session_results['status'] = 'failed'
        
        finally:
            session_end = datetime.now()
            session_results['end_time'] = session_end.isoformat()
            session_results['duration_minutes'] = (session_end - session_start).total_seconds() / 60
            
            # Add to history
            self.session_history.append(session_results)
            self.save_session_history()
            
            logger.info(f"=== R&D SESSION {session_results['status'].upper()}: {session_id} ===")
            logger.info(f"Duration: {session_results['duration_minutes']:.1f} minutes")
            logger.info(f"Generated: {session_results['strategies_generated']} strategies")
            logger.info(f"Synced: {session_results['strategies_synced']} strategies")
            logger.info(f"Deployed: {session_results['strategies_deployed']} strategies")
        
        return session_results
    
    async def auto_deploy_strategies(self) -> int:
        """Automatically deploy high-quality strategies"""
        
        recommended = self.strategy_monitor.adapter.get_recommended_strategies()
        deployed_count = 0
        
        for strategy in recommended:
            assessment = strategy.get('quality_assessment', {})
            
            # Check if meets auto-deployment criteria
            if (assessment.get('overall_score', 0) >= self.orchestration_config['auto_deployment_threshold']
                and assessment.get('risk_level') in ['LOW', 'MEDIUM']):
                
                # Calculate allocation (conservative)
                suggested_allocation = min(
                    assessment.get('suggested_allocation', 0.01),
                    self.orchestration_config['max_auto_allocation']
                )
                
                # Deploy strategy
                success = self.strategy_monitor.adapter.activate_strategy(
                    strategy['strategy_id'], 
                    suggested_allocation
                )
                
                if success:
                    deployed_count += 1
                    logger.info(f"Auto-deployed strategy {strategy['name']} with {suggested_allocation:.1%} allocation")
        
        return deployed_count
    
    def schedule_rd_sessions(self):
        """Schedule R&D sessions based on market hours"""
        
        # Schedule R&D sessions when market is closed
        if self.orchestration_config['rd_session_frequency'] == 'every_4_hours':
            # Run every 4 hours, but only when market is closed
            schedule.every(4).hours.do(self.conditional_rd_session)
        elif self.orchestration_config['rd_session_frequency'] == 'daily':
            # Run once daily at market close
            schedule.every().day.at("16:30").do(self.conditional_rd_session)  # 30 min after close
        
        # Schedule system monitoring
        schedule.every(1).hours.do(self.system_health_check)
        
        logger.info("R&D sessions scheduled")
    
    def conditional_rd_session(self):
        """Run R&D session only if market is closed"""
        
        market_status = self.market_detector.get_market_status()
        
        if not market_status['is_open']:
            logger.info("Market closed - triggering R&D session")
            asyncio.run(self.orchestrated_rd_session())
        else:
            logger.info("Market open - skipping R&D session")
    
    def system_health_check(self):
        """Perform system health check"""
        
        try:
            # Check R&D engine status
            rd_status = self.rd_engine.get_system_status()
            
            # Check strategy integration status  
            integration_report = self.strategy_monitor.generate_integration_report()
            
            # Check recent session performance
            recent_sessions = [s for s in self.session_history 
                             if datetime.fromisoformat(s['start_time']) > 
                             datetime.now() - timedelta(hours=24)]
            
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'rd_engine_healthy': rd_status['system_ready'],
                'total_strategies': rd_status['total_strategies'],
                'deployable_strategies': rd_status['deployable_strategies'],
                'recent_sessions_24h': len(recent_sessions),
                'failed_sessions_24h': len([s for s in recent_sessions if s['status'] == 'failed']),
                'integration_healthy': integration_report['rd_engine_status'],
                'active_allocations': integration_report['total_allocation']
            }
            
            # Log health status
            if health_report['failed_sessions_24h'] > 0:
                logger.warning(f"Health check: {health_report['failed_sessions_24h']} failed sessions in 24h")
            else:
                logger.info(f"Health check: System healthy, {health_report['total_strategies']} total strategies")
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'timestamp': datetime.now().isoformat(), 'status': 'unhealthy', 'error': str(e)}
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status"""
        
        # R&D Engine status
        rd_status = self.rd_engine.get_system_status()
        
        # Integration status
        integration_report = self.strategy_monitor.generate_integration_report()
        
        # Recent session statistics
        recent_sessions = [s for s in self.session_history 
                          if datetime.fromisoformat(s['start_time']) > 
                          datetime.now() - timedelta(days=7)]
        
        session_stats = {
            'total_sessions_7d': len(recent_sessions),
            'successful_sessions': len([s for s in recent_sessions if s['status'] == 'completed']),
            'total_strategies_generated_7d': sum([s.get('strategies_generated', 0) for s in recent_sessions]),
            'total_strategies_deployed_7d': sum([s.get('strategies_deployed', 0) for s in recent_sessions]),
            'avg_session_duration': np.mean([s.get('duration_minutes', 0) for s in recent_sessions]) if recent_sessions else 0
        }
        
        comprehensive_status = {
            'orchestrator_status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'market_status': rd_status['market_status'],
            'rd_engine': rd_status,
            'integration': integration_report,
            'session_statistics': session_stats,
            'config': self.orchestration_config,
            'next_scheduled_check': 'Every hour',
            'system_uptime': 'Running'
        }
        
        return comprehensive_status
    
    async def run_continuous_orchestration(self):
        """Run continuous orchestration (main loop)"""
        
        logger.info("Starting continuous R&D orchestration...")
        logger.info("System will automatically:")
        logger.info("- Run R&D sessions when market is closed")
        logger.info("- Sync validated strategies to Hive Trading")
        logger.info("- Auto-deploy high-quality strategies")
        logger.info("- Monitor system health continuously")
        
        # Schedule tasks
        self.schedule_rd_sessions()
        
        # Main loop
        while True:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a minute
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Orchestration stopped by user")
                break
            except Exception as e:
                logger.error(f"Orchestration error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main entry point with command line options"""
    
    parser = argparse.ArgumentParser(description='Hive Trading R&D Orchestrator')
    parser.add_argument('--mode', choices=['single', 'continuous', 'status'], default='status',
                       help='Run mode: single session, continuous, or show status')
    parser.add_argument('--force', action='store_true',
                       help='Force R&D session even if market is open')
    
    args = parser.parse_args()
    
    orchestrator = HiveRDOrchestrator()
    
    if args.mode == 'single':
        print("Running single R&D session...")
        if args.force or not orchestrator.market_detector.is_market_open():
            asyncio.run(orchestrator.orchestrated_rd_session())
        else:
            print("Market is open - use --force to run anyway")
            
    elif args.mode == 'continuous':
        print("Starting continuous R&D orchestration...")
        asyncio.run(orchestrator.run_continuous_orchestration())
        
    else:  # status
        print("\n" + "="*70)
        print("HIVE TRADING R&D ORCHESTRATOR - SYSTEM STATUS")
        print("="*70)
        
        status = orchestrator.get_comprehensive_status()
        
        print(f"Orchestrator Status: {status['orchestrator_status'].upper()}")
        print(f"Current Time: {status['timestamp']}")
        print(f"Market Session: {status['market_status']['market_session']}")
        
        print(f"\nR&D Engine:")
        print(f"  Total Strategies: {status['rd_engine']['total_strategies']}")
        print(f"  Deployable: {status['rd_engine']['deployable_strategies']}")
        print(f"  Components Ready: {sum(status['rd_engine']['rd_components'].values())}/4")
        
        print(f"\nIntegration:")
        print(f"  Hive Strategies: {status['integration']['total_hive_strategies']}")
        print(f"  Active Allocation: {status['integration']['total_allocation']:.1%}")
        print(f"  Recent Deployments: {status['integration']['recent_deployments']}")
        
        print(f"\nSession Statistics (7 days):")
        print(f"  Total Sessions: {status['session_statistics']['total_sessions_7d']}")
        print(f"  Successful: {status['session_statistics']['successful_sessions']}")
        print(f"  Strategies Generated: {status['session_statistics']['total_strategies_generated_7d']}")
        print(f"  Strategies Deployed: {status['session_statistics']['total_strategies_deployed_7d']}")
        
        print(f"\nConfiguration:")
        print(f"  R&D Frequency: {status['config']['rd_session_frequency']}")
        print(f"  Auto Sync: {'Enabled' if status['config']['auto_sync_enabled'] else 'Disabled'}")
        print(f"  Auto Deploy Threshold: {status['config']['auto_deployment_threshold']}")
        print(f"  Max Auto Allocation: {status['config']['max_auto_allocation']:.1%}")
        
        print(f"\nTo run continuous orchestration: python hive_rd_orchestrator.py --mode continuous")
        print(f"To run single session: python hive_rd_orchestrator.py --mode single")

if __name__ == "__main__":
    import numpy as np  # For session statistics
    main()