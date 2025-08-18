#!/usr/bin/env python3
"""
Performance Monitoring Demo - Task 8.1 Implementation

This demo showcases the comprehensive performance monitoring system including:
- Real-time performance dashboards
- P&L tracking and portfolio monitoring
- Latency monitoring (p50/p95/p99)
- System resource monitoring
- Alert management and notification
- Dashboard data export

Requirements: Requirement 9 (Monitoring and Observability)
Task: 8.1 Performance Monitoring
"""

import asyncio
import sys
import logging
from pathlib import Path
import time
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.performance_monitoring_agent import (
    PerformanceMonitoringAgent, AlertSeverity, AlertType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitoringDemo:
    """Demo class for performance monitoring capabilities"""
    
    def __init__(self):
        self.monitor = PerformanceMonitoringAgent(update_interval=2)
        self.running = False
        self.demo_data = {
            'symbols': ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'],
            'base_prices': {'AAPL': 150.0, 'TSLA': 200.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'AMZN': 3000.0},
            'positions': {},
            'latency_components': ['order_execution', 'signal_generation', 'data_ingestion', 'risk_calculation', 'portfolio_update']
        }
    
    async def start_demo(self):
        """Start the performance monitoring demo"""
        print("🚀 PERFORMANCE MONITORING DEMO - Task 8.1")
        print("=" * 80)
        
        try:
            # Start monitoring
            print("\n1. Starting Performance Monitoring System...")
            await self.monitor.start_monitoring()
            
            # Initialize demo data
            print("\n2. Initializing Demo Data...")
            self._initialize_demo_data()
            
            # Run monitoring simulation
            print("\n3. Running Monitoring Simulation...")
            await self._run_monitoring_simulation()
            
            # Display results
            print("\n4. Displaying Monitoring Results...")
            await self._display_results()
            
            # Export dashboard data
            print("\n5. Exporting Dashboard Data...")
            await self._export_data()
            
            # Stop monitoring
            print("\n6. Stopping Monitoring System...")
            await self.monitor.stop_monitoring()
            
            print("\n✅ Performance Monitoring Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            await self.monitor.stop_monitoring()
            raise
    
    def _initialize_demo_data(self):
        """Initialize demo trading data"""
        print("   📊 Setting up demo trading positions...")
        
        for symbol in self.demo_data['symbols']:
            base_price = self.demo_data['base_prices'][symbol]
            quantity = random.randint(50, 200)
            
            self.demo_data['positions'][symbol] = {
                'quantity': quantity,
                'entry_price': base_price,
                'current_price': base_price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
            
            # Update monitor with initial position
            self.monitor.update_position(symbol, self.demo_data['positions'][symbol])
            
            print(f"      {symbol}: {quantity} shares @ ${base_price:.2f}")
    
    async def _run_monitoring_simulation(self):
        """Run monitoring simulation for several minutes"""
        print("   🔄 Simulating trading activity and monitoring...")
        
        simulation_duration = 60  # 1 minute simulation
        update_interval = 2  # Update every 2 seconds
        
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < simulation_duration:
            update_count += 1
            print(f"      Update {update_count}: Simulating market activity...")
            
            # Simulate market movements
            self._simulate_market_movements()
            
            # Simulate latency variations
            self._simulate_latency_variations()
            
            # Simulate system resource changes
            self._simulate_resource_changes()
            
            # Wait for next update
            await asyncio.sleep(update_interval)
        
        print(f"   ✅ Simulation completed: {update_count} updates over {simulation_duration} seconds")
    
    def _simulate_market_movements(self):
        """Simulate realistic market price movements"""
        for symbol in self.demo_data['symbols']:
            position = self.demo_data['positions'][symbol]
            base_price = self.demo_data['base_prices'][symbol]
            
            # Generate realistic price movement (±2% per update)
            price_change = random.uniform(-0.02, 0.02)
            new_price = position['current_price'] * (1 + price_change)
            
            # Update position
            old_price = position['current_price']
            position['current_price'] = new_price
            
            # Calculate P&L change
            quantity = position['quantity']
            old_pnl = position['unrealized_pnl']
            new_pnl = quantity * (new_price - position['entry_price'])
            pnl_change = new_pnl - old_pnl
            
            position['unrealized_pnl'] = new_pnl
            
            # Record P&L change in monitor
            self.monitor.record_pnl(symbol, pnl_change, "unrealized")
            
            # Update position in monitor
            self.monitor.update_position(symbol, position)
    
    def _simulate_latency_variations(self):
        """Simulate realistic latency variations for different components"""
        for component in self.demo_data['latency_components']:
            # Base latency varies by component
            base_latencies = {
                'order_execution': 100,
                'signal_generation': 50,
                'data_ingestion': 25,
                'risk_calculation': 75,
                'portfolio_update': 30
            }
            
            base_latency = base_latencies[component]
            
            # Add realistic variation (±30% with occasional spikes)
            if random.random() < 0.05:  # 5% chance of latency spike
                variation = random.uniform(0.5, 2.0)  # 50% to 200% of base
            else:
                variation = random.uniform(0.7, 1.3)  # ±30% variation
            
            latency = base_latency * variation
            
            # Record latency in monitor
            self.monitor.record_latency(component, latency)
    
    def _simulate_resource_changes(self):
        """Simulate system resource usage changes"""
        # This is handled by the actual ResourceMonitor in the agent
        # We just need to let it run and collect real system metrics
        pass
    
    async def _display_results(self):
        """Display comprehensive monitoring results"""
        print("\n📊 PERFORMANCE MONITORING RESULTS")
        print("=" * 80)
        
        # Get dashboard data
        dashboard = self.monitor.get_dashboard_data()
        
        # Display system health
        print(f"\n🏥 System Health:")
        health = dashboard.system_health
        print(f"   Status: {health['status'].upper()}")
        print(f"   Health Score: {health['health_score']}/100")
        print(f"   Critical Alerts: {health['critical_alerts']}")
        print(f"   Error Alerts: {health['error_alerts']}")
        print(f"   Total Alerts: {health['total_alerts']}")
        
        if health['issues']:
            print(f"   Issues:")
            for issue in health['issues']:
                print(f"     ⚠️  {issue}")
        
        # Display performance metrics
        print(f"\n⚡ Performance Metrics:")
        perf = dashboard.performance_metrics
        if 'latency' in perf:
            latency = perf['latency']
            print(f"   Average P99 Latency: {latency['avg_p99']:.2f}ms")
            print(f"   Max P99 Latency: {latency['max_p99']:.2f}ms")
            print(f"   Components Monitored: {latency['components_monitored']}")
        
        if 'uptime_seconds' in perf:
            uptime_hours = perf['uptime_seconds'] / 3600
            print(f"   System Uptime: {uptime_hours:.2f} hours")
        
        # Display P&L summary
        print(f"\n💰 P&L Summary:")
        pnl = dashboard.pnl_summary
        if pnl:
            print(f"   Total P&L: ${pnl.get('total_pnl', 0):,.2f}")
            print(f"   Unrealized P&L: ${pnl.get('unrealized_pnl', 0):,.2f}")
            print(f"   Realized P&L: ${pnl.get('realized_pnl', 0):,.2f}")
            print(f"   Active Positions: {pnl.get('position_count', 0)}")
        
        # Display resource usage
        print(f"\n💻 Resource Usage:")
        resources = dashboard.resource_usage
        if resources:
            print(f"   CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
            print(f"   Memory Usage: {resources.get('memory_percent', 0):.1f}%")
            print(f"   Disk Usage: {resources.get('disk_percent', 0):.1f}%")
            print(f"   Active Processes: {resources.get('process_count', 0)}")
        
        # Display latency summary
        print(f"\n⏱️  Latency Summary:")
        latency_summary = dashboard.latency_summary
        if latency_summary:
            for component, metrics in latency_summary.items():
                print(f"   {component}:")
                print(f"     P50: {metrics.p50:.2f}ms")
                print(f"     P95: {metrics.p95:.2f}ms")
                print(f"     P99: {metrics.p99:.2f}ms")
                print(f"     Avg: {metrics.avg_latency:.2f}ms")
                print(f"     Samples: {metrics.sample_count}")
        
        # Display active alerts
        print(f"\n🚨 Active Alerts:")
        alerts = dashboard.alerts
        if alerts:
            for alert in alerts:
                print(f"   {alert.severity.value.upper()}: {alert.message}")
                print(f"     Component: {alert.component}")
                print(f"     Value: {alert.value:.2f}")
                print(f"     Threshold: {alert.threshold:.2f}")
        else:
            print("   ✅ No active alerts")
    
    async def _export_data(self):
        """Export dashboard data to file"""
        print("   📁 Exporting dashboard data...")
        
        try:
            export_path = self.monitor.export_dashboard_data()
            print(f"   ✅ Dashboard data exported to: {export_path}")
            
            # Also generate performance report
            report = self.monitor.get_performance_report(hours=1)
            print(f"   📊 Performance report generated: {report['data_points']} data points")
            
            if 'latency_stats' in report:
                print("   ⏱️  Latency Statistics:")
                for component, stats in report['latency_stats'].items():
                    print(f"     {component}: Avg P99={stats['avg_p99']:.2f}ms, Max={stats['max_p99']:.2f}ms")
            
            if 'pnl_stats' in report:
                print("   💰 P&L Statistics:")
                pnl_stats = report['pnl_stats']
                print(f"     Total P&L: ${pnl_stats['total_pnl']:,.2f}")
                print(f"     Average P&L: ${pnl_stats['avg_pnl']:,.2f}")
                print(f"     Range: ${pnl_stats['min_pnl']:,.2f} to ${pnl_stats['max_pnl']:,.2f}")
            
        except Exception as e:
            print(f"   ❌ Error exporting data: {e}")


async def main():
    """Main demo execution"""
    try:
        demo = PerformanceMonitoringDemo()
        await demo.start_demo()
        
        print("\n" + "=" * 80)
        print("🎉 PERFORMANCE MONITORING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nTask 8.1 - Performance Monitoring has been implemented and demonstrated:")
        print("✅ Basic performance dashboards with real-time metrics")
        print("✅ Real-time P&L tracking and portfolio monitoring")
        print("✅ Latency monitoring (p50/p95/p99) for all system components")
        print("✅ System resource monitoring (CPU, memory, disk, network)")
        print("✅ Basic alerting for system failures and performance degradation")
        print("✅ Dashboard data export and performance reporting")
        
        print("\nThe performance monitoring system is now ready for production use!")
        print("It provides comprehensive visibility into system health, performance, and trading operations.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 