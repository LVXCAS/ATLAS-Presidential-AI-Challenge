"""
Risk Manager Agent Demo

This demo showcases the comprehensive risk management capabilities of the Risk Manager Agent:
- Real-time portfolio risk monitoring and VaR calculation
- Dynamic position limits and exposure controls
- Emergency circuit breakers and kill switch functionality
- Correlation risk management and liquidity analysis
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_manager_agent import (
    RiskManagerAgent, RiskLimits, Position, RiskAlertType, RiskAlertSeverity
)
from config.database import get_database_config


class RiskManagerDemo:
    """Demo class for Risk Manager Agent functionality"""
    
    def __init__(self):
        """Initialize the demo"""
        self.db_config = get_database_config()
        
        # Configure conservative risk limits for demo
        self.risk_limits = RiskLimits(
            max_daily_loss_pct=5.0,      # 5% daily loss limit
            max_position_size_pct=10.0,   # 10% max position size
            max_leverage=2.0,             # 2x max leverage
            max_var_95_pct=3.0,          # 3% max VaR
            max_correlation=0.8,          # 80% max correlation
            min_liquidity_days=5,         # 5 days to liquidate
            max_sector_concentration_pct=25.0,  # 25% max sector concentration
            volatility_spike_threshold=2.0      # 2x volatility spike threshold
        )
        
        self.risk_manager = RiskManagerAgent(self.db_config, self.risk_limits)
    
    async def demo_portfolio_risk_monitoring(self):
        """Demonstrate comprehensive portfolio risk monitoring"""
        print("=" * 80)
        print("RISK MANAGER AGENT - PORTFOLIO RISK MONITORING DEMO")
        print("=" * 80)
        
        try:
            print("\n1. PORTFOLIO RISK ASSESSMENT")
            print("-" * 40)
            
            # Monitor portfolio risk
            risk_metrics = await self.risk_manager.monitor_portfolio_risk()
            
            print(f"Portfolio Value:        ${risk_metrics.portfolio_value:,.2f}")
            print(f"Cash Position:          ${risk_metrics.cash:,.2f}")
            print(f"Gross Exposure:         ${risk_metrics.gross_exposure:,.2f}")
            print(f"Net Exposure:           ${risk_metrics.net_exposure:,.2f}")
            print(f"Portfolio Leverage:     {risk_metrics.leverage:.2f}x")
            print(f"Largest Position:       {risk_metrics.max_position_pct:.2f}%")
            print(f"Sector Concentration:   {risk_metrics.sector_concentration:.2f}%")
            
            print(f"\n📊 VALUE AT RISK ANALYSIS:")
            print(f"VaR 95% (1-day):        ${risk_metrics.var_1d_95:,.2f}")
            print(f"VaR 99% (1-day):        ${risk_metrics.var_1d_99:,.2f}")
            print(f"VaR 95% (5-day):        ${risk_metrics.var_5d_95:,.2f}")
            print(f"Expected Shortfall 95%: ${risk_metrics.expected_shortfall_95:,.2f}")
            
            print(f"\n🔗 CORRELATION & LIQUIDITY RISK:")
            print(f"Correlation Risk:       {risk_metrics.correlation_risk:.3f}")
            print(f"Liquidity Risk:         {risk_metrics.liquidity_risk:.2f} days")
            
            # Risk assessment
            risk_score = self._calculate_overall_risk_score(risk_metrics)
            risk_level = self._get_risk_level(risk_score)
            
            print(f"\n🎯 OVERALL RISK ASSESSMENT:")
            print(f"Risk Score:             {risk_score:.2f}/100")
            print(f"Risk Level:             {risk_level}")
            
            return risk_metrics
            
        except Exception as e:
            print(f"❌ Error in portfolio risk monitoring: {e}")
            return None
    
    async def demo_position_limit_checks(self):
        """Demonstrate position limit checking for new orders"""
        print("\n\n2. POSITION LIMIT CHECKING")
        print("-" * 40)
        
        # Test various order scenarios
        test_orders = [
            {
                'name': 'Small Order (Within Limits)',
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.0,
                'expected': True
            },
            {
                'name': 'Large Order (May Exceed Limits)',
                'symbol': 'TSLA',
                'quantity': 1000,
                'price': 250.0,
                'expected': False
            },
            {
                'name': 'Medium Order (Borderline)',
                'symbol': 'GOOGL',
                'quantity': 50,
                'price': 2800.0,
                'expected': True
            }
        ]
        
        for order in test_orders:
            print(f"\n📋 Testing: {order['name']}")
            print(f"   Symbol: {order['symbol']}")
            print(f"   Quantity: {order['quantity']:,}")
            print(f"   Price: ${order['price']:,.2f}")
            print(f"   Order Value: ${order['quantity'] * order['price']:,.2f}")
            
            try:
                result = await self.risk_manager.check_position_limits({
                    'symbol': order['symbol'],
                    'quantity': order['quantity'],
                    'price': order['price']
                })
                
                status = "✅ APPROVED" if result['approved'] else "❌ REJECTED"
                print(f"   Result: {status}")
                print(f"   Reason: {result['reason']}")
                
            except Exception as e:
                print(f"   ❌ Error checking limits: {e}")
    
    async def demo_emergency_stop_functionality(self):
        """Demonstrate emergency stop and circuit breaker functionality"""
        print("\n\n3. EMERGENCY STOP & CIRCUIT BREAKERS")
        print("-" * 40)
        
        print("\n🚨 Testing Emergency Stop Functionality:")
        
        # Check initial state
        print(f"Initial Emergency Stop Status: {'ACTIVE' if self.risk_manager.is_emergency_stop_active() else 'INACTIVE'}")
        
        # Trigger emergency stop
        print("\n⚡ Triggering Manual Emergency Stop...")
        success = self.risk_manager.trigger_emergency_stop("Demo: Testing emergency procedures")
        
        if success:
            print("✅ Emergency stop activated successfully")
            print(f"Emergency Stop Status: {'ACTIVE' if self.risk_manager.is_emergency_stop_active() else 'INACTIVE'}")
            
            # Test order rejection during emergency stop
            print("\n📋 Testing order rejection during emergency stop...")
            test_order = {
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 150.0
            }
            
            result = await self.risk_manager.check_position_limits(test_order)
            print(f"Order Status: {'APPROVED' if result['approved'] else 'REJECTED'}")
            print(f"Reason: {result['reason']}")
            
            # Reset emergency stop
            print("\n🔄 Resetting Emergency Stop...")
            reset_success = self.risk_manager.reset_emergency_stop()
            
            if reset_success:
                print("✅ Emergency stop reset successfully")
                print(f"Emergency Stop Status: {'ACTIVE' if self.risk_manager.is_emergency_stop_active() else 'INACTIVE'}")
            else:
                print("❌ Failed to reset emergency stop")
        else:
            print("❌ Failed to activate emergency stop")
    
    async def demo_risk_scenario_analysis(self):
        """Demonstrate risk analysis under various market scenarios"""
        print("\n\n4. RISK SCENARIO ANALYSIS")
        print("-" * 40)
        
        scenarios = [
            {
                'name': 'Normal Market Conditions',
                'volatility_multiplier': 1.0,
                'correlation_increase': 0.0,
                'description': 'Baseline market conditions'
            },
            {
                'name': 'High Volatility Regime',
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.2,
                'description': 'Market stress with increased volatility'
            },
            {
                'name': 'Crisis Scenario',
                'volatility_multiplier': 4.0,
                'correlation_increase': 0.4,
                'description': 'Extreme market conditions with correlation breakdown'
            }
        ]
        
        print("\n📈 SCENARIO ANALYSIS RESULTS:")
        
        for scenario in scenarios:
            print(f"\n🎭 Scenario: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   Volatility Multiplier: {scenario['volatility_multiplier']}x")
            print(f"   Correlation Increase: +{scenario['correlation_increase']:.1f}")
            
            # Simulate scenario impact (simplified)
            base_var = 2000.0  # Base VaR
            scenario_var = base_var * scenario['volatility_multiplier']
            
            print(f"   Estimated VaR Impact: ${scenario_var:,.2f}")
            
            # Risk assessment for scenario
            if scenario_var > self.risk_limits.max_var_95_pct * 100000 / 100:  # Assuming $100k portfolio
                print("   ⚠️  VaR LIMIT BREACH - Emergency actions recommended")
            else:
                print("   ✅ Within acceptable risk limits")
    
    def demo_risk_alerts_simulation(self):
        """Simulate various risk alert scenarios"""
        print("\n\n5. RISK ALERTS SIMULATION")
        print("-" * 40)
        
        # Simulate different types of risk alerts
        simulated_alerts = [
            {
                'type': RiskAlertType.DAILY_LOSS_LIMIT,
                'severity': RiskAlertSeverity.CRITICAL,
                'description': 'Daily loss of 8.5% exceeds limit of 5.0%',
                'current_value': 8.5,
                'limit_value': 5.0
            },
            {
                'type': RiskAlertType.POSITION_LIMIT,
                'severity': RiskAlertSeverity.HIGH,
                'description': 'AAPL position 15.2% exceeds limit of 10.0%',
                'current_value': 15.2,
                'limit_value': 10.0
            },
            {
                'type': RiskAlertType.LEVERAGE_LIMIT,
                'severity': RiskAlertSeverity.MEDIUM,
                'description': 'Portfolio leverage 2.3x exceeds limit of 2.0x',
                'current_value': 2.3,
                'limit_value': 2.0
            },
            {
                'type': RiskAlertType.CORRELATION_RISK,
                'severity': RiskAlertSeverity.LOW,
                'description': 'Average correlation 0.85 exceeds limit of 0.80',
                'current_value': 0.85,
                'limit_value': 0.80
            }
        ]
        
        print("\n🚨 SIMULATED RISK ALERTS:")
        
        for alert in simulated_alerts:
            severity_icon = {
                RiskAlertSeverity.CRITICAL: "🔴",
                RiskAlertSeverity.HIGH: "🟠",
                RiskAlertSeverity.MEDIUM: "🟡",
                RiskAlertSeverity.LOW: "🟢"
            }
            
            print(f"\n{severity_icon[alert['severity']]} {alert['severity'].value} - {alert['type'].value}")
            print(f"   Description: {alert['description']}")
            print(f"   Current: {alert['current_value']}")
            print(f"   Limit: {alert['limit_value']}")
            
            breach_pct = (alert['current_value'] - alert['limit_value']) / alert['limit_value'] * 100
            print(f"   Breach: {breach_pct:.1f}%")
            
            # Recommend actions based on severity
            if alert['severity'] == RiskAlertSeverity.CRITICAL:
                print("   🚨 Recommended Action: EMERGENCY STOP")
            elif alert['severity'] == RiskAlertSeverity.HIGH:
                print("   ⚠️  Recommended Action: REDUCE POSITIONS")
            elif alert['severity'] == RiskAlertSeverity.MEDIUM:
                print("   📊 Recommended Action: INCREASE MONITORING")
            else:
                print("   ℹ️  Recommended Action: CONTINUE MONITORING")
    
    def _calculate_overall_risk_score(self, risk_metrics) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0
        
        # Leverage component (0-25 points)
        leverage_score = min(25, (risk_metrics.leverage / self.risk_limits.max_leverage) * 25)
        score += leverage_score
        
        # VaR component (0-25 points)
        var_limit = risk_metrics.portfolio_value * self.risk_limits.max_var_95_pct / 100
        var_score = min(25, (risk_metrics.var_1d_95 / var_limit) * 25) if var_limit > 0 else 0
        score += var_score
        
        # Position concentration component (0-25 points)
        concentration_score = min(25, (risk_metrics.max_position_pct / self.risk_limits.max_position_size_pct) * 25)
        score += concentration_score
        
        # Correlation component (0-25 points)
        correlation_score = min(25, (risk_metrics.correlation_risk / self.risk_limits.max_correlation) * 25)
        score += correlation_score
        
        return min(100, score)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description based on score"""
        if risk_score >= 80:
            return "🔴 CRITICAL - Immediate action required"
        elif risk_score >= 60:
            return "🟠 HIGH - Close monitoring needed"
        elif risk_score >= 40:
            return "🟡 MEDIUM - Normal monitoring"
        elif risk_score >= 20:
            return "🟢 LOW - Acceptable risk level"
        else:
            return "🔵 MINIMAL - Very low risk"
    
    async def run_complete_demo(self):
        """Run the complete Risk Manager Agent demo"""
        print("🚀 Starting Risk Manager Agent Demo...")
        print(f"⏰ Demo started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        try:
            # Run all demo components
            await self.demo_portfolio_risk_monitoring()
            await self.demo_position_limit_checks()
            await self.demo_emergency_stop_functionality()
            await self.demo_risk_scenario_analysis()
            self.demo_risk_alerts_simulation()
            
            print("\n" + "=" * 80)
            print("✅ RISK MANAGER AGENT DEMO COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            print("\n📋 DEMO SUMMARY:")
            print("• Portfolio risk monitoring and VaR calculation ✅")
            print("• Position limit checking for new orders ✅")
            print("• Emergency stop and circuit breaker functionality ✅")
            print("• Risk scenario analysis ✅")
            print("• Risk alerts simulation ✅")
            
            print(f"\n⏰ Demo completed at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function"""
    demo = RiskManagerDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())