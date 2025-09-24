"""
REAL-TIME RISK OVERRIDE SYSTEM
==============================
Instant risk management and trading halt capabilities
Monitors all trading activity and can instantly stop execution
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable
import time
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OverrideAction(Enum):
    HALT_ALL = "halt_all"
    HALT_SYMBOL = "halt_symbol"
    REDUCE_POSITIONS = "reduce_positions"
    BLOCK_NEW_ORDERS = "block_new_orders"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    risk_level: RiskLevel
    risk_type: str
    description: str
    affected_symbols: List[str]
    current_value: float
    threshold_value: float
    recommended_action: OverrideAction
    timestamp: datetime
    requires_immediate_action: bool = False

@dataclass
class OverrideCommand:
    """Risk override command"""
    command_id: str
    action: OverrideAction
    symbols: List[str]
    duration_minutes: Optional[int]
    reason: str
    issued_by: str
    timestamp: datetime

class RealTimeRiskOverrideSystem:
    """
    REAL-TIME RISK OVERRIDE SYSTEM
    Monitors and controls trading risk in real-time
    """

    def __init__(self):
        self.logger = logging.getLogger('RiskOverride')

        # Risk monitoring
        self.monitoring_active = False
        self.risk_alerts = []
        self.active_overrides = {}
        self.alert_queue = asyncio.Queue()

        # Risk thresholds
        self.risk_thresholds = {
            "max_daily_loss": 5000.0,
            "max_position_size": 50000.0,
            "max_portfolio_concentration": 0.20,  # 20% max in single position
            "max_drawdown": 0.15,  # 15% max drawdown
            "volatility_threshold": 0.05,  # 5% volatility spike
            "correlation_threshold": 0.8,  # High correlation warning
            "liquidity_threshold": 1000,  # Min daily volume
            "news_sentiment_extreme": 0.7  # Extreme sentiment threshold
        }

        # Override callbacks - systems that need to be notified of overrides
        self.override_callbacks = []

        # Performance tracking
        self.override_history = []
        self.risk_metrics_history = []

        # Emergency contacts and notifications
        self.emergency_notifications = []

        self.logger.info("Real-time Risk Override System initialized")
        self.logger.info("Ready to monitor and control trading risk")

    def register_override_callback(self, callback: Callable):
        """Register a callback to be notified of risk overrides"""
        self.override_callbacks.append(callback)
        self.logger.info(f"Registered override callback: {callback.__name__}")

    def add_emergency_notification(self, notification_method: str, contact: str):
        """Add emergency notification method"""
        self.emergency_notifications.append({
            'method': notification_method,
            'contact': contact,
            'enabled': True
        })

    async def start_risk_monitoring(self):
        """Start real-time risk monitoring"""
        self.monitoring_active = True
        self.logger.info("Starting real-time risk monitoring")

        # Start monitoring tasks
        tasks = [
            self.portfolio_risk_monitor(),
            self.position_risk_monitor(),
            self.market_risk_monitor(),
            self.liquidity_risk_monitor(),
            self.correlation_risk_monitor(),
            self.news_sentiment_monitor(),
            self.alert_processor()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def portfolio_risk_monitor(self):
        """Monitor overall portfolio risk"""
        while self.monitoring_active:
            try:
                # Get current portfolio data (would interface with execution engine)
                portfolio_data = await self.get_current_portfolio()

                # Check daily P&L
                daily_pnl = portfolio_data.get('daily_pnl', 0)
                if daily_pnl < -self.risk_thresholds['max_daily_loss']:
                    await self.generate_risk_alert(
                        risk_type="DAILY_LOSS_LIMIT",
                        description=f"Daily loss of ${abs(daily_pnl):,.2f} exceeds limit of ${self.risk_thresholds['max_daily_loss']:,.2f}",
                        current_value=abs(daily_pnl),
                        threshold_value=self.risk_thresholds['max_daily_loss'],
                        risk_level=RiskLevel.CRITICAL,
                        recommended_action=OverrideAction.HALT_ALL,
                        requires_immediate_action=True
                    )

                # Check portfolio drawdown
                total_value = portfolio_data.get('total_value', 100000)
                peak_value = portfolio_data.get('peak_value', 100000)
                drawdown = (peak_value - total_value) / peak_value

                if drawdown > self.risk_thresholds['max_drawdown']:
                    await self.generate_risk_alert(
                        risk_type="MAX_DRAWDOWN",
                        description=f"Portfolio drawdown of {drawdown:.1%} exceeds limit of {self.risk_thresholds['max_drawdown']:.1%}",
                        current_value=drawdown,
                        threshold_value=self.risk_thresholds['max_drawdown'],
                        risk_level=RiskLevel.HIGH,
                        recommended_action=OverrideAction.HALT_ALL,
                        requires_immediate_action=True
                    )

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Portfolio risk monitor error: {e}")
                await asyncio.sleep(10)

    async def position_risk_monitor(self):
        """Monitor individual position risks"""
        while self.monitoring_active:
            try:
                # Get current positions
                positions = await self.get_current_positions()

                for symbol, position in positions.items():
                    # Check position size limits
                    position_value = abs(position.get('market_value', 0))
                    if position_value > self.risk_thresholds['max_position_size']:
                        await self.generate_risk_alert(
                            risk_type="POSITION_SIZE_LIMIT",
                            description=f"{symbol} position value ${position_value:,.2f} exceeds limit",
                            current_value=position_value,
                            threshold_value=self.risk_thresholds['max_position_size'],
                            risk_level=RiskLevel.HIGH,
                            recommended_action=OverrideAction.HALT_SYMBOL,
                            affected_symbols=[symbol]
                        )

                    # Check concentration risk
                    portfolio_value = await self.get_portfolio_value()
                    concentration = position_value / portfolio_value
                    if concentration > self.risk_thresholds['max_portfolio_concentration']:
                        await self.generate_risk_alert(
                            risk_type="CONCENTRATION_RISK",
                            description=f"{symbol} represents {concentration:.1%} of portfolio",
                            current_value=concentration,
                            threshold_value=self.risk_thresholds['max_portfolio_concentration'],
                            risk_level=RiskLevel.MEDIUM,
                            recommended_action=OverrideAction.REDUCE_POSITIONS,
                            affected_symbols=[symbol]
                        )

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Position risk monitor error: {e}")
                await asyncio.sleep(15)

    async def market_risk_monitor(self):
        """Monitor market-wide risk factors"""
        while self.monitoring_active:
            try:
                # Simulate market data (in real implementation, get from market data engine)
                market_data = await self.get_market_risk_data()

                # Check VIX levels
                vix_level = market_data.get('VIX', 20)
                if vix_level > 30:  # High volatility
                    await self.generate_risk_alert(
                        risk_type="HIGH_VOLATILITY",
                        description=f"VIX at {vix_level:.1f} indicates high market volatility",
                        current_value=vix_level,
                        threshold_value=30,
                        risk_level=RiskLevel.MEDIUM,
                        recommended_action=OverrideAction.BLOCK_NEW_ORDERS
                    )

                # Check for market gaps
                for symbol, data in market_data.items():
                    if symbol != 'VIX':
                        price_change = data.get('price_change_pct', 0)
                        if abs(price_change) > self.risk_thresholds['volatility_threshold']:
                            await self.generate_risk_alert(
                                risk_type="PRICE_GAP",
                                description=f"{symbol} moved {price_change:.2%} - significant price gap detected",
                                current_value=abs(price_change),
                                threshold_value=self.risk_thresholds['volatility_threshold'],
                                risk_level=RiskLevel.MEDIUM,
                                recommended_action=OverrideAction.HALT_SYMBOL,
                                affected_symbols=[symbol]
                            )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Market risk monitor error: {e}")
                await asyncio.sleep(30)

    async def liquidity_risk_monitor(self):
        """Monitor liquidity risk"""
        while self.monitoring_active:
            try:
                # Check liquidity for all positions
                positions = await self.get_current_positions()

                for symbol in positions.keys():
                    # Simulate liquidity check
                    daily_volume = np.random.randint(500, 5000) * 1000  # Simulated volume

                    if daily_volume < self.risk_thresholds['liquidity_threshold']:
                        await self.generate_risk_alert(
                            risk_type="LOW_LIQUIDITY",
                            description=f"{symbol} daily volume {daily_volume:,} below threshold",
                            current_value=daily_volume,
                            threshold_value=self.risk_thresholds['liquidity_threshold'],
                            risk_level=RiskLevel.MEDIUM,
                            recommended_action=OverrideAction.HALT_SYMBOL,
                            affected_symbols=[symbol]
                        )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Liquidity risk monitor error: {e}")
                await asyncio.sleep(60)

    async def correlation_risk_monitor(self):
        """Monitor correlation risk between positions"""
        while self.monitoring_active:
            try:
                # Get positions and check correlations
                positions = await self.get_current_positions()
                symbols = list(positions.keys())

                if len(symbols) > 1:
                    # Simulate correlation analysis
                    for i, symbol1 in enumerate(symbols):
                        for symbol2 in symbols[i+1:]:
                            # Simulate correlation calculation
                            correlation = np.random.uniform(0.3, 0.9)

                            if correlation > self.risk_thresholds['correlation_threshold']:
                                await self.generate_risk_alert(
                                    risk_type="HIGH_CORRELATION",
                                    description=f"High correlation {correlation:.2f} between {symbol1} and {symbol2}",
                                    current_value=correlation,
                                    threshold_value=self.risk_thresholds['correlation_threshold'],
                                    risk_level=RiskLevel.MEDIUM,
                                    recommended_action=OverrideAction.REDUCE_POSITIONS,
                                    affected_symbols=[symbol1, symbol2]
                                )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Correlation risk monitor error: {e}")
                await asyncio.sleep(300)

    async def news_sentiment_monitor(self):
        """Monitor news sentiment for extreme events"""
        while self.monitoring_active:
            try:
                # Simulate news sentiment analysis
                major_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']

                for symbol in major_symbols:
                    # Simulate sentiment score
                    sentiment = np.random.uniform(-0.8, 0.8)

                    if abs(sentiment) > self.risk_thresholds['news_sentiment_extreme']:
                        risk_level = RiskLevel.HIGH if abs(sentiment) > 0.8 else RiskLevel.MEDIUM

                        await self.generate_risk_alert(
                            risk_type="EXTREME_SENTIMENT",
                            description=f"Extreme news sentiment {sentiment:.2f} for {symbol}",
                            current_value=abs(sentiment),
                            threshold_value=self.risk_thresholds['news_sentiment_extreme'],
                            risk_level=risk_level,
                            recommended_action=OverrideAction.HALT_SYMBOL,
                            affected_symbols=[symbol]
                        )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"News sentiment monitor error: {e}")
                await asyncio.sleep(60)

    async def generate_risk_alert(self, risk_type: str, description: str,
                                 current_value: float, threshold_value: float,
                                 risk_level: RiskLevel, recommended_action: OverrideAction,
                                 affected_symbols: List[str] = None, requires_immediate_action: bool = False):
        """Generate a risk alert"""
        try:
            alert = RiskAlert(
                alert_id=f"RISK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                risk_level=risk_level,
                risk_type=risk_type,
                description=description,
                affected_symbols=affected_symbols or [],
                current_value=current_value,
                threshold_value=threshold_value,
                recommended_action=recommended_action,
                timestamp=datetime.now(),
                requires_immediate_action=requires_immediate_action
            )

            # Add to alert queue for processing
            await self.alert_queue.put(alert)
            self.risk_alerts.append(alert)

            # Log alert
            self.logger.warning(f"RISK ALERT [{risk_level.value.upper()}]: {description}")

            # If critical, execute immediate action
            if requires_immediate_action and risk_level == RiskLevel.CRITICAL:
                await self.execute_emergency_override(alert)

        except Exception as e:
            self.logger.error(f"Error generating risk alert: {e}")

    async def alert_processor(self):
        """Process risk alerts from the queue"""
        while self.monitoring_active:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1)

                # Process alert based on risk level
                if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    await self.handle_high_risk_alert(alert)
                elif alert.risk_level == RiskLevel.MEDIUM:
                    await self.handle_medium_risk_alert(alert)

                # Notify emergency contacts if critical
                if alert.risk_level == RiskLevel.CRITICAL:
                    await self.send_emergency_notifications(alert)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")

    async def handle_high_risk_alert(self, alert: RiskAlert):
        """Handle high risk alerts"""
        try:
            # Create override command
            override_command = OverrideCommand(
                command_id=f"AUTO_OVERRIDE_{alert.alert_id}",
                action=alert.recommended_action,
                symbols=alert.affected_symbols,
                duration_minutes=30,  # 30 minute automatic override
                reason=f"Auto-override for {alert.risk_type}: {alert.description}",
                issued_by="SYSTEM_AUTO",
                timestamp=datetime.now()
            )

            # Execute override
            await self.execute_override_command(override_command)

        except Exception as e:
            self.logger.error(f"Error handling high risk alert: {e}")

    async def handle_medium_risk_alert(self, alert: RiskAlert):
        """Handle medium risk alerts"""
        try:
            # Log warning and add to monitoring
            self.logger.warning(f"Medium risk alert - monitoring closely: {alert.description}")

            # Could implement additional monitoring or human notification here

        except Exception as e:
            self.logger.error(f"Error handling medium risk alert: {e}")

    async def execute_emergency_override(self, alert: RiskAlert):
        """Execute immediate emergency override"""
        try:
            self.logger.critical(f"EXECUTING EMERGENCY OVERRIDE: {alert.description}")

            # Create emergency override command
            emergency_command = OverrideCommand(
                command_id=f"EMERGENCY_{alert.alert_id}",
                action=OverrideAction.HALT_ALL,
                symbols=[],  # All symbols
                duration_minutes=None,  # Indefinite until manual override
                reason=f"EMERGENCY: {alert.description}",
                issued_by="SYSTEM_EMERGENCY",
                timestamp=datetime.now()
            )

            # Execute immediately
            await self.execute_override_command(emergency_command)

            # Send emergency notifications
            await self.send_emergency_notifications(alert)

        except Exception as e:
            self.logger.error(f"Emergency override execution failed: {e}")

    async def execute_override_command(self, command: OverrideCommand):
        """Execute a risk override command"""
        try:
            # Add to active overrides
            self.active_overrides[command.command_id] = command
            self.override_history.append(command)

            # Notify all registered callbacks
            for callback in self.override_callbacks:
                try:
                    await callback(command)
                except Exception as e:
                    self.logger.error(f"Override callback error: {e}")

            # Log override execution
            self.logger.critical(f"RISK OVERRIDE EXECUTED: {command.action.value}")
            self.logger.critical(f"  Command ID: {command.command_id}")
            self.logger.critical(f"  Reason: {command.reason}")
            self.logger.critical(f"  Affected symbols: {command.symbols}")
            self.logger.critical(f"  Duration: {command.duration_minutes} minutes")

        except Exception as e:
            self.logger.error(f"Override command execution failed: {e}")

    async def manual_override(self, action: OverrideAction, symbols: List[str] = None,
                             duration_minutes: int = None, reason: str = "Manual override"):
        """Manually trigger a risk override"""
        try:
            command = OverrideCommand(
                command_id=f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                action=action,
                symbols=symbols or [],
                duration_minutes=duration_minutes,
                reason=reason,
                issued_by="MANUAL",
                timestamp=datetime.now()
            )

            await self.execute_override_command(command)
            return command.command_id

        except Exception as e:
            self.logger.error(f"Manual override failed: {e}")
            return None

    async def remove_override(self, command_id: str):
        """Remove an active override"""
        try:
            if command_id in self.active_overrides:
                command = self.active_overrides[command_id]
                del self.active_overrides[command_id]

                # Notify callbacks of override removal
                for callback in self.override_callbacks:
                    try:
                        # Call with None to indicate override removal
                        await callback(None, removed_command_id=command_id)
                    except Exception as e:
                        self.logger.error(f"Override removal callback error: {e}")

                self.logger.info(f"Risk override removed: {command_id}")
                return True
            else:
                self.logger.warning(f"Override command not found: {command_id}")
                return False

        except Exception as e:
            self.logger.error(f"Override removal failed: {e}")
            return False

    async def send_emergency_notifications(self, alert: RiskAlert):
        """Send emergency notifications"""
        try:
            for notification in self.emergency_notifications:
                if notification['enabled']:
                    # Simulate sending notification
                    self.logger.critical(f"EMERGENCY NOTIFICATION sent via {notification['method']} to {notification['contact']}")
                    self.logger.critical(f"Alert: {alert.description}")

        except Exception as e:
            self.logger.error(f"Emergency notification failed: {e}")

    async def get_current_portfolio(self) -> Dict:
        """Get current portfolio data (simulated)"""
        return {
            'daily_pnl': np.random.uniform(-1000, 1000),
            'total_value': 100000 + np.random.uniform(-5000, 5000),
            'peak_value': 105000
        }

    async def get_current_positions(self) -> Dict:
        """Get current positions (simulated)"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        positions = {}

        for symbol in symbols:
            if np.random.random() > 0.3:  # 70% chance of having position
                positions[symbol] = {
                    'quantity': np.random.randint(10, 100),
                    'market_value': np.random.uniform(1000, 20000),
                    'unrealized_pnl': np.random.uniform(-500, 500)
                }

        return positions

    async def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return 100000.0  # Simulated

    async def get_market_risk_data(self) -> Dict:
        """Get market risk data (simulated)"""
        return {
            'VIX': np.random.uniform(15, 35),
            'SPY': {'price_change_pct': np.random.uniform(-0.03, 0.03)},
            'QQQ': {'price_change_pct': np.random.uniform(-0.03, 0.03)},
            'AAPL': {'price_change_pct': np.random.uniform(-0.05, 0.05)}
        }

    def get_risk_status(self) -> Dict:
        """Get current risk system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "total_alerts": len(self.risk_alerts),
            "active_overrides": len(self.active_overrides),
            "critical_alerts": len([a for a in self.risk_alerts if a.risk_level == RiskLevel.CRITICAL]),
            "recent_alerts": len([a for a in self.risk_alerts if (datetime.now() - a.timestamp).seconds < 300]),
            "override_history": len(self.override_history),
            "emergency_notifications_configured": len(self.emergency_notifications)
        }

    def stop_risk_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        self.logger.info("Risk monitoring stopped")

async def demo_risk_override_system():
    """Demo the risk override system"""
    print("="*80)
    print("REAL-TIME RISK OVERRIDE SYSTEM DEMO")
    print("Instant risk management and trading halt capabilities")
    print("="*80)

    # Initialize risk system
    risk_system = RealTimeRiskOverrideSystem()

    # Add emergency notification
    risk_system.add_emergency_notification("email", "trader@example.com")
    risk_system.add_emergency_notification("sms", "+1234567890")

    # Demo override callback
    async def demo_override_callback(command):
        if command:
            print(f"OVERRIDE EXECUTED: {command.action.value} - {command.reason}")
        else:
            print("Override removed")

    risk_system.register_override_callback(demo_override_callback)

    print(f"\nStarting risk monitoring demo for 20 seconds...")
    try:
        await asyncio.wait_for(risk_system.start_risk_monitoring(), timeout=20)
    except asyncio.TimeoutError:
        print("\nDemo completed")
    finally:
        risk_system.stop_risk_monitoring()

        # Show final status
        status = risk_system.get_risk_status()
        print(f"\nRisk System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    print(f"\nReal-time Risk Override System ready for live trading!")

if __name__ == "__main__":
    asyncio.run(demo_risk_override_system())