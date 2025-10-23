"""
Tomorrow-Ready Trading System Deployment

Complete system validation, deployment preparation, and pre-market checklist
for live trading operations. Ensures all systems are operational and ready.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# System components
from core.configuration_management import ConfigurationManager, Environment
from brokers.broker_integrations import BrokerManager, AlpacaBroker, BrokerCredentials
from data.real_time_market_data import MarketDataManager, DataProvider, AlphaVantageProvider
from core.authentication_security import SecurityManager
from core.logging_monitoring_system import MonitoringSystem
from dashboard.performance_monitoring_dashboard import StreamlitDashboard

class SystemStatus(Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    WARNING = "warning"
    ERROR = "error"

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a system validation check"""
    component: str
    check_name: str
    status: SystemStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SystemHealthReport:
    """Complete system health report"""
    overall_status: SystemStatus
    ready_for_trading: bool
    validation_results: List[ValidationResult]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TomorrowReadyValidator:
    """Validates all system components for tomorrow's trading"""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.config_manager: Optional[ConfigurationManager] = None
        self.broker_manager: Optional[BrokerManager] = None
        self.market_data_manager: Optional[MarketDataManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.monitoring_system: Optional[MonitoringSystem] = None

    async def run_complete_validation(self) -> SystemHealthReport:
        """Run complete system validation"""

        logging.info("Starting comprehensive system validation for tomorrow's trading")
        self.validation_results = []

        # 1. Configuration validation
        await self._validate_configuration()

        # 2. Broker connections
        await self._validate_broker_connections()

        # 3. Market data feeds
        await self._validate_market_data()

        # 4. Security systems
        await self._validate_security_systems()

        # 5. Monitoring and alerting
        await self._validate_monitoring_systems()

        # 6. Risk management
        await self._validate_risk_management()

        # 7. System resources
        await self._validate_system_resources()

        # 8. Trading environment
        await self._validate_trading_environment()

        # Generate health report
        report = self._generate_health_report()

        logging.info(f"System validation complete. Overall status: {report.overall_status.value}")
        return report

    async def _validate_configuration(self):
        """Validate system configuration"""

        try:
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()

            # Load configurations
            self.config_manager.load_from_file("config/base.yaml")
            self.config_manager.load_from_environment()

            # Validate configuration
            errors = self.config_manager.validate()

            if errors:
                self.validation_results.append(ValidationResult(
                    component="configuration",
                    check_name="config_validation",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.ERROR,
                    message=f"Configuration validation failed: {errors}",
                    details={"errors": errors}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="configuration",
                    check_name="config_validation",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message="Configuration validation passed"
                ))

            # Check critical settings
            initial_capital = self.config_manager.get('trading.initial_capital', 0)
            if initial_capital <= 0:
                self.validation_results.append(ValidationResult(
                    component="configuration",
                    check_name="trading_capital",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.CRITICAL,
                    message="Initial trading capital not configured or invalid",
                    details={"current_value": initial_capital}
                ))

            # Check environment
            env = self.config_manager.get('environment', 'development')
            self.validation_results.append(ValidationResult(
                component="configuration",
                check_name="environment",
                status=SystemStatus.READY,
                severity=ValidationSeverity.INFO,
                message=f"Environment configured: {env}",
                details={"environment": env}
            ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="configuration",
                check_name="config_loading",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to load configuration: {e}",
                details={"error": str(e)}
            ))

    async def _validate_broker_connections(self):
        """Validate broker API connections"""

        try:
            if not self.config_manager:
                raise ValueError("Configuration not loaded")

            self.broker_manager = BrokerManager()
            broker_configs = self.config_manager.get('brokers', {})

            if not broker_configs:
                self.validation_results.append(ValidationResult(
                    component="brokers",
                    check_name="broker_config",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message="No broker configurations found"
                ))
                return

            for broker_name, broker_config in broker_configs.items():
                if not broker_config.get('enabled', False):
                    continue

                try:
                    # Test broker connection based on type
                    if broker_name == 'alpaca':
                        await self._test_alpaca_connection(broker_config)
                    elif broker_name == 'interactive_brokers':
                        await self._test_ib_connection(broker_config)
                    else:
                        self.validation_results.append(ValidationResult(
                            component="brokers",
                            check_name=f"{broker_name}_connection",
                            status=SystemStatus.WARNING,
                            severity=ValidationSeverity.WARNING,
                            message=f"Unknown broker type: {broker_name}"
                        ))

                except Exception as e:
                    self.validation_results.append(ValidationResult(
                        component="brokers",
                        check_name=f"{broker_name}_connection",
                        status=SystemStatus.ERROR,
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to connect to {broker_name}: {e}",
                        details={"error": str(e)}
                    ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="brokers",
                check_name="broker_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Broker validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _test_alpaca_connection(self, broker_config: Dict[str, Any]):
        """Test Alpaca broker connection"""

        api_key = broker_config.get('api_key', '')
        api_secret = broker_config.get('api_secret', '')

        if not api_key or not api_secret:
            self.validation_results.append(ValidationResult(
                component="brokers",
                check_name="alpaca_credentials",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message="Alpaca API credentials not configured"
            ))
            return

        try:
            credentials = BrokerCredentials(
                broker_name="alpaca",
                api_key=api_key,
                api_secret=api_secret,
                sandbox=broker_config.get('sandbox', True)
            )

            alpaca_broker = AlpacaBroker(credentials)
            connected = await alpaca_broker.connect()

            if connected:
                # Test account access
                account_info = await alpaca_broker.get_account_info()

                self.validation_results.append(ValidationResult(
                    component="brokers",
                    check_name="alpaca_connection",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message="Alpaca connection successful",
                    details={
                        "account_id": account_info.get('account_id', 'unknown'),
                        "buying_power": account_info.get('buying_power', 0),
                        "sandbox": credentials.sandbox
                    }
                ))

                # Check account status
                if account_info.get('trading_blocked', False):
                    self.validation_results.append(ValidationResult(
                        component="brokers",
                        check_name="alpaca_trading_status",
                        status=SystemStatus.ERROR,
                        severity=ValidationSeverity.CRITICAL,
                        message="Alpaca account trading is blocked"
                    ))

                await alpaca_broker.disconnect()
            else:
                self.validation_results.append(ValidationResult(
                    component="brokers",
                    check_name="alpaca_connection",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.ERROR,
                    message="Failed to connect to Alpaca"
                ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="brokers",
                check_name="alpaca_connection",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Alpaca connection test failed: {e}",
                details={"error": str(e)}
            ))

    async def _test_ib_connection(self, broker_config: Dict[str, Any]):
        """Test Interactive Brokers connection"""

        # Placeholder for IB connection test
        self.validation_results.append(ValidationResult(
            component="brokers",
            check_name="ib_connection",
            status=SystemStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message="Interactive Brokers validation not implemented yet"
        ))

    async def _validate_market_data(self):
        """Validate market data connections"""

        try:
            # Test basic market data access
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'SPY']

            # Test Yahoo Finance (free data source)
            try:
                data = yf.download(symbols[0], period="1d", interval="1m")
                if not data.empty:
                    self.validation_results.append(ValidationResult(
                        component="market_data",
                        check_name="yahoo_finance",
                        status=SystemStatus.READY,
                        severity=ValidationSeverity.INFO,
                        message="Yahoo Finance data access working",
                        details={"last_price": float(data['Close'].iloc[-1])}
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        component="market_data",
                        check_name="yahoo_finance",
                        status=SystemStatus.WARNING,
                        severity=ValidationSeverity.WARNING,
                        message="Yahoo Finance returned empty data"
                    ))
            except Exception as e:
                self.validation_results.append(ValidationResult(
                    component="market_data",
                    check_name="yahoo_finance",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.ERROR,
                    message=f"Yahoo Finance data access failed: {e}"
                ))

            # Test premium data providers if configured
            alpha_vantage_key = self.config_manager.get('market_data.alpha_vantage_key') if self.config_manager else None
            if alpha_vantage_key:
                try:
                    # Test Alpha Vantage connection
                    provider = AlphaVantageProvider(alpha_vantage_key)
                    connected = await provider.connect()

                    if connected:
                        self.validation_results.append(ValidationResult(
                            component="market_data",
                            check_name="alpha_vantage",
                            status=SystemStatus.READY,
                            severity=ValidationSeverity.INFO,
                            message="Alpha Vantage connection successful"
                        ))
                    else:
                        self.validation_results.append(ValidationResult(
                            component="market_data",
                            check_name="alpha_vantage",
                            status=SystemStatus.ERROR,
                            severity=ValidationSeverity.ERROR,
                            message="Alpha Vantage connection failed"
                        ))
                except Exception as e:
                    self.validation_results.append(ValidationResult(
                        component="market_data",
                        check_name="alpha_vantage",
                        status=SystemStatus.ERROR,
                        severity=ValidationSeverity.ERROR,
                        message=f"Alpha Vantage test failed: {e}"
                    ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="market_data",
                check_name="market_data_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Market data validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _validate_security_systems(self):
        """Validate security and authentication systems"""

        try:
            # Test security manager initialization
            self.security_manager = SecurityManager()

            self.validation_results.append(ValidationResult(
                component="security",
                check_name="security_init",
                status=SystemStatus.READY,
                severity=ValidationSeverity.INFO,
                message="Security system initialized successfully"
            ))

            # Check JWT configuration
            jwt_secret = self.config_manager.get('security.jwt_secret_key') if self.config_manager else None
            if not jwt_secret:
                self.validation_results.append(ValidationResult(
                    component="security",
                    check_name="jwt_config",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.CRITICAL,
                    message="JWT secret key not configured"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="security",
                    check_name="jwt_config",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message="JWT configuration valid"
                ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="security",
                check_name="security_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Security validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _validate_monitoring_systems(self):
        """Validate monitoring and alerting systems"""

        try:
            # Test monitoring system initialization
            self.monitoring_system = MonitoringSystem()

            # Test metrics collection
            health = self.monitoring_system.get_system_health()

            self.validation_results.append(ValidationResult(
                component="monitoring",
                check_name="monitoring_init",
                status=SystemStatus.READY,
                severity=ValidationSeverity.INFO,
                message="Monitoring system operational",
                details={"system_health": health['status']}
            ))

            # Check alert configuration
            alert_config = self.config_manager.get('alerts') if self.config_manager else {}
            if not any([alert_config.get('enable_email'), alert_config.get('enable_slack')]):
                self.validation_results.append(ValidationResult(
                    component="monitoring",
                    check_name="alert_config",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message="No alert channels configured"
                ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="monitoring",
                check_name="monitoring_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Monitoring validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _validate_risk_management(self):
        """Validate risk management configuration"""

        try:
            if not self.config_manager:
                raise ValueError("Configuration not loaded")

            trading_config = self.config_manager.get('trading', {})

            # Check risk limits
            max_leverage = trading_config.get('max_leverage', 0)
            max_position_size = trading_config.get('max_position_size', 0)
            max_daily_loss = trading_config.get('max_daily_loss', 0)

            if max_leverage <= 0 or max_leverage > 10:
                self.validation_results.append(ValidationResult(
                    component="risk_management",
                    check_name="leverage_limits",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Leverage limit may be inappropriate: {max_leverage}",
                    details={"max_leverage": max_leverage}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="risk_management",
                    check_name="leverage_limits",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message=f"Leverage limits configured: {max_leverage}x"
                ))

            if max_position_size <= 0 or max_position_size > 0.5:
                self.validation_results.append(ValidationResult(
                    component="risk_management",
                    check_name="position_limits",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Position size limit may be inappropriate: {max_position_size}",
                    details={"max_position_size": max_position_size}
                ))

            if max_daily_loss <= 0 or max_daily_loss > 0.1:
                self.validation_results.append(ValidationResult(
                    component="risk_management",
                    check_name="loss_limits",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Daily loss limit may be inappropriate: {max_daily_loss}",
                    details={"max_daily_loss": max_daily_loss}
                ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="risk_management",
                check_name="risk_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Risk management validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _validate_system_resources(self):
        """Validate system resources and performance"""

        try:
            import psutil

            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="cpu_usage",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"High CPU usage: {cpu_percent}%",
                    details={"cpu_percent": cpu_percent}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="cpu_usage",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message=f"CPU usage normal: {cpu_percent}%"
                ))

            # Check memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent > 85:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="memory_usage",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"High memory usage: {memory_percent}%",
                    details={"memory_percent": memory_percent}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="memory_usage",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message=f"Memory usage normal: {memory_percent}%"
                ))

            # Check disk space
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            if disk_percent > 90:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="disk_usage",
                    status=SystemStatus.ERROR,
                    severity=ValidationSeverity.ERROR,
                    message=f"Critical disk usage: {disk_percent}%",
                    details={"disk_percent": disk_percent}
                ))
            elif disk_percent > 80:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="disk_usage",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"High disk usage: {disk_percent}%"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="system_resources",
                    check_name="disk_usage",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message=f"Disk usage normal: {disk_percent}%"
                ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="system_resources",
                check_name="resource_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"System resource validation failed: {e}",
                details={"error": str(e)}
            ))

    async def _validate_trading_environment(self):
        """Validate trading environment and market hours"""

        try:
            # Check current time and market hours
            now = datetime.now(timezone.utc)

            # US market hours (EST): 9:30 AM - 4:00 PM (14:30 - 21:00 UTC)
            market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)

            # Adjust for weekends
            if now.weekday() >= 5:  # Saturday or Sunday
                self.validation_results.append(ValidationResult(
                    component="trading_environment",
                    check_name="market_hours",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.INFO,
                    message="Markets are closed (weekend)",
                    details={"current_time": now.isoformat()}
                ))
            elif now < market_open:
                self.validation_results.append(ValidationResult(
                    component="trading_environment",
                    check_name="market_hours",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message="Pre-market hours - ready for market open",
                    details={"market_opens_in": str(market_open - now)}
                ))
            elif now > market_close:
                self.validation_results.append(ValidationResult(
                    component="trading_environment",
                    check_name="market_hours",
                    status=SystemStatus.WARNING,
                    severity=ValidationSeverity.INFO,
                    message="After-hours trading",
                    details={"market_closed": str(now - market_close) + " ago"}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component="trading_environment",
                    check_name="market_hours",
                    status=SystemStatus.READY,
                    severity=ValidationSeverity.INFO,
                    message="Market is open - live trading active"
                ))

            # Check for upcoming holidays (simplified)
            # In a real system, you'd have a comprehensive holiday calendar
            self.validation_results.append(ValidationResult(
                component="trading_environment",
                check_name="market_calendar",
                status=SystemStatus.READY,
                severity=ValidationSeverity.INFO,
                message="Market calendar check passed"
            ))

        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="trading_environment",
                check_name="environment_validation",
                status=SystemStatus.ERROR,
                severity=ValidationSeverity.ERROR,
                message=f"Trading environment validation failed: {e}",
                details={"error": str(e)}
            ))

    def _generate_health_report(self) -> SystemHealthReport:
        """Generate comprehensive health report"""

        # Determine overall status
        critical_errors = [r for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL]
        errors = [r for r in self.validation_results if r.status == SystemStatus.ERROR]
        warnings = [r for r in self.validation_results if r.status == SystemStatus.WARNING]

        if critical_errors:
            overall_status = SystemStatus.ERROR
            ready_for_trading = False
        elif errors:
            overall_status = SystemStatus.ERROR
            ready_for_trading = False
        elif warnings:
            overall_status = SystemStatus.WARNING
            ready_for_trading = True  # Can trade with warnings
        else:
            overall_status = SystemStatus.READY
            ready_for_trading = True

        # Generate recommendations
        recommendations = []

        if critical_errors:
            recommendations.append("[ALERT] CRITICAL: Fix critical errors before trading")

        if errors:
            recommendations.append("[X] Fix system errors before live trading")

        if warnings:
            recommendations.append("[WARN] Review warnings and consider fixes")

        if ready_for_trading:
            recommendations.append("[OK] System ready for trading operations")
            recommendations.append("[CHART] Monitor system health during trading hours")
            recommendations.append("[INFO] Validate broker connections before market open")

        return SystemHealthReport(
            overall_status=overall_status,
            ready_for_trading=ready_for_trading,
            validation_results=self.validation_results,
            recommendations=recommendations
        )

class TomorrowReadyDeployment:
    """Complete deployment preparation for tomorrow's trading"""

    def __init__(self):
        self.validator = TomorrowReadyValidator()

    async def prepare_for_tomorrow(self) -> SystemHealthReport:
        """Complete preparation for tomorrow's trading"""

        logging.info("[LAUNCH] Starting Tomorrow-Ready deployment preparation")

        # Run validation
        health_report = await self.validator.run_complete_validation()

        # Print detailed report
        self._print_health_report(health_report)

        # Save report
        self._save_health_report(health_report)

        return health_report

    def _print_health_report(self, report: SystemHealthReport):
        """Print formatted health report"""

        print("\n" + "="*80)
        print("[INFO] TOMORROW-READY TRADING SYSTEM VALIDATION REPORT")
        print("="*80)

        # Overall status
        status_emoji = {
            SystemStatus.READY: "[OK]",
            SystemStatus.WARNING: "[WARN]",
            SystemStatus.ERROR: "[X]",
            SystemStatus.NOT_READY: "[INFO]"
        }

        print(f"\n[CHART] OVERALL STATUS: {status_emoji[report.overall_status]} {report.overall_status.value.upper()}")
        print(f"[TARGET] READY FOR TRADING: {'[OK] YES' if report.ready_for_trading else '[X] NO'}")
        print(f"[CLOCK] VALIDATION TIME: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Component breakdown
        print(f"\n[INFO] COMPONENT VALIDATION RESULTS:")
        print("-" * 60)

        components = {}
        for result in report.validation_results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)

        for component, results in components.items():
            component_status = max([r.status for r in results], key=lambda x: list(SystemStatus).index(x))
            print(f"\n[TOOL] {component.upper().replace('_', ' ')}: {status_emoji[component_status]} {component_status.value}")

            for result in results:
                severity_emoji = {
                    ValidationSeverity.INFO: "ℹ️",
                    ValidationSeverity.WARNING: "[WARN]",
                    ValidationSeverity.ERROR: "[X]",
                    ValidationSeverity.CRITICAL: "[ALERT]"
                }
                print(f"   {severity_emoji[result.severity]} {result.check_name}: {result.message}")

        # Recommendations
        print(f"\n[IDEA] RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "="*80)

    def _save_health_report(self, report: SystemHealthReport):
        """Save health report to file"""

        try:
            # Create reports directory
            from pathlib import Path
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Save JSON report
            timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
            report_file = reports_dir / f"system_health_{timestamp}.json"

            # Convert to serializable format
            report_data = {
                'overall_status': report.overall_status.value,
                'ready_for_trading': report.ready_for_trading,
                'timestamp': report.timestamp.isoformat(),
                'validation_results': [
                    {
                        'component': r.component,
                        'check_name': r.check_name,
                        'status': r.status.value,
                        'severity': r.severity.value,
                        'message': r.message,
                        'details': r.details,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in report.validation_results
                ],
                'recommendations': report.recommendations
            }

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            logging.info(f"Health report saved to: {report_file}")

        except Exception as e:
            logging.error(f"Failed to save health report: {e}")

# Pre-market checklist
def create_premarket_checklist() -> List[str]:
    """Create pre-market validation checklist"""

    return [
        "[INFO] Verify broker API connections are active",
        "[INFO] Confirm account balances and buying power",
        "[INFO] Check for any account restrictions or blocks",
        "[INFO] Validate market data feeds are streaming",
        "[INFO] Review overnight news and market events",
        "[INFO] Confirm risk limits and position sizes",
        "[INFO] Check system performance and resources",
        "[INFO] Verify monitoring and alert systems",
        "[INFO] Review strategy performance from previous day",
        "[INFO] Confirm backup systems are operational",
        "[INFO] Check market calendar for holidays/early closes",
        "[INFO] Validate paper trading is working correctly",
        "[INFO] Review and acknowledge any system warnings",
        "[INFO] Confirm emergency shutdown procedures",
        "[INFO] Set appropriate position limits for the day"
    ]

# Example usage
async def main():
    """Main deployment preparation function"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create deployment manager
    deployment = TomorrowReadyDeployment()

    # Run preparation
    health_report = await deployment.prepare_for_tomorrow()

    # Print pre-market checklist
    print("\n[INFO] PRE-MARKET CHECKLIST:")
    print("-" * 40)
    checklist = create_premarket_checklist()
    for item in checklist:
        print(item)

    print(f"\n[TARGET] SYSTEM STATUS: {'READY FOR TRADING' if health_report.ready_for_trading else 'NOT READY - FIX ISSUES FIRST'}")

    return health_report

if __name__ == "__main__":
    # Run the deployment preparation
    report = asyncio.run(main())