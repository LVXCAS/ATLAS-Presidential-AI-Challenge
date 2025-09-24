#!/usr/bin/env python3
"""
ADVANCED R&D SYSTEM CONFIGURATION
=================================

Comprehensive configuration management for the Hive Trading R&D System,
allowing fine-tuning of all research parameters, risk controls, and
operational settings for maximum performance.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration"""
    portfolio_simulations: int = 10000
    strategy_simulations: int = 5000
    time_horizon_days: int = 252
    confidence_levels: List[float] = None
    stress_scenarios: List[str] = None
    correlation_breakdown: bool = True
    tail_analysis: bool = True
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.stress_scenarios is None:
            self.stress_scenarios = ['2008_crisis', '2020_covid', 'dot_com_bubble', 'flash_crash']

@dataclass
class QlibConfig:
    """Qlib strategy generation configuration"""
    factor_categories: List[str] = None
    lookback_periods: List[int] = None
    rebalance_frequencies: List[str] = None
    universe_size: int = 500
    min_market_cap: float = 1e9  # $1B minimum
    sector_constraints: Dict[str, float] = None
    factor_decay_analysis: bool = True
    regime_awareness: bool = True
    
    def __post_init__(self):
        if self.factor_categories is None:
            self.factor_categories = [
                'momentum', 'mean_reversion', 'volatility', 'quality', 
                'value', 'growth', 'profitability', 'leverage'
            ]
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 60, 120, 252]
        if self.rebalance_frequencies is None:
            self.rebalance_frequencies = ['daily', 'weekly', 'monthly']
        if self.sector_constraints is None:
            self.sector_constraints = {
                'technology': 0.30,
                'healthcare': 0.25,
                'financials': 0.20,
                'consumer_discretionary': 0.15,
                'industrials': 0.15,
                'other': 0.10
            }

@dataclass  
class GSQuantConfig:
    """GS-Quant risk modeling configuration"""
    risk_models: List[str] = None
    factor_exposures: List[str] = None
    sector_models: List[str] = None
    stress_scenarios: List[str] = None
    correlation_models: List[str] = None
    liquidity_analysis: bool = True
    credit_analysis: bool = True
    
    def __post_init__(self):
        if self.risk_models is None:
            self.risk_models = ['US_EQUITY', 'GLOBAL_EQUITY', 'FIXED_INCOME']
        if self.factor_exposures is None:
            self.factor_exposures = [
                'market_beta', 'size', 'value', 'momentum', 'quality',
                'volatility', 'growth', 'leverage', 'profitability'
            ]
        if self.sector_models is None:
            self.sector_models = ['GICS_LEVEL_1', 'GICS_LEVEL_2', 'CUSTOM']
        if self.stress_scenarios is None:
            self.stress_scenarios = [
                'rates_up_100bp', 'rates_down_100bp', 'equity_down_20pct',
                'vix_spike_40', 'credit_spread_widen', 'dollar_strength'
            ]
        if self.correlation_models is None:
            self.correlation_models = ['EWMA', 'DCC_GARCH', 'FACTOR_MODEL']

@dataclass
class LEANConfig:
    """LEAN backtesting configuration"""
    backtest_period_years: int = 5
    benchmark: str = 'SPY'
    initial_cash: float = 1000000
    transaction_costs: Dict[str, float] = None
    slippage_model: str = 'VolumeShareSlippageModel'
    fill_model: str = 'ImmediateFillModel'
    reality_modeling: bool = True
    survivorship_bias_free: bool = True
    
    def __post_init__(self):
        if self.transaction_costs is None:
            self.transaction_costs = {
                'equity_commission': 0.001,  # 10 bps
                'options_commission': 1.0,   # $1 per contract
                'futures_commission': 2.0,   # $2 per contract
                'forex_spread': 0.0001       # 1 pip
            }

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_portfolio_volatility: float = 0.20
    max_individual_weight: float = 0.10
    max_sector_weight: float = 0.25
    max_drawdown_limit: float = 0.15
    var_confidence_level: float = 0.95
    correlation_threshold: float = 0.70
    min_sharpe_ratio: float = 1.0
    max_leverage: float = 1.0
    stress_test_frequency: str = 'weekly'
    rebalance_threshold: float = 0.05
    
@dataclass
class DeploymentConfig:
    """Strategy deployment configuration"""
    quality_score_threshold: float = 70.0
    min_backtest_period_months: int = 24
    min_sharpe_ratio: float = 1.0
    max_drawdown_threshold: float = 0.20
    min_win_rate: float = 0.45
    statistical_significance: float = 0.05  # p-value
    max_auto_allocation: float = 0.05  # 5%
    approval_required_above: float = 0.10  # 10%
    paper_trading_period_days: int = 30
    gradual_scaling_steps: int = 5

@dataclass
class OperationalConfig:
    """Operational settings"""
    rd_session_frequency: str = 'every_4_hours'
    max_concurrent_strategies: int = 50
    strategy_retirement_criteria: Dict[str, Union[int, float]] = None
    performance_review_frequency: str = 'monthly'
    system_health_check_frequency: str = 'hourly'
    data_retention_days: int = 365
    log_level: str = 'INFO'
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.strategy_retirement_criteria is None:
            self.strategy_retirement_criteria = {
                'consecutive_months_underperforming': 6,
                'max_drawdown_breach': 0.25,
                'sharpe_ratio_below': 0.5,
                'correlation_too_high': 0.80
            }
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'portfolio_drawdown': 0.10,
                'daily_var_breach': 1.5,
                'correlation_spike': 0.75,
                'volatility_spike': 2.0
            }

class RDSystemConfig:
    """Master configuration class for the entire R&D system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'rd_system_config.json'
        
        # Initialize all configuration components
        self.monte_carlo = MonteCarloConfig()
        self.qlib = QlibConfig()
        self.gs_quant = GSQuantConfig()
        self.lean = LEANConfig()
        self.risk = RiskConfig()
        self.deployment = DeploymentConfig()
        self.operational = OperationalConfig()
        
        # System metadata
        self.metadata = {
            'config_version': '1.0',
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'environment': 'production'
        }
        
        # Load existing configuration if available
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'monte_carlo' in config_data:
                self.monte_carlo = MonteCarloConfig(**config_data['monte_carlo'])
            if 'qlib' in config_data:
                self.qlib = QlibConfig(**config_data['qlib'])
            if 'gs_quant' in config_data:
                self.gs_quant = GSQuantConfig(**config_data['gs_quant'])
            if 'lean' in config_data:
                self.lean = LEANConfig(**config_data['lean'])
            if 'risk' in config_data:
                self.risk = RiskConfig(**config_data['risk'])
            if 'deployment' in config_data:
                self.deployment = DeploymentConfig(**config_data['deployment'])
            if 'operational' in config_data:
                self.operational = OperationalConfig(**config_data['operational'])
            if 'metadata' in config_data:
                self.metadata.update(config_data['metadata'])
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except FileNotFoundError:
            logger.info(f"Config file {self.config_file} not found, using defaults")
            self.save_config()  # Save default configuration
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        config_data = {
            'monte_carlo': asdict(self.monte_carlo),
            'qlib': asdict(self.qlib),
            'gs_quant': asdict(self.gs_quant),
            'lean': asdict(self.lean),
            'risk': asdict(self.risk),
            'deployment': asdict(self.deployment),
            'operational': asdict(self.operational),
            'metadata': self.metadata
        }
        
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {self.config_file}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings"""
        warnings = []
        
        # Risk validation
        if self.risk.max_portfolio_volatility > 0.30:
            warnings.append("Portfolio volatility limit very high (>30%)")
        
        if self.risk.max_drawdown_limit > 0.25:
            warnings.append("Drawdown limit very high (>25%)")
        
        # Deployment validation
        if self.deployment.quality_score_threshold < 50:
            warnings.append("Quality score threshold very low (<50)")
        
        if self.deployment.max_auto_allocation > 0.10:
            warnings.append("Auto allocation limit very high (>10%)")
        
        # Monte Carlo validation
        if self.monte_carlo.portfolio_simulations < 1000:
            warnings.append("Portfolio simulations count low (<1000)")
        
        # Operational validation
        if self.operational.max_concurrent_strategies > 100:
            warnings.append("Max concurrent strategies very high (>100)")
        
        return warnings
    
    def get_conservative_config(self) -> 'RDSystemConfig':
        """Get conservative configuration for risk-averse environments"""
        conservative = RDSystemConfig()
        
        # More conservative risk settings
        conservative.risk.max_portfolio_volatility = 0.12
        conservative.risk.max_individual_weight = 0.05
        conservative.risk.max_drawdown_limit = 0.10
        conservative.risk.min_sharpe_ratio = 1.5
        
        # More stringent deployment criteria
        conservative.deployment.quality_score_threshold = 80.0
        conservative.deployment.min_sharpe_ratio = 1.5
        conservative.deployment.max_drawdown_threshold = 0.15
        conservative.deployment.max_auto_allocation = 0.02
        
        # More Monte Carlo simulations for validation
        conservative.monte_carlo.portfolio_simulations = 20000
        conservative.monte_carlo.strategy_simulations = 10000
        
        return conservative
    
    def get_aggressive_config(self) -> 'RDSystemConfig':
        """Get aggressive configuration for higher risk tolerance"""
        aggressive = RDSystemConfig()
        
        # More aggressive risk settings
        aggressive.risk.max_portfolio_volatility = 0.25
        aggressive.risk.max_individual_weight = 0.15
        aggressive.risk.max_drawdown_limit = 0.20
        aggressive.risk.min_sharpe_ratio = 0.8
        aggressive.risk.max_leverage = 1.5
        
        # Less stringent deployment criteria
        aggressive.deployment.quality_score_threshold = 60.0
        aggressive.deployment.min_sharpe_ratio = 0.8
        aggressive.deployment.max_drawdown_threshold = 0.25
        aggressive.deployment.max_auto_allocation = 0.08
        
        # More strategy generation
        aggressive.operational.max_concurrent_strategies = 75
        aggressive.operational.rd_session_frequency = 'every_2_hours'
        
        return aggressive
    
    def optimize_for_environment(self, environment: str):
        """Optimize configuration for specific environment"""
        
        if environment.lower() == 'development':
            # Faster iterations, more logging
            self.monte_carlo.portfolio_simulations = 1000
            self.monte_carlo.strategy_simulations = 500
            self.lean.backtest_period_years = 2
            self.operational.log_level = 'DEBUG'
            self.deployment.paper_trading_period_days = 7
            
        elif environment.lower() == 'testing':
            # Comprehensive validation
            self.monte_carlo.portfolio_simulations = 5000
            self.monte_carlo.strategy_simulations = 2000
            self.lean.backtest_period_years = 3
            self.deployment.paper_trading_period_days = 14
            
        elif environment.lower() == 'production':
            # Full validation and conservative settings
            self.monte_carlo.portfolio_simulations = 10000
            self.monte_carlo.strategy_simulations = 5000
            self.lean.backtest_period_years = 5
            self.deployment.paper_trading_period_days = 30
            
        self.metadata['environment'] = environment
    
    def create_custom_profile(self, name: str, modifications: Dict):
        """Create custom configuration profile"""
        profile = {
            'name': name,
            'created_date': datetime.now().isoformat(),
            'modifications': modifications,
            'base_config': 'default'
        }
        
        # Apply modifications
        for section, settings in modifications.items():
            if hasattr(self, section):
                config_section = getattr(self, section)
                for key, value in settings.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
        
        # Save profile
        profile_file = f'rd_profile_{name.lower()}.json'
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Custom profile '{name}' created and saved to {profile_file}")
    
    def generate_config_report(self) -> str:
        """Generate comprehensive configuration report"""
        
        report = f"""
HIVE TRADING R&D SYSTEM CONFIGURATION REPORT
=============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Environment: {self.metadata['environment']}
Config Version: {self.metadata['config_version']}

MONTE CARLO SIMULATION SETTINGS:
- Portfolio Simulations: {self.monte_carlo.portfolio_simulations:,}
- Strategy Simulations: {self.monte_carlo.strategy_simulations:,}
- Time Horizon: {self.monte_carlo.time_horizon_days} days
- Confidence Levels: {self.monte_carlo.confidence_levels}
- Stress Scenarios: {len(self.monte_carlo.stress_scenarios)}

QLIB FACTOR RESEARCH SETTINGS:
- Factor Categories: {len(self.qlib.factor_categories)}
- Lookback Periods: {self.qlib.lookback_periods}
- Universe Size: {self.qlib.universe_size:,} stocks
- Min Market Cap: ${self.qlib.min_market_cap/1e9:.1f}B
- Regime Awareness: {self.qlib.regime_awareness}

GS-QUANT RISK MODELING:
- Risk Models: {self.gs_quant.risk_models}
- Factor Exposures: {len(self.gs_quant.factor_exposures)}
- Stress Scenarios: {len(self.gs_quant.stress_scenarios)}
- Liquidity Analysis: {self.gs_quant.liquidity_analysis}

LEAN BACKTESTING SETTINGS:
- Backtest Period: {self.lean.backtest_period_years} years
- Initial Cash: ${self.lean.initial_cash:,.0f}
- Equity Commission: {self.lean.transaction_costs['equity_commission']:.3f}
- Reality Modeling: {self.lean.reality_modeling}

RISK MANAGEMENT CONTROLS:
- Max Portfolio Vol: {self.risk.max_portfolio_volatility:.1%}
- Max Individual Weight: {self.risk.max_individual_weight:.1%}
- Max Drawdown: {self.risk.max_drawdown_limit:.1%}
- Min Sharpe Ratio: {self.risk.min_sharpe_ratio:.1f}
- VaR Confidence: {self.risk.var_confidence_level:.1%}

DEPLOYMENT CRITERIA:
- Quality Threshold: {self.deployment.quality_score_threshold:.1f}
- Min Sharpe Ratio: {self.deployment.min_sharpe_ratio:.1f}
- Max Auto Allocation: {self.deployment.max_auto_allocation:.1%}
- Paper Trading Period: {self.deployment.paper_trading_period_days} days

OPERATIONAL SETTINGS:
- R&D Frequency: {self.operational.rd_session_frequency}
- Max Strategies: {self.operational.max_concurrent_strategies}
- Performance Review: {self.operational.performance_review_frequency}
- Data Retention: {self.operational.data_retention_days} days

CONFIGURATION HEALTH CHECK:
"""
        
        warnings = self.validate_config()
        if warnings:
            report += "\nWARNINGS:\n"
            for warning in warnings:
                report += f"- {warning}\n"
        else:
            report += "\nALL SETTINGS VALIDATED - NO WARNINGS\n"
        
        return report

def main():
    """Configuration management interface"""
    
    # Create default configuration
    config = RDSystemConfig()
    
    print("HIVE TRADING R&D SYSTEM CONFIGURATION")
    print("=" * 50)
    
    # Generate and display configuration report
    report = config.generate_config_report()
    print(report)
    
    # Demonstrate profile creation
    print("\nCREATING SAMPLE PROFILES...")
    
    # Conservative profile
    conservative_modifications = {
        'risk': {
            'max_portfolio_volatility': 0.12,
            'max_drawdown_limit': 0.08,
            'min_sharpe_ratio': 1.8
        },
        'deployment': {
            'quality_score_threshold': 85.0,
            'max_auto_allocation': 0.01
        }
    }
    config.create_custom_profile('Conservative', conservative_modifications)
    
    # High-frequency profile
    hf_modifications = {
        'operational': {
            'rd_session_frequency': 'every_2_hours',
            'max_concurrent_strategies': 100
        },
        'qlib': {
            'rebalance_frequencies': ['daily', 'intraday_4h', 'intraday_1h']
        }
    }
    config.create_custom_profile('HighFrequency', hf_modifications)
    
    print("Sample profiles created!")
    print("\nConfiguration system ready for production use.")

if __name__ == "__main__":
    main()