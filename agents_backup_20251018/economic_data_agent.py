"""
Economic Data Agent - FRED API Integration
Provides macro economic analysis for better options trading decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)

class EconomicDataAgent:
    """Economic data analysis using FRED API for macro-aware trading"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY', '98e96c3261987f1c116c1506e6dde103')
        self.fred = None
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour cache
        
        # Initialize FRED connection
        if FRED_AVAILABLE and self.api_key:
            try:
                self.fred = Fred(api_key=self.api_key)
                logger.info("FRED API initialized successfully")
            except Exception as e:
                logger.error(f"FRED API initialization failed: {e}")
                self.fred = None
        else:
            logger.warning("FRED API not available - economic analysis will be limited")
    
    async def get_comprehensive_economic_analysis(self) -> Dict:
        """Get comprehensive economic analysis affecting options markets"""
        
        if not self.fred:
            return self._get_default_economic_data()
        
        # Check cache
        cache_key = "comprehensive_economic"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            analysis = {
                'timestamp': datetime.now(),
                'market_regime': 'NEUTRAL',
                'volatility_environment': 'NORMAL',
                'options_strategy_bias': 'NEUTRAL',
                'fed_policy': {},
                'inflation_data': {},
                'economic_stress': {},
                'options_implications': {}
            }
            
            # Get key economic indicators
            analysis['fed_policy'] = await self._get_fed_policy_data()
            analysis['inflation_data'] = await self._get_inflation_data()
            analysis['economic_stress'] = await self._get_stress_indicators()
            
            # Synthesize options trading implications
            analysis['options_implications'] = self._analyze_options_implications(analysis)
            analysis['market_regime'] = self._determine_market_regime(analysis)
            analysis['volatility_environment'] = self._assess_volatility_environment(analysis)
            analysis['options_strategy_bias'] = self._get_strategy_bias(analysis)
            
            # Cache results
            self.cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Economic analysis error: {e}")
            return self._get_default_economic_data()
    
    async def _get_fed_policy_data(self) -> Dict:
        """Get Federal Reserve policy indicators"""
        fed_data = {
            'fed_funds_rate': 0.0,
            'fed_funds_change_6m': 0.0,
            'policy_stance': 'NEUTRAL',
            'next_meeting_impact': 'LOW'
        }
        
        try:
            # Federal Funds Rate
            fed_funds = self.fred.get_series('FEDFUNDS', limit=12)  # Last 12 months
            if not fed_funds.empty:
                fed_data['fed_funds_rate'] = float(fed_funds.iloc[-1])
                
                # 6-month change
                if len(fed_funds) >= 6:
                    fed_data['fed_funds_change_6m'] = fed_data['fed_funds_rate'] - float(fed_funds.iloc[-7])
                
                # Policy stance
                if fed_data['fed_funds_change_6m'] > 0.5:
                    fed_data['policy_stance'] = 'TIGHTENING'
                elif fed_data['fed_funds_change_6m'] < -0.5:
                    fed_data['policy_stance'] = 'EASING'
                else:
                    fed_data['policy_stance'] = 'NEUTRAL'
                
                # Meeting impact assessment
                if abs(fed_data['fed_funds_change_6m']) > 1.0:
                    fed_data['next_meeting_impact'] = 'HIGH'
                elif abs(fed_data['fed_funds_change_6m']) > 0.25:
                    fed_data['next_meeting_impact'] = 'MODERATE'
                else:
                    fed_data['next_meeting_impact'] = 'LOW'
            
        except Exception as e:
            logger.warning(f"Fed policy data error: {e}")
        
        return fed_data
    
    async def _get_inflation_data(self) -> Dict:
        """Get inflation indicators"""
        inflation_data = {
            'cpi_current': 0.0,
            'cpi_yoy': 0.0,
            'core_cpi_yoy': 0.0,
            'inflation_trend': 'STABLE',
            'fed_target_distance': 0.0
        }
        
        try:
            # CPI Data
            cpi = self.fred.get_series('CPIAUCSL', limit=24)  # Last 24 months
            if not cpi.empty and len(cpi) >= 12:
                inflation_data['cpi_current'] = float(cpi.iloc[-1])
                inflation_data['cpi_yoy'] = ((cpi.iloc[-1] / cpi.iloc[-13]) - 1) * 100
                
                # Core CPI
                core_cpi = self.fred.get_series('CPILFESL', limit=24)
                if not core_cpi.empty and len(core_cpi) >= 12:
                    inflation_data['core_cpi_yoy'] = ((core_cpi.iloc[-1] / core_cpi.iloc[-13]) - 1) * 100
                
                # Distance from Fed 2% target
                inflation_data['fed_target_distance'] = inflation_data['cpi_yoy'] - 2.0
                
                # Inflation trend
                if len(cpi) >= 6:
                    recent_trend = ((cpi.iloc[-1] / cpi.iloc[-7]) - 1) * 100 * 2  # Annualized 6-month
                    if recent_trend > inflation_data['cpi_yoy'] + 0.5:
                        inflation_data['inflation_trend'] = 'ACCELERATING'
                    elif recent_trend < inflation_data['cpi_yoy'] - 0.5:
                        inflation_data['inflation_trend'] = 'DECELERATING'
                    else:
                        inflation_data['inflation_trend'] = 'STABLE'
        
        except Exception as e:
            logger.warning(f"Inflation data error: {e}")
        
        return inflation_data
    
    async def _get_stress_indicators(self) -> Dict:
        """Get economic stress and recession indicators"""
        stress_data = {
            'recession_probability': 'LOW',
            'leading_indicators_trend': 'STABLE',
            'consumer_confidence': 'NORMAL',
            'credit_spreads': {'high_yield': 0.0, 'investment_grade': 0.0, 'stress_level': 'LOW'},
            'dollar_strength': {'index_value': 0.0, 'trend': 'NEUTRAL'},
            'yield_curve': {'spread_10y2y': 0.0, 'shape': 'NORMAL'}
        }
        
        try:
            # Leading Economic Indicators
            lei = self.fred.get_series('USSLIND', limit=12)
            if not lei.empty and len(lei) >= 6:
                recent_change = ((lei.iloc[-1] / lei.iloc[-7]) - 1) * 100
                if recent_change < -2:
                    stress_data['leading_indicators_trend'] = 'DETERIORATING'
                elif recent_change < -0.5:
                    stress_data['leading_indicators_trend'] = 'WEAKENING'
                elif recent_change > 1:
                    stress_data['leading_indicators_trend'] = 'IMPROVING'
                else:
                    stress_data['leading_indicators_trend'] = 'STABLE'
                
                # Simple recession probability based on LEI trend
                if recent_change < -3:
                    stress_data['recession_probability'] = 'HIGH'
                elif recent_change < -1:
                    stress_data['recession_probability'] = 'MODERATE'
                else:
                    stress_data['recession_probability'] = 'LOW'
            
            # Credit Spreads - High Yield (BAMLH0A0HYM2)
            hy_spread = self.fred.get_series('BAMLH0A0HYM2', limit=6)
            if not hy_spread.empty:
                stress_data['credit_spreads']['high_yield'] = float(hy_spread.iloc[-1])
                
                # Credit stress assessment
                if stress_data['credit_spreads']['high_yield'] > 800:  # 8%
                    stress_data['credit_spreads']['stress_level'] = 'EXTREME'
                elif stress_data['credit_spreads']['high_yield'] > 500:  # 5%
                    stress_data['credit_spreads']['stress_level'] = 'HIGH'
                elif stress_data['credit_spreads']['high_yield'] > 300:  # 3%
                    stress_data['credit_spreads']['stress_level'] = 'MODERATE'
                else:
                    stress_data['credit_spreads']['stress_level'] = 'LOW'
                    
                # Update recession probability based on credit spreads
                if stress_data['credit_spreads']['stress_level'] in ['EXTREME', 'HIGH']:
                    stress_data['recession_probability'] = 'HIGH'
            
            # Dollar Index (DTWEXBGS)
            dollar_idx = self.fred.get_series('DTWEXBGS', limit=12)
            if not dollar_idx.empty:
                stress_data['dollar_strength']['index_value'] = float(dollar_idx.iloc[-1])
                
                # Dollar trend over 3 months
                if len(dollar_idx) >= 3:
                    dollar_change = ((dollar_idx.iloc[-1] / dollar_idx.iloc[-4]) - 1) * 100
                    if dollar_change > 3:
                        stress_data['dollar_strength']['trend'] = 'STRENGTHENING'
                    elif dollar_change < -3:
                        stress_data['dollar_strength']['trend'] = 'WEAKENING'
                    else:
                        stress_data['dollar_strength']['trend'] = 'NEUTRAL'
            
            # Yield Curve - 10Y vs 2Y spread (GS10 - GS2)
            gs10 = self.fred.get_series('GS10', limit=3)
            gs2 = self.fred.get_series('GS2', limit=3)
            
            if not gs10.empty and not gs2.empty:
                spread = float(gs10.iloc[-1]) - float(gs2.iloc[-1])
                stress_data['yield_curve']['spread_10y2y'] = spread
                
                if spread < 0:
                    stress_data['yield_curve']['shape'] = 'INVERTED'
                    stress_data['recession_probability'] = 'HIGH'  # Inverted curve is strong recession signal
                elif spread < 0.5:
                    stress_data['yield_curve']['shape'] = 'FLAT'
                elif spread > 2.5:
                    stress_data['yield_curve']['shape'] = 'STEEP'
                else:
                    stress_data['yield_curve']['shape'] = 'NORMAL'
        
        except Exception as e:
            logger.warning(f"Stress indicators error: {e}")
        
        return stress_data
    
    def _determine_market_regime(self, analysis: Dict) -> str:
        """Determine overall market regime using comprehensive economic data"""
        fed_policy = analysis['fed_policy']['policy_stance']
        inflation_trend = analysis['inflation_data']['inflation_trend']
        stress_level = analysis['economic_stress']['recession_probability']
        
        # Get additional stress indicators
        stress_data = analysis['economic_stress']
        credit_stress = stress_data.get('credit_spreads', {}).get('stress_level', 'LOW')
        yield_curve = stress_data.get('yield_curve', {}).get('shape', 'NORMAL')
        dollar_trend = stress_data.get('dollar_strength', {}).get('trend', 'NEUTRAL')
        
        # Crisis conditions - multiple stress signals
        if (stress_level == 'HIGH' or 
            credit_stress in ['EXTREME', 'HIGH'] or 
            yield_curve == 'INVERTED'):
            return 'CRISIS'
        
        # Financial stress but not full crisis
        elif credit_stress == 'MODERATE' or yield_curve == 'FLAT':
            return 'FINANCIAL_STRESS'
        
        # Fed policy driven regimes
        elif fed_policy == 'TIGHTENING' and inflation_trend == 'ACCELERATING':
            return 'HAWKISH_FED'
        elif fed_policy == 'EASING':
            return 'DOVISH_FED'
        
        # Dollar strength implications
        elif dollar_trend == 'STRENGTHENING' and fed_policy == 'TIGHTENING':
            return 'STRONG_DOLLAR'
        
        else:
            return 'NEUTRAL'
    
    def _assess_volatility_environment(self, analysis: Dict) -> str:
        """Assess current volatility environment"""
        stress = analysis['economic_stress']
        fed_policy = analysis['fed_policy']
        
        if stress['recession_probability'] == 'HIGH':
            return 'HIGH_VOLATILITY'
        elif fed_policy['next_meeting_impact'] == 'HIGH':
            return 'ELEVATED_VOLATILITY'
        else:
            return 'NORMAL_VOLATILITY'
    
    def _get_strategy_bias(self, analysis: Dict) -> str:
        """Get options strategy bias based on economic conditions"""
        regime = analysis['market_regime']
        vol_env = analysis['volatility_environment']
        
        # Get additional context
        stress_data = analysis['economic_stress']
        credit_stress = stress_data.get('credit_spreads', {}).get('stress_level', 'LOW')
        dollar_trend = stress_data.get('dollar_strength', {}).get('trend', 'NEUTRAL')
        
        if regime == 'CRISIS':
            return 'PROTECTIVE_PUTS'
        elif regime == 'FINANCIAL_STRESS':
            return 'DEFENSIVE_SPREADS'
        elif vol_env == 'HIGH_VOLATILITY' and credit_stress == 'LOW':
            return 'VOLATILITY_SELLING'
        elif regime == 'HAWKISH_FED':
            return 'BEAR_PUT_SPREADS'
        elif regime == 'DOVISH_FED':
            return 'BULL_CALL_SPREADS'
        elif regime == 'STRONG_DOLLAR':
            return 'EXPORT_SECTOR_PUTS'  # Strong dollar hurts exporters
        else:
            return 'NEUTRAL'
    
    def _analyze_options_implications(self, analysis: Dict) -> Dict:
        """Analyze implications for options trading"""
        return {
            'volatility_bias': 'NEUTRAL',
            'term_structure_bias': 'NEUTRAL',
            'event_risk_assessment': 'NORMAL',
            'liquidity_environment': 'NORMAL'
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _get_default_economic_data(self) -> Dict:
        """Return default economic data when FRED is unavailable"""
        return {
            'timestamp': datetime.now(),
            'market_regime': 'NEUTRAL',
            'volatility_environment': 'NORMAL',
            'options_strategy_bias': 'NEUTRAL',
            'fed_policy': {'policy_stance': 'NEUTRAL', 'next_meeting_impact': 'LOW'},
            'inflation_data': {'cpi_yoy': 3.0, 'inflation_trend': 'STABLE'},
            'economic_stress': {'recession_probability': 'LOW'},
            'options_implications': {'volatility_bias': 'NEUTRAL', 'event_risk_assessment': 'NORMAL'}
        }

# Singleton instance  
economic_data_agent = EconomicDataAgent()
