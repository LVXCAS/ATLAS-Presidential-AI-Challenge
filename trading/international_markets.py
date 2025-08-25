"""
Hive Trade International Markets Integration
Comprehensive forex and commodities trading with global market analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CurrencyPair:
    """Currency pair information"""
    base: str
    quote: str
    symbol: str
    pip_value: float
    spread: float
    session_times: Dict[str, Tuple[int, int]]  # Trading session hours

@dataclass
class CommodityInfo:
    """Commodity information"""
    name: str
    symbol: str
    unit: str
    contract_size: float
    tick_size: float
    margin_requirement: float
    trading_hours: Tuple[int, int]

class ForexMarket:
    """Forex market integration and analysis"""
    
    def __init__(self):
        self.currency_pairs = self._initialize_currency_pairs()
        self.economic_calendar = {}
        self.central_bank_rates = self._initialize_cb_rates()
        
    def _initialize_currency_pairs(self) -> Dict[str, CurrencyPair]:
        """Initialize major currency pairs"""
        pairs = {
            'EURUSD': CurrencyPair(
                'EUR', 'USD', 'EURUSD', 0.0001, 0.00015,
                {'london': (8, 17), 'newyork': (13, 22), 'tokyo': (0, 9)}
            ),
            'GBPUSD': CurrencyPair(
                'GBP', 'USD', 'GBPUSD', 0.0001, 0.00018,
                {'london': (8, 17), 'newyork': (13, 22), 'tokyo': (0, 9)}
            ),
            'USDJPY': CurrencyPair(
                'USD', 'JPY', 'USDJPY', 0.01, 0.015,
                {'london': (8, 17), 'newyork': (13, 22), 'tokyo': (0, 9)}
            ),
            'AUDUSD': CurrencyPair(
                'AUD', 'USD', 'AUDUSD', 0.0001, 0.00020,
                {'sydney': (22, 7), 'london': (8, 17), 'newyork': (13, 22)}
            ),
            'USDCAD': CurrencyPair(
                'USD', 'CAD', 'USDCAD', 0.0001, 0.00018,
                {'london': (8, 17), 'newyork': (13, 22), 'tokyo': (0, 9)}
            ),
            'USDCHF': CurrencyPair(
                'USD', 'CHF', 'USDCHF', 0.0001, 0.00020,
                {'london': (8, 17), 'newyork': (13, 22), 'tokyo': (0, 9)}
            )
        }
        return pairs
    
    def _initialize_cb_rates(self) -> Dict[str, float]:
        """Initialize central bank interest rates (mock data)"""
        return {
            'USD': 5.25,  # Federal Reserve
            'EUR': 4.50,  # ECB
            'GBP': 5.25,  # Bank of England
            'JPY': -0.10, # Bank of Japan
            'AUD': 4.35,  # Reserve Bank of Australia
            'CAD': 5.00,  # Bank of Canada
            'CHF': 1.75   # Swiss National Bank
        }
    
    def get_forex_price(self, symbol: str) -> Dict[str, float]:
        """Get current forex price (mock)"""
        base_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2600,
            'USDJPY': 149.50,
            'AUDUSD': 0.6500,
            'USDCAD': 1.3650,
            'USDCHF': 0.9150
        }
        
        base_price = base_prices.get(symbol, 1.0)
        pair = self.currency_pairs.get(symbol)
        
        if not pair:
            return {}
        
        # Add random variation
        variation = np.random.normal(0, 0.005)  # 0.5% volatility
        mid_price = base_price * (1 + variation)
        
        # Calculate bid/ask with spread
        spread_half = pair.spread / 2
        bid = mid_price - spread_half
        ask = mid_price + spread_half
        
        return {
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'mid': mid_price,
            'spread': ask - bid,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_carry_trade_return(self, base_currency: str, quote_currency: str) -> float:
        """Calculate carry trade potential return"""
        base_rate = self.central_bank_rates.get(base_currency, 0)
        quote_rate = self.central_bank_rates.get(quote_currency, 0)
        
        # Carry trade return = interest rate differential
        carry_return = base_rate - quote_rate
        return carry_return
    
    def analyze_currency_strength(self) -> Dict[str, float]:
        """Analyze relative currency strength"""
        
        # Mock currency strength index (0-100)
        strength_scores = {}
        
        for currency in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']:
            # Factors: interest rates, economic indicators, market sentiment
            interest_factor = self.central_bank_rates.get(currency, 0) * 10
            economic_factor = np.random.uniform(-20, 20)  # Mock economic strength
            sentiment_factor = np.random.uniform(-15, 15)  # Mock market sentiment
            
            strength = 50 + interest_factor + economic_factor + sentiment_factor
            strength = max(0, min(100, strength))  # Clamp to 0-100
            
            strength_scores[currency] = strength
        
        return strength_scores
    
    def detect_forex_patterns(self, symbol: str, periods: int = 100) -> Dict[str, Any]:
        """Detect forex-specific patterns"""
        
        # Generate mock historical data
        prices = []
        base_price = self.get_forex_price(symbol)['mid']
        
        for i in range(periods):
            variation = np.random.normal(0, 0.008)  # Daily variation
            price = base_price * (1 + variation)
            prices.append(price)
            base_price = price
        
        df = pd.DataFrame({'price': prices})
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        
        patterns = {
            'trend': 'neutral',
            'support_resistance': [],
            'breakout_potential': False,
            'range_bound': False
        }
        
        # Trend detection
        if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            if df['sma_20'].iloc[-5] <= df['sma_50'].iloc[-5]:
                patterns['trend'] = 'bullish_crossover'
            else:
                patterns['trend'] = 'bullish'
        elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
            if df['sma_20'].iloc[-5] >= df['sma_50'].iloc[-5]:
                patterns['trend'] = 'bearish_crossover'
            else:
                patterns['trend'] = 'bearish'
        
        # Support/Resistance levels
        highs = df['price'].rolling(10).max()
        lows = df['price'].rolling(10).min()
        
        resistance_levels = highs.dropna().unique()[-3:]  # Last 3 resistance levels
        support_levels = lows.dropna().unique()[-3:]      # Last 3 support levels
        
        patterns['support_resistance'] = {
            'resistance': resistance_levels.tolist(),
            'support': support_levels.tolist()
        }
        
        # Range-bound detection
        recent_high = df['price'].tail(20).max()
        recent_low = df['price'].tail(20).min()
        range_size = (recent_high - recent_low) / df['price'].iloc[-1]
        
        patterns['range_bound'] = range_size < 0.02  # Less than 2% range
        patterns['breakout_potential'] = range_size < 0.015  # Very tight range
        
        return patterns
    
    def get_economic_events(self) -> List[Dict[str, Any]]:
        """Get upcoming economic events (mock data)"""
        
        events = [
            {
                'date': datetime.now() + timedelta(days=1),
                'time': '14:30',
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'High',
                'previous': '150K',
                'forecast': '180K'
            },
            {
                'date': datetime.now() + timedelta(days=2),
                'time': '12:30',
                'currency': 'EUR',
                'event': 'ECB Interest Rate Decision',
                'impact': 'High',
                'previous': '4.50%',
                'forecast': '4.50%'
            },
            {
                'date': datetime.now() + timedelta(days=3),
                'time': '09:30',
                'currency': 'GBP',
                'event': 'UK CPI',
                'impact': 'Medium',
                'previous': '4.0%',
                'forecast': '3.8%'
            }
        ]
        
        return events

class CommoditiesMarket:
    """Commodities market integration and analysis"""
    
    def __init__(self):
        self.commodities = self._initialize_commodities()
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
    def _initialize_commodities(self) -> Dict[str, CommodityInfo]:
        """Initialize major commodities"""
        commodities = {
            'GOLD': CommodityInfo(
                'Gold', 'XAUUSD', 'troy ounces', 100, 0.01, 2000, (0, 24)
            ),
            'SILVER': CommodityInfo(
                'Silver', 'XAGUSD', 'troy ounces', 5000, 0.001, 5000, (0, 24)
            ),
            'CRUDE_OIL': CommodityInfo(
                'Crude Oil', 'USOIL', 'barrels', 1000, 0.01, 3000, (0, 24)
            ),
            'NATURAL_GAS': CommodityInfo(
                'Natural Gas', 'NGAS', 'mmbtu', 10000, 0.001, 1500, (0, 24)
            ),
            'WHEAT': CommodityInfo(
                'Wheat', 'WHEAT', 'bushels', 5000, 0.25, 1000, (1, 20)
            ),
            'CORN': CommodityInfo(
                'Corn', 'CORN', 'bushels', 5000, 0.25, 1200, (1, 20)
            ),
            'COPPER': CommodityInfo(
                'Copper', 'COPPER', 'pounds', 25000, 0.0005, 2500, (0, 24)
            )
        }
        return commodities
    
    def _initialize_seasonal_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize seasonal trading patterns for commodities"""
        patterns = {
            'GOLD': {
                'Q1': 'neutral', 'Q2': 'weak', 'Q3': 'strong', 'Q4': 'strong'
            },
            'CRUDE_OIL': {
                'Q1': 'strong', 'Q2': 'strong', 'Q3': 'neutral', 'Q4': 'weak'
            },
            'NATURAL_GAS': {
                'Q1': 'strong', 'Q2': 'weak', 'Q3': 'neutral', 'Q4': 'strong'
            },
            'WHEAT': {
                'Q1': 'neutral', 'Q2': 'strong', 'Q3': 'strong', 'Q4': 'weak'
            },
            'CORN': {
                'Q1': 'weak', 'Q2': 'strong', 'Q3': 'strong', 'Q4': 'neutral'
            }
        }
        return patterns
    
    def get_commodity_price(self, symbol: str) -> Dict[str, float]:
        """Get current commodity price (mock)"""
        base_prices = {
            'GOLD': 2000.00,
            'SILVER': 25.00,
            'CRUDE_OIL': 80.00,
            'NATURAL_GAS': 3.50,
            'WHEAT': 650.00,
            'CORN': 520.00,
            'COPPER': 3.80
        }
        
        base_price = base_prices.get(symbol, 100)
        commodity = self.commodities.get(symbol)
        
        if not commodity:
            return {}
        
        # Add random variation with commodity-specific volatility
        volatility = {
            'GOLD': 0.015, 'SILVER': 0.025, 'CRUDE_OIL': 0.030,
            'NATURAL_GAS': 0.050, 'WHEAT': 0.020, 'CORN': 0.018, 'COPPER': 0.022
        }
        
        variation = np.random.normal(0, volatility.get(symbol, 0.02))
        current_price = base_price * (1 + variation)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'change_24h': variation * 100,
            'volume': np.random.uniform(10000, 100000),
            'open_interest': np.random.uniform(50000, 500000),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_supply_demand(self, symbol: str) -> Dict[str, Any]:
        """Analyze supply and demand factors for commodity"""
        
        # Mock supply/demand analysis
        supply_factors = {
            'GOLD': ['Mining output', 'Central bank sales', 'Recycling'],
            'CRUDE_OIL': ['OPEC production', 'US shale', 'Geopolitical risks'],
            'WHEAT': ['Weather conditions', 'Acreage planted', 'Export restrictions'],
            'NATURAL_GAS': ['Production levels', 'Storage levels', 'Pipeline capacity']
        }
        
        demand_factors = {
            'GOLD': ['Jewelry demand', 'Investment demand', 'Industrial use'],
            'CRUDE_OIL': ['Economic growth', 'Transportation', 'Industrial demand'],
            'WHEAT': ['Population growth', 'Biofuel demand', 'Feed demand'],
            'NATURAL_GAS': ['Power generation', 'Industrial use', 'Heating demand']
        }
        
        # Mock current conditions
        supply_outlook = np.random.choice(['tight', 'adequate', 'surplus'])
        demand_outlook = np.random.choice(['weak', 'steady', 'strong'])
        
        # Calculate supply/demand balance score (-100 to 100)
        supply_score = {'tight': 30, 'adequate': 0, 'surplus': -30}[supply_outlook]
        demand_score = {'weak': -20, 'steady': 0, 'strong': 20}[demand_outlook]
        balance_score = supply_score + demand_score
        
        return {
            'symbol': symbol,
            'supply_factors': supply_factors.get(symbol, []),
            'demand_factors': demand_factors.get(symbol, []),
            'supply_outlook': supply_outlook,
            'demand_outlook': demand_outlook,
            'balance_score': balance_score,
            'seasonal_bias': self.get_seasonal_bias(symbol)
        }
    
    def get_seasonal_bias(self, symbol: str) -> str:
        """Get current seasonal bias for commodity"""
        current_quarter = f"Q{((datetime.now().month - 1) // 3) + 1}"
        return self.seasonal_patterns.get(symbol, {}).get(current_quarter, 'neutral')
    
    def calculate_contango_backwardation(self, symbol: str) -> Dict[str, Any]:
        """Calculate contango/backwardation for commodity futures"""
        
        # Mock futures curve data
        months_out = [1, 2, 3, 6, 9, 12]
        spot_price = self.get_commodity_price(symbol)['price']
        
        futures_prices = []
        for month in months_out:
            # Add storage costs, interest, convenience yield
            storage_cost = 0.002 * month  # 0.2% per month
            convenience_yield = np.random.uniform(-0.01, 0.01)  # -1% to 1%
            
            future_price = spot_price * (1 + storage_cost + convenience_yield)
            futures_prices.append(future_price)
        
        # Determine market structure
        if futures_prices[-1] > spot_price:
            structure = 'contango'
            intensity = (futures_prices[-1] - spot_price) / spot_price
        else:
            structure = 'backwardation'
            intensity = (spot_price - futures_prices[-1]) / spot_price
        
        return {
            'symbol': symbol,
            'structure': structure,
            'intensity': intensity,
            'spot_price': spot_price,
            'futures_curve': dict(zip(months_out, futures_prices)),
            'carry_cost': futures_prices[0] - spot_price
        }

class InternationalMarketsAnalyzer:
    """Comprehensive international markets analysis"""
    
    def __init__(self):
        self.forex_market = ForexMarket()
        self.commodities_market = CommoditiesMarket()
        
    def analyze_forex_markets(self, currency_pairs: List[str]) -> Dict[str, Any]:
        """Comprehensive forex market analysis"""
        
        print("\nFOREX MARKET ANALYSIS")
        print("="*22)
        
        forex_analysis = {}
        
        # Currency strength analysis
        currency_strength = self.forex_market.analyze_currency_strength()
        print("\nCurrency Strength Index:")
        for currency, strength in currency_strength.items():
            print(f"  {currency}: {strength:.1f}/100")
        
        # Individual pair analysis
        for pair in currency_pairs:
            print(f"\nAnalyzing {pair}...")
            
            # Get current price
            price_data = self.forex_market.get_forex_price(pair)
            
            # Carry trade analysis
            pair_info = self.forex_market.currency_pairs.get(pair)
            if pair_info:
                carry_return = self.forex_market.calculate_carry_trade_return(
                    pair_info.base, pair_info.quote
                )
            else:
                carry_return = 0
            
            # Pattern analysis
            patterns = self.forex_market.detect_forex_patterns(pair)
            
            # Trading signal
            signal = self._generate_forex_signal(price_data, patterns, carry_return, currency_strength)
            
            forex_analysis[pair] = {
                'price_data': price_data,
                'carry_return': carry_return,
                'patterns': patterns,
                'signal': signal,
                'base_strength': currency_strength.get(pair[:3], 50),
                'quote_strength': currency_strength.get(pair[3:], 50)
            }
            
            print(f"  Price: {price_data.get('mid', 0):.5f}")
            print(f"  Carry Return: {carry_return:.2f}%")
            print(f"  Signal: {signal}")
        
        # Economic events
        economic_events = self.forex_market.get_economic_events()
        
        return {
            'currency_strength': currency_strength,
            'pair_analysis': forex_analysis,
            'economic_events': [
                {**event, 'date': event['date'].isoformat()} 
                for event in economic_events
            ]
        }
    
    def analyze_commodities_markets(self, commodities: List[str]) -> Dict[str, Any]:
        """Comprehensive commodities market analysis"""
        
        print("\nCOMMODITIES MARKET ANALYSIS")
        print("="*28)
        
        commodities_analysis = {}
        
        for commodity in commodities:
            print(f"\nAnalyzing {commodity}...")
            
            # Get current price
            price_data = self.commodities_market.get_commodity_price(commodity)
            
            # Supply/demand analysis
            supply_demand = self.commodities_market.analyze_supply_demand(commodity)
            
            # Futures curve analysis
            curve_analysis = self.commodities_market.calculate_contango_backwardation(commodity)
            
            # Trading signal
            signal = self._generate_commodity_signal(price_data, supply_demand, curve_analysis)
            
            commodities_analysis[commodity] = {
                'price_data': price_data,
                'supply_demand': supply_demand,
                'curve_analysis': curve_analysis,
                'signal': signal
            }
            
            print(f"  Price: ${price_data.get('price', 0):.2f}")
            print(f"  Supply/Demand Balance: {supply_demand.get('balance_score', 0)}")
            print(f"  Seasonal Bias: {supply_demand.get('seasonal_bias', 'neutral')}")
            print(f"  Signal: {signal}")
        
        return commodities_analysis
    
    def _generate_forex_signal(self, price_data: Dict, patterns: Dict, 
                             carry_return: float, currency_strength: Dict) -> str:
        """Generate forex trading signal"""
        
        signal_score = 0
        
        # Trend factor
        if patterns['trend'] == 'bullish':
            signal_score += 2
        elif patterns['trend'] == 'bearish':
            signal_score -= 2
        elif 'crossover' in patterns['trend']:
            signal_score += 3 if 'bullish' in patterns['trend'] else -3
        
        # Carry trade factor
        if carry_return > 2:
            signal_score += 1
        elif carry_return < -2:
            signal_score -= 1
        
        # Range/breakout factor
        if patterns['breakout_potential']:
            signal_score += 1 if signal_score > 0 else -1
        
        # Currency strength factor (simplified)
        # Would need pair-specific logic in real implementation
        
        # Convert score to signal
        if signal_score >= 3:
            return 'STRONG_BUY'
        elif signal_score >= 1:
            return 'BUY'
        elif signal_score <= -3:
            return 'STRONG_SELL'
        elif signal_score <= -1:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_commodity_signal(self, price_data: Dict, supply_demand: Dict, 
                                 curve_analysis: Dict) -> str:
        """Generate commodity trading signal"""
        
        signal_score = 0
        
        # Supply/demand balance
        balance = supply_demand.get('balance_score', 0)
        if balance > 20:
            signal_score += 2
        elif balance < -20:
            signal_score -= 2
        elif balance > 10:
            signal_score += 1
        elif balance < -10:
            signal_score -= 1
        
        # Seasonal bias
        seasonal = supply_demand.get('seasonal_bias', 'neutral')
        if seasonal == 'strong':
            signal_score += 1
        elif seasonal == 'weak':
            signal_score -= 1
        
        # Futures curve structure
        structure = curve_analysis.get('structure')
        intensity = curve_analysis.get('intensity', 0)
        
        if structure == 'backwardation' and intensity > 0.05:
            signal_score += 2  # Strong backwardation is bullish
        elif structure == 'contango' and intensity > 0.05:
            signal_score -= 1  # Strong contango can be bearish
        
        # Convert score to signal
        if signal_score >= 3:
            return 'STRONG_BUY'
        elif signal_score >= 1:
            return 'BUY'
        elif signal_score <= -3:
            return 'STRONG_SELL'
        elif signal_score <= -1:
            return 'SELL'
        else:
            return 'HOLD'
    
    def run_international_analysis(self) -> Dict[str, Any]:
        """Run comprehensive international markets analysis"""
        
        print("HIVE TRADE INTERNATIONAL MARKETS ANALYSIS")
        print("="*42)
        
        # Major forex pairs
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
        # Major commodities
        commodities = ['GOLD', 'SILVER', 'CRUDE_OIL', 'NATURAL_GAS', 'WHEAT', 'COPPER']
        
        # Run analysis
        forex_analysis = self.analyze_forex_markets(forex_pairs)
        commodities_analysis = self.analyze_commodities_markets(commodities)
        
        # Cross-asset correlations (simplified)
        correlations = self._calculate_cross_asset_correlations()
        
        return {
            'forex_analysis': forex_analysis,
            'commodities_analysis': commodities_analysis,
            'cross_asset_correlations': correlations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_cross_asset_correlations(self) -> Dict[str, float]:
        """Calculate correlations between different asset classes"""
        
        # Mock correlation data (would use real price data in practice)
        correlations = {
            'USD_index_vs_gold': -0.65,
            'oil_vs_cad': 0.72,
            'gold_vs_real_rates': -0.58,
            'eur_vs_dxy': -0.85,
            'copper_vs_aud': 0.63,
            'yen_vs_risk_assets': -0.45
        }
        
        return correlations

def main():
    """Run international markets analysis"""
    
    # Initialize analyzer
    analyzer = InternationalMarketsAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_international_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"international_markets_{timestamp}.json"
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if isinstance(v, (np.integer, np.floating)):
                    cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (list, dict)):
                    cleaned[k] = clean_for_json(v)
                elif isinstance(v, np.ndarray):
                    cleaned[k] = v.tolist()
                elif isinstance(v, (np.bool_, bool)):
                    cleaned[k] = bool(v)
                else:
                    try:
                        json.dumps(v)  # Test serializable
                        cleaned[k] = v
                    except:
                        cleaned[k] = str(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    clean_results = clean_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\nINTERNATIONAL MARKETS ANALYSIS COMPLETE!")
    print("="*42)
    
    return results

if __name__ == "__main__":
    main()