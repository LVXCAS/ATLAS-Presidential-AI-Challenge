"""
Hive Trade Advanced Options Trading Strategies
Comprehensive options trading system with multiple strategies
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Black-Scholes implementation for option pricing
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math

class BlackScholesModel:
    """Black-Scholes option pricing model"""
    
    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call') -> float:
        """
        Calculate option price using Black-Scholes formula
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def greeks(S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (theta1 + theta2) / 365  # Per day
        else:
            theta2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (theta1 + theta2) / 365  # Per day
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class OptionsStrategy:
    """Base class for options strategies"""
    
    def __init__(self, symbol: str, risk_free_rate: float = 0.05):
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.bs_model = BlackScholesModel()
        
    def get_stock_data(self, period: str = '1y') -> pd.DataFrame:
        """Get stock price data and calculate volatility"""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period)
        
        # Calculate historical volatility (annualized)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        df['volatility'] = volatility
        return df
    
    def get_options_chain(self) -> Dict[str, pd.DataFrame]:
        """Get options chain data"""
        try:
            ticker = yf.Ticker(self.symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {}
            
            chains = {}
            for date in options_dates[:4]:  # Get first 4 expiration dates
                try:
                    chain = ticker.option_chain(date)
                    chains[date] = {
                        'calls': chain.calls,
                        'puts': chain.puts
                    }
                except:
                    continue
            
            return chains
        except:
            return {}
    
    def calculate_days_to_expiration(self, expiration_date: str) -> float:
        """Calculate days to expiration"""
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        days = (exp_date - today).days
        return max(days / 365.0, 1/365.0)  # At least 1 day

class CoveredCallStrategy(OptionsStrategy):
    """Covered Call Strategy Implementation"""
    
    def analyze_covered_calls(self, min_premium: float = 1.0, 
                             max_days: int = 45) -> List[Dict[str, Any]]:
        """Analyze covered call opportunities"""
        
        print(f"Analyzing covered call opportunities for {self.symbol}...")
        
        stock_data = self.get_stock_data()
        current_price = stock_data['Close'].iloc[-1]
        volatility = stock_data['volatility'].iloc[-1]
        
        options_chains = self.get_options_chain()
        opportunities = []
        
        for exp_date, chain_data in options_chains.items():
            time_to_exp = self.calculate_days_to_expiration(exp_date)
            
            if time_to_exp * 365 > max_days:
                continue
            
            calls = chain_data['calls']
            
            # Filter for out-of-the-money calls
            otm_calls = calls[calls['strike'] > current_price * 1.02]  # At least 2% OTM
            
            for _, option in otm_calls.iterrows():
                strike = option['strike']
                bid = option['bid']
                ask = option['ask']
                
                if bid < min_premium:
                    continue
                
                # Calculate metrics
                premium = (bid + ask) / 2
                
                # Theoretical price using Black-Scholes
                theoretical_price = self.bs_model.option_price(
                    current_price, strike, time_to_exp, self.risk_free_rate, volatility, 'call'
                )
                
                # Greeks
                greeks = self.bs_model.greeks(
                    current_price, strike, time_to_exp, self.risk_free_rate, volatility, 'call'
                )
                
                # Strategy metrics
                max_profit = premium + (strike - current_price)
                max_loss = current_price - premium  # If stock goes to 0
                breakeven = current_price - premium
                
                # Annualized return calculations
                return_if_called = ((strike + premium) / current_price - 1) * (365 / (time_to_exp * 365))
                return_if_unchanged = (premium / current_price) * (365 / (time_to_exp * 365))
                
                # Probability calculations
                prob_itm = norm.cdf((np.log(strike / current_price) - 
                                   (self.risk_free_rate - 0.5 * volatility**2) * time_to_exp) / 
                                  (volatility * np.sqrt(time_to_exp)))
                
                opportunities.append({
                    'expiration': exp_date,
                    'strike': strike,
                    'current_price': current_price,
                    'bid': bid,
                    'ask': ask,
                    'premium': premium,
                    'theoretical_price': theoretical_price,
                    'days_to_exp': int(time_to_exp * 365),
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'breakeven': breakeven,
                    'return_if_called_annual': return_if_called,
                    'return_if_unchanged_annual': return_if_unchanged,
                    'prob_profit': 1 - prob_itm,
                    'delta': greeks['delta'],
                    'theta': greeks['theta'],
                    'implied_vol': volatility,  # Simplified - would need IV calculation
                    'volume': option.get('volume', 0),
                    'open_interest': option.get('openInterest', 0)
                })
        
        # Sort by return if unchanged (annualized)
        opportunities.sort(key=lambda x: x['return_if_unchanged_annual'], reverse=True)
        
        return opportunities
    
    def recommend_covered_call(self, opportunities: List[Dict[str, Any]], 
                              max_risk_tolerance: float = 0.1) -> Optional[Dict[str, Any]]:
        """Recommend best covered call based on risk tolerance"""
        
        if not opportunities:
            return None
        
        # Filter by risk tolerance (probability of being called)
        suitable_opportunities = [
            opp for opp in opportunities 
            if opp['prob_profit'] >= (1 - max_risk_tolerance)
        ]
        
        if not suitable_opportunities:
            return opportunities[0] if opportunities else None
        
        # Return highest return opportunity within risk tolerance
        return suitable_opportunities[0]

class ProtectivePutStrategy(OptionsStrategy):
    """Protective Put Strategy Implementation"""
    
    def analyze_protective_puts(self, protection_level: float = 0.95,
                               max_days: int = 90) -> List[Dict[str, Any]]:
        """Analyze protective put opportunities"""
        
        print(f"Analyzing protective put opportunities for {self.symbol}...")
        
        stock_data = self.get_stock_data()
        current_price = stock_data['Close'].iloc[-1]
        volatility = stock_data['volatility'].iloc[-1]
        
        options_chains = self.get_options_chain()
        opportunities = []
        
        target_strike = current_price * protection_level
        
        for exp_date, chain_data in options_chains.items():
            time_to_exp = self.calculate_days_to_expiration(exp_date)
            
            if time_to_exp * 365 > max_days:
                continue
            
            puts = chain_data['puts']
            
            # Find puts near target strike
            suitable_puts = puts[
                (puts['strike'] >= target_strike * 0.95) & 
                (puts['strike'] <= target_strike * 1.05)
            ]
            
            for _, option in suitable_puts.iterrows():
                strike = option['strike']
                bid = option['bid']
                ask = option['ask']
                
                premium = (bid + ask) / 2
                
                if premium <= 0:
                    continue
                
                # Theoretical price
                theoretical_price = self.bs_model.option_price(
                    current_price, strike, time_to_exp, self.risk_free_rate, volatility, 'put'
                )
                
                # Greeks
                greeks = self.bs_model.greeks(
                    current_price, strike, time_to_exp, self.risk_free_rate, volatility, 'put'
                )
                
                # Protection metrics
                protection_cost = premium / current_price
                protected_value = strike
                max_loss = current_price - strike + premium
                breakeven_up = current_price + premium
                
                # Insurance value
                insurance_annual_cost = protection_cost * (365 / (time_to_exp * 365))
                
                opportunities.append({
                    'expiration': exp_date,
                    'strike': strike,
                    'current_price': current_price,
                    'bid': bid,
                    'ask': ask,
                    'premium': premium,
                    'theoretical_price': theoretical_price,
                    'days_to_exp': int(time_to_exp * 365),
                    'protection_level': strike / current_price,
                    'protection_cost_pct': protection_cost,
                    'insurance_annual_cost_pct': insurance_annual_cost,
                    'max_loss': max_loss,
                    'max_loss_pct': max_loss / current_price,
                    'breakeven_up': breakeven_up,
                    'delta': greeks['delta'],
                    'theta': greeks['theta'],
                    'volume': option.get('volume', 0),
                    'open_interest': option.get('openInterest', 0)
                })
        
        # Sort by cost-effectiveness (lowest annual insurance cost)
        opportunities.sort(key=lambda x: x['insurance_annual_cost_pct'])
        
        return opportunities

class SpreadStrategies(OptionsStrategy):
    """Bull/Bear Spread Strategy Implementation"""
    
    def analyze_bull_call_spreads(self, max_width: float = 10.0,
                                 max_days: int = 45) -> List[Dict[str, Any]]:
        """Analyze bull call spread opportunities"""
        
        print(f"Analyzing bull call spread opportunities for {self.symbol}...")
        
        stock_data = self.get_stock_data()
        current_price = stock_data['Close'].iloc[-1]
        volatility = stock_data['volatility'].iloc[-1]
        
        options_chains = self.get_options_chain()
        opportunities = []
        
        for exp_date, chain_data in options_chains.items():
            time_to_exp = self.calculate_days_to_expiration(exp_date)
            
            if time_to_exp * 365 > max_days:
                continue
            
            calls = chain_data['calls'].sort_values('strike')
            
            # Create spreads
            for i, long_call in calls.iterrows():
                long_strike = long_call['strike']
                long_premium = (long_call['bid'] + long_call['ask']) / 2
                
                if long_strike < current_price * 0.95:  # Skip deep ITM
                    continue
                
                # Find suitable short strikes
                short_calls = calls[
                    (calls['strike'] > long_strike) & 
                    (calls['strike'] <= long_strike + max_width)
                ]
                
                for j, short_call in short_calls.iterrows():
                    short_strike = short_call['strike']
                    short_premium = (short_call['bid'] + short_call['ask']) / 2
                    
                    if short_premium <= 0 or long_premium <= 0:
                        continue
                    
                    # Spread metrics
                    net_debit = long_premium - short_premium
                    max_profit = (short_strike - long_strike) - net_debit
                    max_loss = net_debit
                    breakeven = long_strike + net_debit
                    
                    if max_profit <= 0:
                        continue
                    
                    # Return metrics
                    max_return = max_profit / max_loss
                    prob_profit = 1 - norm.cdf(
                        (np.log(breakeven / current_price) - 
                         (self.risk_free_rate - 0.5 * volatility**2) * time_to_exp) / 
                        (volatility * np.sqrt(time_to_exp))
                    )
                    
                    opportunities.append({
                        'expiration': exp_date,
                        'long_strike': long_strike,
                        'short_strike': short_strike,
                        'spread_width': short_strike - long_strike,
                        'current_price': current_price,
                        'net_debit': net_debit,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'max_return': max_return,
                        'breakeven': breakeven,
                        'prob_profit': prob_profit,
                        'days_to_exp': int(time_to_exp * 365),
                        'long_volume': long_call.get('volume', 0),
                        'short_volume': short_call.get('volume', 0)
                    })
        
        # Sort by risk-adjusted return
        opportunities.sort(key=lambda x: x['max_return'] * x['prob_profit'], reverse=True)
        
        return opportunities
    
    def analyze_iron_condors(self, wing_width: float = 5.0,
                            max_days: int = 30) -> List[Dict[str, Any]]:
        """Analyze iron condor opportunities"""
        
        print(f"Analyzing iron condor opportunities for {self.symbol}...")
        
        stock_data = self.get_stock_data()
        current_price = stock_data['Close'].iloc[-1]
        volatility = stock_data['volatility'].iloc[-1]
        
        options_chains = self.get_options_chain()
        opportunities = []
        
        for exp_date, chain_data in options_chains.items():
            time_to_exp = self.calculate_days_to_expiration(exp_date)
            
            if time_to_exp * 365 > max_days:
                continue
            
            calls = chain_data['calls'].sort_values('strike')
            puts = chain_data['puts'].sort_values('strike')
            
            # Find ATM strikes
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            if atm_call.empty or atm_put.empty:
                continue
            
            atm_strike = atm_call['strike'].iloc[0]
            
            # Define condor strikes
            short_call_strike = atm_strike + wing_width
            long_call_strike = short_call_strike + wing_width
            short_put_strike = atm_strike - wing_width
            long_put_strike = short_put_strike - wing_width
            
            # Find corresponding options
            def find_closest_option(df, target_strike):
                closest_idx = (df['strike'] - target_strike).abs().argsort()[:1]
                return df.iloc[closest_idx] if not closest_idx.empty else None
            
            short_call = find_closest_option(calls, short_call_strike)
            long_call = find_closest_option(calls, long_call_strike)
            short_put = find_closest_option(puts, short_put_strike)
            long_put = find_closest_option(puts, long_put_strike)
            
            if any(opt is None or opt.empty for opt in [short_call, long_call, short_put, long_put]):
                continue
            
            # Calculate premiums
            short_call_premium = (short_call['bid'].iloc[0] + short_call['ask'].iloc[0]) / 2
            long_call_premium = (long_call['bid'].iloc[0] + long_call['ask'].iloc[0]) / 2
            short_put_premium = (short_put['bid'].iloc[0] + short_put['ask'].iloc[0]) / 2
            long_put_premium = (long_put['bid'].iloc[0] + long_put['ask'].iloc[0]) / 2
            
            # Iron condor metrics
            net_credit = (short_call_premium + short_put_premium) - (long_call_premium + long_put_premium)
            max_profit = net_credit
            max_loss = wing_width - net_credit
            
            if max_loss <= 0 or net_credit <= 0:
                continue
            
            # Breakeven points
            upper_breakeven = short_call_strike + net_credit
            lower_breakeven = short_put_strike - net_credit
            
            # Probability of profit (stock stays between breakevens)
            prob_profit = (
                norm.cdf((np.log(upper_breakeven / current_price) - 
                         (self.risk_free_rate - 0.5 * volatility**2) * time_to_exp) / 
                        (volatility * np.sqrt(time_to_exp))) -
                norm.cdf((np.log(lower_breakeven / current_price) - 
                         (self.risk_free_rate - 0.5 * volatility**2) * time_to_exp) / 
                        (volatility * np.sqrt(time_to_exp)))
            )
            
            opportunities.append({
                'expiration': exp_date,
                'long_put_strike': long_put_strike,
                'short_put_strike': short_put_strike,
                'short_call_strike': short_call_strike,
                'long_call_strike': long_call_strike,
                'current_price': current_price,
                'net_credit': net_credit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'return_on_risk': max_profit / max_loss,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'profit_range': upper_breakeven - lower_breakeven,
                'prob_profit': prob_profit,
                'days_to_exp': int(time_to_exp * 365)
            })
        
        # Sort by risk-adjusted return
        opportunities.sort(key=lambda x: x['return_on_risk'] * x['prob_profit'], reverse=True)
        
        return opportunities

class AdvancedOptionsAnalyzer:
    """Advanced options analysis and strategy recommendation system"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.strategies = {}
        
    def analyze_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all options strategies for all symbols"""
        
        print("HIVE TRADE ADVANCED OPTIONS ANALYSIS")
        print("="*45)
        
        results = {}
        
        for symbol in self.symbols:
            print(f"\nAnalyzing {symbol}...")
            print("-" * 25)
            
            symbol_results = {}
            
            try:
                # Covered Calls
                cc_strategy = CoveredCallStrategy(symbol)
                cc_opportunities = cc_strategy.analyze_covered_calls()
                cc_recommendation = cc_strategy.recommend_covered_call(cc_opportunities)
                
                symbol_results['covered_calls'] = {
                    'opportunities': cc_opportunities[:5],  # Top 5
                    'recommendation': cc_recommendation,
                    'total_opportunities': len(cc_opportunities)
                }
                
                print(f"  Covered Calls: {len(cc_opportunities)} opportunities")
                
                # Protective Puts
                pp_strategy = ProtectivePutStrategy(symbol)
                pp_opportunities = pp_strategy.analyze_protective_puts()
                
                symbol_results['protective_puts'] = {
                    'opportunities': pp_opportunities[:5],  # Top 5
                    'total_opportunities': len(pp_opportunities)
                }
                
                print(f"  Protective Puts: {len(pp_opportunities)} opportunities")
                
                # Spreads
                spread_strategy = SpreadStrategies(symbol)
                bull_spreads = spread_strategy.analyze_bull_call_spreads()
                iron_condors = spread_strategy.analyze_iron_condors()
                
                symbol_results['bull_call_spreads'] = {
                    'opportunities': bull_spreads[:5],  # Top 5
                    'total_opportunities': len(bull_spreads)
                }
                
                symbol_results['iron_condors'] = {
                    'opportunities': iron_condors[:3],  # Top 3
                    'total_opportunities': len(iron_condors)
                }
                
                print(f"  Bull Call Spreads: {len(bull_spreads)} opportunities")
                print(f"  Iron Condors: {len(iron_condors)} opportunities")
                
            except Exception as e:
                print(f"  Error analyzing {symbol}: {e}")
                symbol_results['error'] = str(e)
            
            results[symbol] = symbol_results
        
        return results
    
    def generate_options_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive options analysis report"""
        
        report = []
        report.append("ADVANCED OPTIONS TRADING ANALYSIS REPORT")
        report.append("="*50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_opportunities = 0
        strategy_counts = {}
        
        for symbol, data in results.items():
            if 'error' not in data:
                for strategy_type, strategy_data in data.items():
                    if isinstance(strategy_data, dict) and 'total_opportunities' in strategy_data:
                        total_opportunities += strategy_data['total_opportunities']
                        strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + strategy_data['total_opportunities']
        
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 20)
        report.append(f"Total Opportunities Found: {total_opportunities}")
        report.append("")
        
        for strategy, count in strategy_counts.items():
            report.append(f"{strategy.replace('_', ' ').title()}: {count}")
        
        report.append("")
        
        # Top Recommendations
        report.append("TOP RECOMMENDATIONS BY STRATEGY:")
        report.append("-" * 35)
        
        for symbol, data in results.items():
            if 'error' in data:
                continue
                
            report.append(f"\n{symbol}:")
            
            # Covered Call recommendation
            if 'covered_calls' in data and data['covered_calls']['recommendation']:
                cc = data['covered_calls']['recommendation']
                report.append(f"  Covered Call: Strike {cc['strike']}, Premium ${cc['premium']:.2f}")
                report.append(f"    Annual Return if Unchanged: {cc['return_if_unchanged_annual']:.1%}")
                report.append(f"    Prob Profit: {cc['prob_profit']:.1%}")
            
            # Best protective put
            if 'protective_puts' in data and data['protective_puts']['opportunities']:
                pp = data['protective_puts']['opportunities'][0]
                report.append(f"  Protective Put: Strike {pp['strike']}, Premium ${pp['premium']:.2f}")
                report.append(f"    Protection Level: {pp['protection_level']:.1%}")
                report.append(f"    Annual Insurance Cost: {pp['insurance_annual_cost_pct']:.1%}")
            
            # Best bull call spread
            if 'bull_call_spreads' in data and data['bull_call_spreads']['opportunities']:
                bcs = data['bull_call_spreads']['opportunities'][0]
                report.append(f"  Bull Call Spread: {bcs['long_strike']}/{bcs['short_strike']}")
                report.append(f"    Max Return: {bcs['max_return']:.1%}, Prob Profit: {bcs['prob_profit']:.1%}")
            
            # Best iron condor
            if 'iron_condors' in data and data['iron_condors']['opportunities']:
                ic = data['iron_condors']['opportunities'][0]
                report.append(f"  Iron Condor: {ic['long_put_strike']}/{ic['short_put_strike']}/{ic['short_call_strike']}/{ic['long_call_strike']}")
                report.append(f"    Return on Risk: {ic['return_on_risk']:.1%}, Prob Profit: {ic['prob_profit']:.1%}")
        
        return "\n".join(report)

def main():
    """Run advanced options analysis"""
    
    # Analyze major tech stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    
    analyzer = AdvancedOptionsAnalyzer(symbols)
    results = analyzer.analyze_all_strategies()
    
    # Generate report
    report = analyzer.generate_options_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    results_file = f"options_analysis_{timestamp}.json"
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if isinstance(v, (np.integer, np.floating)):
                    cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (list, dict)):
                    cleaned[k] = clean_for_json(v)
                else:
                    cleaned[k] = v
            return cleaned
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        else:
            return obj
    
    clean_results = clean_for_json(results)
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Save report
    report_file = f"options_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n\nResults saved:")
    print(f"- JSON: {results_file}")
    print(f"- Report: {report_file}")
    
    print("\nADVANCED OPTIONS ANALYSIS COMPLETE!")
    print("="*45)
    
    return results

if __name__ == "__main__":
    main()