"""
Hive Trade Real-time Portfolio Heat Maps
Advanced risk visualization and concentration analysis
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

# For visualization (would use plotly/matplotlib in real implementation)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Matplotlib/Seaborn not available - generating text-based visualizations")

@dataclass
class PortfolioPosition:
    """Portfolio position data"""
    symbol: str
    asset_class: str
    sector: str
    quantity: float
    market_value: float
    weight: float
    daily_pnl: float
    total_pnl: float
    beta: float
    volatility: float
    correlation_to_market: float
    var_contribution: float

class RiskCalculator:
    """Advanced risk calculation utilities"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def calculate_portfolio_var(self, positions: List[PortfolioPosition], 
                               confidence: float = 0.95, time_horizon: int = 1) -> float:
        """Calculate portfolio Value at Risk"""
        
        if not positions:
            return 0.0
        
        # Create weights and volatilities arrays
        weights = np.array([pos.weight / 100 for pos in positions])  # Convert percentage to decimal
        volatilities = np.array([pos.volatility for pos in positions])
        values = np.array([pos.market_value for pos in positions])
        
        # Mock correlation matrix (would use real historical data)
        n_assets = len(positions)
        correlation_matrix = self._generate_mock_correlation_matrix(positions)
        
        # Calculate covariance matrix
        vol_matrix = np.outer(volatilities, volatilities)
        covariance_matrix = correlation_matrix * vol_matrix
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio value
        portfolio_value = np.sum(values)
        
        # VaR calculation (normal distribution assumption)
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        var = portfolio_value * portfolio_volatility * z_score * np.sqrt(time_horizon)
        
        return abs(var)
    
    def _generate_mock_correlation_matrix(self, positions: List[PortfolioPosition]) -> np.ndarray:
        """Generate mock correlation matrix based on asset classes and sectors"""
        
        n = len(positions)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                pos_i, pos_j = positions[i], positions[j]
                
                # Base correlation on asset class and sector similarity
                if pos_i.asset_class == pos_j.asset_class:
                    if pos_i.sector == pos_j.sector:
                        correlation = np.random.uniform(0.6, 0.9)  # High correlation same sector
                    else:
                        correlation = np.random.uniform(0.3, 0.7)  # Medium correlation same asset class
                else:
                    correlation = np.random.uniform(-0.2, 0.4)  # Low/varied correlation different asset classes
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def calculate_marginal_var(self, positions: List[PortfolioPosition], 
                              portfolio_var: float) -> Dict[str, float]:
        """Calculate marginal VaR contribution for each position"""
        
        marginal_vars = {}
        
        for position in positions:
            # Simplified marginal VaR calculation
            # In practice, would use partial derivatives of VaR w.r.t. position weights
            weight_factor = position.weight / 100
            volatility_factor = position.volatility
            correlation_factor = abs(position.correlation_to_market)
            
            marginal_var = portfolio_var * weight_factor * volatility_factor * correlation_factor
            marginal_vars[position.symbol] = marginal_var
        
        return marginal_vars

class ConcentrationAnalyzer:
    """Portfolio concentration risk analysis"""
    
    def __init__(self):
        self.concentration_limits = {
            'single_position': 0.10,  # 10% max single position
            'sector': 0.25,           # 25% max sector exposure
            'asset_class': 0.40,      # 40% max asset class exposure
            'country': 0.30,          # 30% max country exposure
            'currency': 0.50          # 50% max currency exposure
        }
    
    def analyze_concentration_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Comprehensive concentration risk analysis"""
        
        total_value = sum(pos.market_value for pos in positions)
        
        # Single position concentration
        position_weights = {pos.symbol: pos.market_value / total_value for pos in positions}
        max_position_weight = max(position_weights.values()) if position_weights else 0
        
        # Sector concentration
        sector_exposure = {}
        for pos in positions:
            sector_exposure[pos.sector] = sector_exposure.get(pos.sector, 0) + pos.market_value
        
        sector_weights = {k: v / total_value for k, v in sector_exposure.items()}
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        
        # Asset class concentration
        asset_class_exposure = {}
        for pos in positions:
            asset_class_exposure[pos.asset_class] = asset_class_exposure.get(pos.asset_class, 0) + pos.market_value
        
        asset_class_weights = {k: v / total_value for k, v in asset_class_exposure.items()}
        max_asset_class_weight = max(asset_class_weights.values()) if asset_class_weights else 0
        
        # Herfindahl-Hirschman Index (HHI) for diversification
        hhi = sum((weight * 100) ** 2 for weight in position_weights.values())
        
        # Concentration alerts
        alerts = []
        
        if max_position_weight > self.concentration_limits['single_position']:
            alerts.append(f"Single position concentration: {max_position_weight:.1%} > {self.concentration_limits['single_position']:.1%}")
        
        if max_sector_weight > self.concentration_limits['sector']:
            alerts.append(f"Sector concentration: {max_sector_weight:.1%} > {self.concentration_limits['sector']:.1%}")
        
        if max_asset_class_weight > self.concentration_limits['asset_class']:
            alerts.append(f"Asset class concentration: {max_asset_class_weight:.1%} > {self.concentration_limits['asset_class']:.1%}")
        
        return {
            'position_weights': position_weights,
            'sector_weights': sector_weights,
            'asset_class_weights': asset_class_weights,
            'max_position_weight': max_position_weight,
            'max_sector_weight': max_sector_weight,
            'max_asset_class_weight': max_asset_class_weight,
            'hhi_index': hhi,
            'diversification_score': max(0, 100 - hhi / 100),  # Higher is more diversified
            'concentration_alerts': alerts
        }

class HeatMapGenerator:
    """Generate portfolio heat maps and visualizations"""
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.concentration_analyzer = ConcentrationAnalyzer()
    
    def generate_performance_heatmap(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Generate performance-based heat map data"""
        
        # Create heat map matrix organized by asset class and sector
        heatmap_data = {}
        
        # Group by asset class and sector
        for pos in positions:
            if pos.asset_class not in heatmap_data:
                heatmap_data[pos.asset_class] = {}
            
            if pos.sector not in heatmap_data[pos.asset_class]:
                heatmap_data[pos.asset_class][pos.sector] = []
            
            heatmap_data[pos.asset_class][pos.sector].append({
                'symbol': pos.symbol,
                'daily_pnl_pct': (pos.daily_pnl / pos.market_value) * 100 if pos.market_value > 0 else 0,
                'total_pnl_pct': (pos.total_pnl / pos.market_value) * 100 if pos.market_value > 0 else 0,
                'weight': pos.weight,
                'market_value': pos.market_value
            })
        
        return heatmap_data
    
    def generate_risk_heatmap(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Generate risk-based heat map data"""
        
        portfolio_var = self.risk_calculator.calculate_portfolio_var(positions)
        marginal_vars = self.risk_calculator.calculate_marginal_var(positions, portfolio_var)
        
        # Create risk heat map
        risk_heatmap = {}
        
        for pos in positions:
            if pos.asset_class not in risk_heatmap:
                risk_heatmap[pos.asset_class] = {}
            
            if pos.sector not in risk_heatmap[pos.asset_class]:
                risk_heatmap[pos.asset_class][pos.sector] = []
            
            risk_heatmap[pos.asset_class][pos.sector].append({
                'symbol': pos.symbol,
                'volatility': pos.volatility * 100,  # Convert to percentage
                'beta': pos.beta,
                'var_contribution': marginal_vars.get(pos.symbol, 0),
                'weight': pos.weight,
                'risk_score': pos.volatility * pos.weight * abs(pos.beta)  # Combined risk score
            })
        
        return {
            'risk_heatmap': risk_heatmap,
            'portfolio_var': portfolio_var,
            'total_positions': len(positions)
        }
    
    def generate_correlation_heatmap(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Generate correlation heat map between positions"""
        
        symbols = [pos.symbol for pos in positions]
        n = len(symbols)
        
        if n < 2:
            return {'correlation_matrix': {}, 'symbols': symbols}
        
        # Generate mock correlation matrix
        correlation_matrix = self.risk_calculator._generate_mock_correlation_matrix(positions)
        
        # Convert to dictionary format for JSON serialization
        correlation_dict = {}
        for i, symbol_i in enumerate(symbols):
            correlation_dict[symbol_i] = {}
            for j, symbol_j in enumerate(symbols):
                correlation_dict[symbol_i][symbol_j] = float(correlation_matrix[i, j])
        
        return {
            'correlation_matrix': correlation_dict,
            'symbols': symbols,
            'avg_correlation': float(np.mean(correlation_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0
        }
    
    def create_text_heatmap(self, data: Dict[str, Any], title: str, 
                           value_key: str, format_str: str = "{:.1f}") -> str:
        """Create text-based heat map visualization"""
        
        output = [f"\n{title}", "=" * len(title)]
        
        if isinstance(data, dict) and 'risk_heatmap' in data:
            # Risk heat map format
            heatmap_data = data['risk_heatmap']
            
            for asset_class, sectors in heatmap_data.items():
                output.append(f"\n{asset_class.upper()}:")
                
                for sector, positions in sectors.items():
                    output.append(f"  {sector}:")
                    
                    for pos in positions:
                        value = pos.get(value_key, 0)
                        weight = pos.get('weight', 0)
                        symbol = pos.get('symbol', '')
                        
                        # Color coding with text
                        if value > 30:
                            color = "HIGH"
                        elif value > 15:
                            color = "MED "
                        else:
                            color = "LOW "
                        
                        formatted_value = format_str.format(value)
                        output.append(f"    {symbol:8} {color} {formatted_value:>8} (Weight: {weight:.1f}%)")
        
        else:
            # Standard heat map format
            for asset_class, sectors in data.items():
                output.append(f"\n{asset_class.upper()}:")
                
                for sector, positions in sectors.items():
                    output.append(f"  {sector}:")
                    
                    for pos in positions:
                        value = pos.get(value_key, 0)
                        weight = pos.get('weight', 0)
                        symbol = pos.get('symbol', '')
                        
                        # Color coding with text
                        if value > 5:
                            color = "GAIN"
                        elif value < -5:
                            color = "LOSS"
                        else:
                            color = "FLAT"
                        
                        formatted_value = format_str.format(value)
                        output.append(f"    {symbol:8} {color} {formatted_value:>8}% (Weight: {weight:.1f}%)")
        
        return "\n".join(output)

class PortfolioHeatMapSystem:
    """Complete portfolio heat map system"""
    
    def __init__(self):
        self.heatmap_generator = HeatMapGenerator()
        self.concentration_analyzer = ConcentrationAnalyzer()
        self.risk_calculator = RiskCalculator()
    
    def create_sample_portfolio(self) -> List[PortfolioPosition]:
        """Create sample portfolio for demonstration"""
        
        sample_positions = [
            # Tech stocks
            PortfolioPosition("AAPL", "Equities", "Technology", 100, 18000, 15.0, 200, 1500, 1.2, 0.25, 0.8, 0),
            PortfolioPosition("GOOGL", "Equities", "Technology", 50, 15000, 12.5, -150, 800, 1.1, 0.28, 0.85, 0),
            PortfolioPosition("MSFT", "Equities", "Technology", 40, 14000, 11.7, 100, 1200, 1.0, 0.22, 0.82, 0),
            PortfolioPosition("NVDA", "Equities", "Technology", 20, 12000, 10.0, 300, 2000, 1.8, 0.35, 0.9, 0),
            
            # Financial stocks
            PortfolioPosition("JPM", "Equities", "Financials", 80, 11000, 9.2, -50, 500, 1.4, 0.32, 0.75, 0),
            PortfolioPosition("BAC", "Equities", "Financials", 150, 8000, 6.7, -80, 200, 1.6, 0.38, 0.78, 0),
            
            # Healthcare
            PortfolioPosition("JNJ", "Equities", "Healthcare", 60, 9000, 7.5, 50, 600, 0.8, 0.18, 0.6, 0),
            PortfolioPosition("PFE", "Equities", "Healthcare", 120, 7000, 5.8, -20, 300, 0.9, 0.20, 0.65, 0),
            
            # Bonds
            PortfolioPosition("TLT", "Fixed Income", "Government", 200, 10000, 8.3, -100, -500, -0.5, 0.15, -0.3, 0),
            PortfolioPosition("LQD", "Fixed Income", "Corporate", 150, 8000, 6.7, -50, -200, 0.3, 0.12, 0.2, 0),
            
            # Commodities
            PortfolioPosition("GLD", "Commodities", "Precious Metals", 80, 6000, 5.0, 80, 400, -0.2, 0.20, -0.1, 0),
            PortfolioPosition("USO", "Commodities", "Energy", 100, 2000, 1.7, 40, 150, 1.2, 0.45, 0.3, 0)
        ]
        
        return sample_positions
    
    def run_portfolio_analysis(self, positions: Optional[List[PortfolioPosition]] = None) -> Dict[str, Any]:
        """Run comprehensive portfolio heat map analysis"""
        
        if positions is None:
            positions = self.create_sample_portfolio()
        
        print("HIVE TRADE PORTFOLIO HEAT MAP ANALYSIS")
        print("="*40)
        
        # Calculate VaR contributions
        portfolio_var = self.risk_calculator.calculate_portfolio_var(positions)
        marginal_vars = self.risk_calculator.calculate_marginal_var(positions, portfolio_var)
        
        # Update positions with VaR contributions
        for pos in positions:
            pos.var_contribution = marginal_vars.get(pos.symbol, 0)
        
        # Generate heat maps
        performance_heatmap = self.heatmap_generator.generate_performance_heatmap(positions)
        risk_heatmap = self.heatmap_generator.generate_risk_heatmap(positions)
        correlation_heatmap = self.heatmap_generator.generate_correlation_heatmap(positions)
        
        # Concentration analysis
        concentration_analysis = self.concentration_analyzer.analyze_concentration_risk(positions)
        
        # Portfolio summary statistics
        total_value = sum(pos.market_value for pos in positions)
        total_daily_pnl = sum(pos.daily_pnl for pos in positions)
        total_pnl = sum(pos.total_pnl for pos in positions)
        
        portfolio_summary = {
            'total_value': total_value,
            'daily_pnl': total_daily_pnl,
            'daily_return_pct': (total_daily_pnl / total_value) * 100,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / (total_value - total_pnl)) * 100,
            'portfolio_var_95': portfolio_var,
            'var_as_pct_portfolio': (portfolio_var / total_value) * 100,
            'num_positions': len(positions)
        }
        
        # Print summaries
        print(f"\nPORTFOLIO SUMMARY:")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Daily P&L: ${total_daily_pnl:,.2f} ({portfolio_summary['daily_return_pct']:.2f}%)")
        print(f"Total P&L: ${total_pnl:,.2f} ({portfolio_summary['total_return_pct']:.2f}%)")
        print(f"Portfolio VaR (95%): ${portfolio_var:,.2f} ({portfolio_summary['var_as_pct_portfolio']:.2f}%)")
        
        print(f"\nCONCENTRATION ANALYSIS:")
        print(f"Max Single Position: {concentration_analysis['max_position_weight']:.1%}")
        print(f"Max Sector Exposure: {concentration_analysis['max_sector_weight']:.1%}")
        print(f"Max Asset Class: {concentration_analysis['max_asset_class_weight']:.1%}")
        print(f"Diversification Score: {concentration_analysis['diversification_score']:.1f}/100")
        
        if concentration_analysis['concentration_alerts']:
            print(f"\nCONCENTRATION ALERTS:")
            for alert in concentration_analysis['concentration_alerts']:
                print(f"  WARNING: {alert}")
        
        # Generate text heat maps
        performance_text = self.heatmap_generator.create_text_heatmap(
            performance_heatmap, 
            "DAILY PERFORMANCE HEAT MAP", 
            "daily_pnl_pct",
            "{:.2f}"
        )
        
        risk_text = self.heatmap_generator.create_text_heatmap(
            risk_heatmap, 
            "RISK HEAT MAP (Volatility %)", 
            "volatility",
            "{:.1f}"
        )
        
        print(performance_text)
        print(risk_text)
        
        # Top risk contributors
        risk_contributors = sorted(positions, key=lambda x: x.var_contribution, reverse=True)[:5]
        print(f"\nTOP RISK CONTRIBUTORS:")
        for i, pos in enumerate(risk_contributors, 1):
            print(f"  {i}. {pos.symbol}: ${pos.var_contribution:,.2f} VaR contribution")
        
        return {
            'portfolio_summary': portfolio_summary,
            'performance_heatmap': performance_heatmap,
            'risk_heatmap': risk_heatmap,
            'correlation_heatmap': correlation_heatmap,
            'concentration_analysis': concentration_analysis,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'asset_class': pos.asset_class,
                    'sector': pos.sector,
                    'market_value': pos.market_value,
                    'weight': pos.weight,
                    'daily_pnl': pos.daily_pnl,
                    'total_pnl': pos.total_pnl,
                    'volatility': pos.volatility,
                    'beta': pos.beta,
                    'var_contribution': pos.var_contribution
                }
                for pos in positions
            ],
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run portfolio heat map analysis"""
    
    # Initialize system
    heatmap_system = PortfolioHeatMapSystem()
    
    # Run analysis
    results = heatmap_system.run_portfolio_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"portfolio_heatmap_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\nPORTFOLIO HEAT MAP ANALYSIS COMPLETE!")
    print("="*40)
    
    return results

if __name__ == "__main__":
    main()