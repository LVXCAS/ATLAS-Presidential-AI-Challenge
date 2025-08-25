"""
Hive Trade Correlation-Based Position Sizing
Advanced position sizing based on correlation analysis and portfolio optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AssetData:
    """Asset data for correlation analysis"""
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    market_value: float
    current_weight: float

@dataclass
class PositionSizingResult:
    """Position sizing recommendation result"""
    symbol: str
    current_weight: float
    recommended_weight: float
    weight_change: float
    position_change_usd: float
    reason: str
    confidence: float

class CorrelationAnalyzer:
    """Advanced correlation analysis for portfolio optimization"""
    
    def __init__(self):
        self.correlation_threshold_high = 0.7  # High correlation threshold
        self.correlation_threshold_medium = 0.4  # Medium correlation threshold
        
    def calculate_correlation_matrix(self, assets: List[AssetData], 
                                   time_periods: int = 252) -> np.ndarray:
        """Calculate correlation matrix for assets (mock implementation)"""
        
        n_assets = len(assets)
        correlation_matrix = np.eye(n_assets)
        
        # Generate realistic correlations based on asset types
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset_i, asset_j = assets[i], assets[j]
                
                # Determine correlation based on asset similarity
                correlation = self._estimate_correlation(asset_i.symbol, asset_j.symbol)
                
                # Add some randomness
                correlation += np.random.normal(0, 0.1)
                correlation = np.clip(correlation, -0.95, 0.95)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two assets based on their types"""
        
        # Asset type classification
        tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'TSLA']
        financial_stocks = ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC']
        healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV']
        energy_stocks = ['XOM', 'CVX', 'COP', 'SLB']
        utilities = ['NEE', 'DUK', 'SO', 'D']
        bonds = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']
        commodities = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']
        crypto = ['BTCUSD', 'ETHUSD', 'BTC', 'ETH']
        
        # Same sector correlations
        sectors = [tech_stocks, financial_stocks, healthcare_stocks, energy_stocks, utilities]
        
        for sector in sectors:
            if symbol1 in sector and symbol2 in sector:
                return np.random.uniform(0.6, 0.85)  # High correlation within sector
        
        # Cross-sector correlations
        if ((symbol1 in tech_stocks and symbol2 in financial_stocks) or
            (symbol1 in financial_stocks and symbol2 in tech_stocks)):
            return np.random.uniform(0.3, 0.6)  # Medium correlation
        
        # Bonds vs stocks
        if ((symbol1 in bonds and symbol2 not in bonds) or
            (symbol1 not in bonds and symbol2 in bonds)):
            return np.random.uniform(-0.4, 0.2)  # Low/negative correlation
        
        # Commodities vs stocks
        if ((symbol1 in commodities and symbol2 not in commodities) or
            (symbol1 not in commodities and symbol2 in commodities)):
            return np.random.uniform(-0.2, 0.3)  # Low correlation
        
        # Crypto correlations
        if symbol1 in crypto or symbol2 in crypto:
            if symbol1 in crypto and symbol2 in crypto:
                return np.random.uniform(0.7, 0.9)  # High crypto-crypto correlation
            else:
                return np.random.uniform(-0.1, 0.4)  # Low crypto-traditional correlation
        
        # Default correlation for different sectors
        return np.random.uniform(0.2, 0.5)
    
    def identify_correlation_clusters(self, correlation_matrix: np.ndarray, 
                                    assets: List[AssetData]) -> Dict[str, List[str]]:
        """Identify clusters of highly correlated assets"""
        
        clusters = {}
        cluster_id = 0
        processed_assets = set()
        
        n_assets = len(assets)
        
        for i in range(n_assets):
            if assets[i].symbol in processed_assets:
                continue
            
            cluster_name = f"Cluster_{cluster_id}"
            cluster_assets = [assets[i].symbol]
            processed_assets.add(assets[i].symbol)
            
            # Find highly correlated assets
            for j in range(i + 1, n_assets):
                if (correlation_matrix[i, j] > self.correlation_threshold_high and 
                    assets[j].symbol not in processed_assets):
                    cluster_assets.append(assets[j].symbol)
                    processed_assets.add(assets[j].symbol)
            
            if len(cluster_assets) > 1:  # Only create cluster if more than one asset
                clusters[cluster_name] = cluster_assets
                cluster_id += 1
        
        return clusters
    
    def calculate_portfolio_diversification_ratio(self, weights: np.ndarray, 
                                                correlation_matrix: np.ndarray,
                                                volatilities: np.ndarray) -> float:
        """Calculate portfolio diversification ratio"""
        
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Portfolio volatility
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        if portfolio_vol > 0:
            diversification_ratio = weighted_avg_vol / portfolio_vol
        else:
            diversification_ratio = 1.0
        
        return diversification_ratio

class OptimalPositionSizer:
    """Advanced position sizing using modern portfolio theory"""
    
    def __init__(self):
        self.risk_free_rate = 0.04  # 4% risk-free rate
        self.correlation_analyzer = CorrelationAnalyzer()
        
    def calculate_efficient_frontier_weights(self, assets: List[AssetData],
                                           correlation_matrix: np.ndarray,
                                           target_return: Optional[float] = None) -> np.ndarray:
        """Calculate optimal weights using mean-variance optimization"""
        
        n_assets = len(assets)
        
        # Expected returns and volatilities
        expected_returns = np.array([asset.expected_return for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)
        
        def negative_sharpe_ratio(weights):
            port_return = portfolio_return(weights)
            port_vol = portfolio_volatility(weights)
            if port_vol == 0:
                return -np.inf
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
        
        # Bounds (0% to 40% per position to avoid concentration)
        bounds = tuple((0, 0.4) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimization
        try:
            if target_return is None:
                # Maximum Sharpe ratio portfolio
                result = optimize.minimize(negative_sharpe_ratio, x0, method='SLSQP',
                                         bounds=bounds, constraints=constraints)
            else:
                # Minimum variance for target return
                result = optimize.minimize(portfolio_volatility, x0, method='SLSQP',
                                         bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                print(f"Optimization failed: {result.message}")
                return np.array([1/n_assets] * n_assets)  # Equal weights fallback
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return np.array([1/n_assets] * n_assets)  # Equal weights fallback
    
    def calculate_black_litterman_weights(self, assets: List[AssetData],
                                        correlation_matrix: np.ndarray,
                                        market_cap_weights: np.ndarray,
                                        views: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Calculate Black-Litterman optimal weights"""
        
        # This is a simplified Black-Litterman implementation
        # In practice, would need more sophisticated parameter estimation
        
        n_assets = len(assets)
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Market equilibrium returns (CAPM)
        market_risk_premium = 0.06  # 6% market risk premium
        market_volatility = 0.16    # 16% market volatility
        
        # Implied equilibrium returns
        risk_aversion = market_risk_premium / (market_volatility ** 2)
        pi = risk_aversion * np.dot(covariance_matrix, market_cap_weights)
        
        if views is None or len(views) == 0:
            # No views - return market cap weights
            return market_cap_weights
        
        # Incorporate views (simplified)
        tau = 0.1  # Uncertainty parameter
        
        # Create view matrix and return vector
        view_symbols = list(views.keys())
        k = len(view_symbols)
        P = np.zeros((k, n_assets))
        Q = np.zeros(k)
        
        for i, symbol in enumerate(view_symbols):
            asset_idx = next((j for j, asset in enumerate(assets) if asset.symbol == symbol), -1)
            if asset_idx >= 0:
                P[i, asset_idx] = 1
                Q[i] = views[symbol]
        
        # View uncertainty matrix (diagonal)
        omega = np.eye(k) * 0.01  # 1% view uncertainty
        
        # Black-Litterman formula
        tau_sigma = tau * covariance_matrix
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)
        
        # New expected returns
        M1 = inv_tau_sigma + np.dot(P.T, np.dot(inv_omega, P))
        M2 = np.dot(inv_tau_sigma, pi) + np.dot(P.T, np.dot(inv_omega, Q))
        mu_bl = np.dot(np.linalg.inv(M1), M2)
        
        # New covariance matrix
        sigma_bl = np.linalg.inv(M1)
        
        # Optimal weights
        weights = np.dot(np.linalg.inv(risk_aversion * sigma_bl), mu_bl)
        
        # Normalize weights to sum to 1
        weights = np.maximum(weights, 0)  # No short selling
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights
    
    def risk_parity_weights(self, assets: List[AssetData],
                           correlation_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk parity (equal risk contribution) weights"""
        
        n_assets = len(assets)
        volatilities = np.array([asset.volatility for asset in assets])
        
        # Covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return contrib
        
        def risk_parity_objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.4) for _ in range(n_assets))  # 1% to 40% per asset
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        try:
            result = optimize.minimize(risk_parity_objective, x0, method='SLSQP',
                                     bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                return np.array([1/n_assets] * n_assets)
                
        except Exception as e:
            print(f"Risk parity optimization error: {e}")
            return np.array([1/n_assets] * n_assets)

class CorrelationBasedPositionSizer:
    """Main correlation-based position sizing system"""
    
    def __init__(self):
        self.correlation_analyzer = CorrelationAnalyzer()
        self.position_sizer = OptimalPositionSizer()
        self.rebalancing_threshold = 0.05  # 5% threshold for rebalancing
        
    def create_sample_portfolio(self) -> List[AssetData]:
        """Create sample portfolio for analysis"""
        
        sample_assets = [
            AssetData("AAPL", 0.12, 0.25, 0.48, 50000, 20.0),
            AssetData("GOOGL", 0.14, 0.28, 0.50, 40000, 16.0),
            AssetData("MSFT", 0.11, 0.22, 0.50, 35000, 14.0),
            AssetData("NVDA", 0.18, 0.35, 0.51, 30000, 12.0),
            AssetData("AMZN", 0.15, 0.30, 0.50, 25000, 10.0),
            AssetData("JPM", 0.10, 0.32, 0.31, 20000, 8.0),
            AssetData("JNJ", 0.08, 0.18, 0.44, 15000, 6.0),
            AssetData("TLT", 0.04, 0.15, 0.27, 20000, 8.0),
            AssetData("GLD", 0.06, 0.20, 0.30, 10000, 4.0),
            AssetData("VTI", 0.10, 0.16, 0.63, 5000, 2.0)
        ]
        
        return sample_assets
    
    def analyze_portfolio_correlations(self, assets: List[AssetData]) -> Dict[str, Any]:
        """Analyze portfolio correlation structure"""
        
        print("\nCORRELATION ANALYSIS")
        print("="*20)
        
        # Calculate correlation matrix
        correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(assets)
        
        # Identify correlation clusters
        clusters = self.correlation_analyzer.identify_correlation_clusters(
            correlation_matrix, assets
        )
        
        # Calculate diversification metrics
        current_weights = np.array([asset.current_weight / 100 for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])
        
        diversification_ratio = self.correlation_analyzer.calculate_portfolio_diversification_ratio(
            current_weights, correlation_matrix, volatilities
        )
        
        # Find highest correlations
        n = len(assets)
        high_correlations = []
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.6:
                    high_correlations.append({
                        'asset1': assets[i].symbol,
                        'asset2': assets[j].symbol,
                        'correlation': corr
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print(f"Portfolio Diversification Ratio: {diversification_ratio:.2f}")
        print(f"Number of Correlation Clusters: {len(clusters)}")
        
        if clusters:
            print("\nCorrelation Clusters:")
            for cluster_name, cluster_assets in clusters.items():
                print(f"  {cluster_name}: {', '.join(cluster_assets)}")
        
        if high_correlations:
            print(f"\nTop High Correlations:")
            for corr in high_correlations[:5]:
                print(f"  {corr['asset1']} - {corr['asset2']}: {corr['correlation']:.3f}")
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'clusters': clusters,
            'diversification_ratio': diversification_ratio,
            'high_correlations': high_correlations,
            'avg_correlation': float(np.mean(correlation_matrix[np.triu_indices(n, k=1)]))
        }
    
    def generate_position_sizing_recommendations(self, assets: List[AssetData],
                                               correlation_matrix: np.ndarray,
                                               method: str = 'max_sharpe') -> List[PositionSizingResult]:
        """Generate position sizing recommendations"""
        
        print(f"\nPOSITION SIZING RECOMMENDATIONS ({method.upper()})")
        print("="*45)
        
        # Current weights
        current_weights = np.array([asset.current_weight / 100 for asset in assets])
        
        # Calculate optimal weights based on method
        if method == 'max_sharpe':
            optimal_weights = self.position_sizer.calculate_efficient_frontier_weights(
                assets, correlation_matrix
            )
        elif method == 'risk_parity':
            optimal_weights = self.position_sizer.risk_parity_weights(
                assets, correlation_matrix
            )
        elif method == 'black_litterman':
            # Use market cap weights as starting point
            market_cap_weights = current_weights / np.sum(current_weights)
            optimal_weights = self.position_sizer.calculate_black_litterman_weights(
                assets, correlation_matrix, market_cap_weights
            )
        else:
            optimal_weights = current_weights  # No change
        
        # Generate recommendations
        recommendations = []
        total_portfolio_value = sum(asset.market_value for asset in assets)
        
        for i, asset in enumerate(assets):
            current_weight = current_weights[i] * 100  # Convert to percentage
            recommended_weight = optimal_weights[i] * 100  # Convert to percentage
            weight_change = recommended_weight - current_weight
            position_change_usd = (weight_change / 100) * total_portfolio_value
            
            # Determine reason for recommendation
            if abs(weight_change) < 1:
                reason = "Maintain current position"
                confidence = 0.7
            elif weight_change > 5:
                reason = "Increase position - attractive risk-adjusted return"
                confidence = 0.8
            elif weight_change < -5:
                reason = "Decrease position - overweight relative to optimal"
                confidence = 0.8
            elif weight_change > 0:
                reason = "Small increase recommended"
                confidence = 0.6
            else:
                reason = "Small decrease recommended"
                confidence = 0.6
            
            recommendations.append(PositionSizingResult(
                symbol=asset.symbol,
                current_weight=current_weight,
                recommended_weight=recommended_weight,
                weight_change=weight_change,
                position_change_usd=position_change_usd,
                reason=reason,
                confidence=confidence
            ))
        
        # Sort by absolute weight change
        recommendations.sort(key=lambda x: abs(x.weight_change), reverse=True)
        
        # Print recommendations
        for rec in recommendations:
            print(f"{rec.symbol:8} Current: {rec.current_weight:5.1f}% -> "
                  f"Target: {rec.recommended_weight:5.1f}% "
                  f"(Change: {rec.weight_change:+5.1f}%, ${rec.position_change_usd:+8,.0f})")
        
        return recommendations
    
    def calculate_rebalancing_urgency(self, recommendations: List[PositionSizingResult]) -> Dict[str, Any]:
        """Calculate rebalancing urgency and priority"""
        
        urgent_rebalances = []
        total_rebalancing_value = 0
        
        for rec in recommendations:
            if abs(rec.weight_change) > self.rebalancing_threshold * 100:  # Convert to percentage
                urgency_score = abs(rec.weight_change) * rec.confidence
                urgent_rebalances.append({
                    'symbol': rec.symbol,
                    'weight_change': rec.weight_change,
                    'position_change_usd': rec.position_change_usd,
                    'urgency_score': urgency_score,
                    'reason': rec.reason
                })
                total_rebalancing_value += abs(rec.position_change_usd)
        
        # Sort by urgency score
        urgent_rebalances.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return {
            'urgent_rebalances': urgent_rebalances,
            'total_rebalancing_value': total_rebalancing_value,
            'num_positions_to_rebalance': len(urgent_rebalances),
            'rebalancing_threshold': self.rebalancing_threshold
        }
    
    def run_correlation_analysis(self, assets: Optional[List[AssetData]] = None) -> Dict[str, Any]:
        """Run complete correlation-based position sizing analysis"""
        
        if assets is None:
            assets = self.create_sample_portfolio()
        
        print("HIVE TRADE CORRELATION-BASED POSITION SIZING")
        print("="*48)
        
        # Portfolio summary
        total_value = sum(asset.market_value for asset in assets)
        weighted_return = sum(asset.expected_return * asset.current_weight / 100 for asset in assets)
        weighted_vol = np.sqrt(sum((asset.volatility * asset.current_weight / 100) ** 2 for asset in assets))
        portfolio_sharpe = (weighted_return - 0.04) / weighted_vol if weighted_vol > 0 else 0
        
        print(f"\nPORTFOLIO SUMMARY:")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Expected Return: {weighted_return:.1%}")
        print(f"Portfolio Volatility: {weighted_vol:.1%}")
        print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
        
        # Correlation analysis
        correlation_analysis = self.analyze_portfolio_correlations(assets)
        correlation_matrix = np.array(correlation_analysis['correlation_matrix'])
        
        # Position sizing recommendations for different methods
        methods = ['max_sharpe', 'risk_parity', 'black_litterman']
        all_recommendations = {}
        
        for method in methods:
            recommendations = self.generate_position_sizing_recommendations(
                assets, correlation_matrix, method
            )
            all_recommendations[method] = recommendations
        
        # Rebalancing analysis (using max_sharpe recommendations)
        rebalancing_analysis = self.calculate_rebalancing_urgency(
            all_recommendations['max_sharpe']
        )
        
        print(f"\nREBALANCING SUMMARY:")
        print(f"Positions needing rebalancing: {rebalancing_analysis['num_positions_to_rebalance']}")
        print(f"Total rebalancing value: ${rebalancing_analysis['total_rebalancing_value']:,.2f}")
        
        if rebalancing_analysis['urgent_rebalances']:
            print(f"\nTOP REBALANCING PRIORITIES:")
            for rebal in rebalancing_analysis['urgent_rebalances'][:5]:
                print(f"  {rebal['symbol']}: {rebal['weight_change']:+.1f}% "
                      f"(${rebal['position_change_usd']:+,.0f}) - {rebal['reason']}")
        
        return {
            'portfolio_summary': {
                'total_value': total_value,
                'expected_return': weighted_return,
                'portfolio_volatility': weighted_vol,
                'portfolio_sharpe': portfolio_sharpe
            },
            'correlation_analysis': correlation_analysis,
            'position_recommendations': {
                method: [
                    {
                        'symbol': rec.symbol,
                        'current_weight': rec.current_weight,
                        'recommended_weight': rec.recommended_weight,
                        'weight_change': rec.weight_change,
                        'position_change_usd': rec.position_change_usd,
                        'reason': rec.reason,
                        'confidence': rec.confidence
                    }
                    for rec in recommendations
                ]
                for method, recommendations in all_recommendations.items()
            },
            'rebalancing_analysis': rebalancing_analysis,
            'assets': [
                {
                    'symbol': asset.symbol,
                    'expected_return': asset.expected_return,
                    'volatility': asset.volatility,
                    'sharpe_ratio': asset.sharpe_ratio,
                    'market_value': asset.market_value,
                    'current_weight': asset.current_weight
                }
                for asset in assets
            ],
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run correlation-based position sizing analysis"""
    
    # Initialize system
    position_sizer = CorrelationBasedPositionSizer()
    
    # Run analysis
    results = position_sizer.run_correlation_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"correlation_position_sizing_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    print("\nCORRELATION-BASED POSITION SIZING COMPLETE!")
    print("="*48)
    
    return results

if __name__ == "__main__":
    main()