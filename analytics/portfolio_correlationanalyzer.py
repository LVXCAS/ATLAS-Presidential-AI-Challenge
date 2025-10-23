#!/usr/bin/env python3
"""
PORTFOLIO CORRELATION ANALYZER - Week 5+ Feature
=================================================
Analyze correlations between positions to avoid concentration risk

Critical for $8M across 80 accounts:
- Avoid all accounts holding same correlated stocks
- Diversify across sectors and market cap
- Detect hidden correlations (e.g., tech stocks moving together)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt


class PortfolioCorrelationAnalyzer:
    """Analyze correlations between portfolio positions"""

    def __init__(self):
        self.correlation_matrix = None
        self.sector_map = {}

    def get_correlation_matrix(self, symbols, lookback_days=60):
        """Calculate correlation matrix for list of symbols"""

        print(f"\n{'='*80}")
        print(f"CALCULATING CORRELATIONS: {len(symbols)} symbols")
        print(f"Lookback: {lookback_days} days")
        print(f"{'='*80}")

        # Download price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)

        price_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    price_data[symbol] = hist['Close']
            except Exception as e:
                print(f"  [WARNING] Failed to get data for {symbol}: {e}")

        if len(price_data) < 2:
            print(f"  [ERROR] Need at least 2 symbols with data")
            return None

        # Create DataFrame
        df = pd.DataFrame(price_data)

        # Calculate returns
        returns = df.pct_change().dropna()

        # Calculate correlation matrix
        self.correlation_matrix = returns.corr()

        print(f"  [OK] Correlation matrix calculated")
        return self.correlation_matrix

    def check_new_position_correlation(self, new_symbol, existing_symbols, max_correlation=0.7):
        """Check if adding new position would create too much correlation"""

        if not existing_symbols:
            print(f"  [OK] No existing positions, {new_symbol} approved")
            return True

        print(f"\n{'='*80}")
        print(f"CORRELATION CHECK: {new_symbol}")
        print(f"Against {len(existing_symbols)} existing positions")
        print(f"{'='*80}")

        # Get correlation matrix including new symbol
        all_symbols = existing_symbols + [new_symbol]
        corr_matrix = self.get_correlation_matrix(all_symbols, lookback_days=60)

        if corr_matrix is None:
            print(f"  [WARNING] Could not calculate correlations, allowing trade")
            return True

        # Check correlations between new symbol and existing
        high_correlations = []

        for existing_symbol in existing_symbols:
            if existing_symbol in corr_matrix.index and new_symbol in corr_matrix.columns:
                corr = corr_matrix.loc[existing_symbol, new_symbol]

                if abs(corr) > max_correlation:
                    high_correlations.append({
                        'symbol': existing_symbol,
                        'correlation': corr
                    })
                    print(f"  [WARNING] High correlation with {existing_symbol}: {corr:.3f}")

        if high_correlations:
            print(f"\n  [REJECT] {len(high_correlations)} high correlations detected (>{max_correlation})")
            print(f"  Reduce exposure or wait for diversification opportunity")
            return False
        else:
            print(f"\n  [APPROVE] Correlations within acceptable range (<{max_correlation})")
            return True

    def get_portfolio_diversification_score(self, symbols):
        """Calculate overall portfolio diversification score (0-100)"""

        corr_matrix = self.get_correlation_matrix(symbols)

        if corr_matrix is None or len(corr_matrix) < 2:
            return 100  # Single position = fully diversified

        # Get average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = corr_matrix.where(mask).stack().mean()

        # Score: 100 = no correlation, 0 = perfect correlation
        diversification_score = (1 - abs(avg_correlation)) * 100

        print(f"\n{'='*80}")
        print(f"PORTFOLIO DIVERSIFICATION SCORE")
        print(f"{'='*80}")
        print(f"Positions: {len(symbols)}")
        print(f"Average Correlation: {avg_correlation:.3f}")
        print(f"Diversification Score: {diversification_score:.1f}/100")

        if diversification_score > 80:
            print(f"[EXCELLENT] Well diversified portfolio")
        elif diversification_score > 60:
            print(f"[GOOD] Acceptable diversification")
        elif diversification_score > 40:
            print(f"[MODERATE] Consider adding uncorrelated positions")
        else:
            print(f"[POOR] High correlation risk - reduce concentrated positions")

        return diversification_score

    def identify_clusters(self, symbols, correlation_threshold=0.6):
        """Identify groups of highly correlated stocks"""

        corr_matrix = self.get_correlation_matrix(symbols)

        if corr_matrix is None:
            return []

        print(f"\n{'='*80}")
        print(f"IDENTIFYING CORRELATION CLUSTERS")
        print(f"Threshold: {correlation_threshold}")
        print(f"{'='*80}")

        clusters = []
        processed = set()

        for i, symbol1 in enumerate(symbols):
            if symbol1 in processed:
                continue

            cluster = [symbol1]

            for symbol2 in symbols[i+1:]:
                if symbol2 in processed:
                    continue

                if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                    corr = corr_matrix.loc[symbol1, symbol2]

                    if abs(corr) >= correlation_threshold:
                        cluster.append(symbol2)
                        processed.add(symbol2)

            if len(cluster) > 1:
                clusters.append(cluster)
                processed.add(symbol1)

                # Calculate average correlation within cluster
                cluster_corrs = []
                for s1 in cluster:
                    for s2 in cluster:
                        if s1 != s2 and s1 in corr_matrix.index and s2 in corr_matrix.columns:
                            cluster_corrs.append(corr_matrix.loc[s1, s2])

                avg_cluster_corr = np.mean(cluster_corrs) if cluster_corrs else 0

                print(f"\n  CLUSTER {len(clusters)}: {len(cluster)} stocks")
                print(f"    Symbols: {', '.join(cluster)}")
                print(f"    Avg Correlation: {avg_cluster_corr:.3f}")
                print(f"    [WARNING] Consider limiting exposure to this cluster")

        if not clusters:
            print(f"  [OK] No high-correlation clusters detected")

        return clusters


def test_correlation_analyzer():
    """Test correlation analyzer"""

    analyzer = PortfolioCorrelationAnalyzer()

    # Test with tech stocks (expect high correlation)
    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD']

    print("="*80)
    print("TEST: Tech Stock Correlations")
    print("="*80)

    # Get diversification score
    analyzer.get_portfolio_diversification_score(tech_symbols)

    # Identify clusters
    analyzer.identify_clusters(tech_symbols, correlation_threshold=0.6)

    # Check if adding another tech stock is risky
    analyzer.check_new_position_correlation('INTC', tech_symbols, max_correlation=0.7)

    # Check if adding defensive stock is better
    analyzer.check_new_position_correlation('PG', tech_symbols, max_correlation=0.7)


if __name__ == "__main__":
    test_correlation_analyzer()
