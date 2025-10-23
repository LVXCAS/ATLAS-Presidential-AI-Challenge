#!/usr/bin/env python3
"""
Comprehensive Enhanced Model Training
Combines extended historical data + existing trade history for superior AI models
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import enhanced models
from ai.enhanced_models import EnhancedTradingModel, MarketRegimeDetector

class ComprehensiveTrainer:
    """Train enhanced AI models with maximum available data"""

    def __init__(self):
        # Extended symbol universe for better training
        self.symbols = [
            # Major Indices & ETFs

        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        market_data = {}
        success_count = 0

        for i, symbol in enumerate(self.symbols, 1):
            try:
                print(f"[{i}/{len(self.symbols)}] Fetching {symbol}...", end=' ')
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, auto_adjust=True)

                if not data.empty and len(data) > 252:  # At least 1 year
                    market_data[symbol] = data
                    success_count += 1
                    print(f"OK {len(data)} days")
                else:
                    print(f"SKIP Insufficient data")

            except Exception as e:
                print(f"ERROR: {e}")

        print(f"\nSuccessfully loaded {success_count}/{len(self.symbols)} symbols")
        print(f"Total data points: {sum(len(df) for df in market_data.values()):,}")

        return market_data

    def load_existing_trade_history(self) -> pd.DataFrame:
        """Load 201 trades from trading_performance.db"""
        print(f"\n{'='*70}")
        print("LOADING EXISTING TRADE HISTORY")
        print(f"{'='*70}")

        try:
            conn = sqlite3.connect('trading_performance.db')
            trades_df = pd.read_sql_query("""
                SELECT
                    symbol,
                    strategy,
                    entry_time,
                    exit_time,
                    entry_confidence,
                    entry_price,
                    exit_price,
                    pnl,
                    max_profit,
                    max_loss,
                    market_conditions,
                    exit_reason,
                    days_held,
                    win
                FROM trades
                WHERE exit_time IS NOT NULL
                ORDER BY entry_time DESC
            """, conn)
            conn.close()

            print(f"Loaded {len(trades_df)} completed trades")
            print(f"Win rate: {trades_df['win'].mean()*100:.1f}%")
            print(f"Total P&L: ${trades_df['pnl'].sum():.2f}")
            print(f"Avg days held: {trades_df['days_held'].mean():.1f}")

            # Parse market conditions JSON
            trades_df['market_conditions_dict'] = trades_df['market_conditions'].apply(
                lambda x: json.loads(x) if x else {}
            )

            return trades_df

        except Exception as e:
            print(f"Error loading trade history: {e}")
            return pd.DataFrame()

    def enrich_trades_with_market_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to each trade from historical data"""
        print(f"\n{'='*70}")
        print("ENRICHING TRADES WITH TECHNICAL FEATURES")
        print(f"{'='*70}")

        enriched_trades = []

        for idx, trade in trades_df.iterrows():
            try:
                symbol = trade['symbol']
                entry_date = pd.to_datetime(trade['entry_time'])

                # Download data around trade entry
                start = (entry_date - timedelta(days=60)).strftime('%Y-%m-%d')
                end = (entry_date + timedelta(days=5)).strftime('%Y-%m-%d')

                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start, end=end)

                if len(hist) > 20:
                    # Calculate features at entry point
                    features = self.trading_model.prepare_trading_features(hist)
                    entry_features = features.iloc[-1].to_dict()

                    # Combine trade data with features
                    enriched_trade = trade.to_dict()
                    enriched_trade['features'] = entry_features
                    enriched_trades.append(enriched_trade)

                    if (idx + 1) % 20 == 0:
                        print(f"Processed {idx + 1}/{len(trades_df)} trades...")

            except Exception as e:
                continue

        print(f"Enriched {len(enriched_trades)}/{len(trades_df)} trades with technical features")

        return pd.DataFrame(enriched_trades)

    def train_comprehensive_models(self):
        """Train models with all available data"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE ENHANCED MODEL TRAINING")
        print(f"{'='*70}\n")

        # 1. Load historical market data (5 years, 80+ symbols)
        market_data = self.load_historical_market_data(years=5)

        if not market_data:
            print("No market data available. Cannot train models.")
            return

        # 2. Load existing trade history
        trades_df = self.load_existing_trade_history()

        # 3. Enrich trades with technical features
        enriched_trades = pd.DataFrame()
        if not trades_df.empty:
            enriched_trades = self.enrich_trades_with_market_data(trades_df)

        # 4. Train market regime detector
        print(f"\n{'='*70}")
        print("TRAINING MARKET REGIME DETECTOR")
        print(f"{'='*70}")
        regime_results = self.regime_detector.train_regime_detector(market_data)
        print(f"Regime detector trained:")
        for metric, value in regime_results.items():
            print(f"  {metric}: {value:.4f}")

        # 5. Train trading models (RandomForest, XGBoost, Deep Learning)
        print(f"\n{'='*70}")
        print("TRAINING TRADING MODELS (RF, XGBoost, DL)")
        print(f"{'='*70}")
        trading_results = self.trading_model.train_trading_models(market_data)
        print(f"Trading models trained:")
        for model_name, metrics in trading_results.items():
            if isinstance(metrics, dict):
                print(f"  {model_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")

        # 6. Validate with real trade history
        if not enriched_trades.empty:
            print(f"\n{'='*70}")
            print("VALIDATING WITH REAL TRADE HISTORY")
            print(f"{'='*70}")
            self.validate_with_trade_history(enriched_trades)

        # 7. Save results
        print(f"\n{'='*70}")
        print("SAVING MODELS")
        print(f"{'='*70}")

        import os
        os.makedirs('models', exist_ok=True)

        results = {
            'training_date': datetime.now().isoformat(),
            'symbols_trained': len(market_data),
            'total_datapoints': sum(len(df) for df in market_data.values()),
            'real_trades_used': len(enriched_trades),
            'regime_results': regime_results,
        }

        with open('models/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Models saved successfully")
        print(f"Training results saved to: models/training_results.json")

        return results

    def validate_with_trade_history(self, enriched_trades: pd.DataFrame):
        """Validate model predictions against actual trade outcomes"""
        print(f"Validating model predictions against {len(enriched_trades)} real trades...")

        correct_predictions = 0
        total_predictions = 0

        for _, trade in enriched_trades.iterrows():
            if 'features' not in trade or not trade['features']:
                continue

            # Convert features to DataFrame for prediction
            features_df = pd.DataFrame([trade['features']])

            try:
                # Get model prediction
                prediction = self.trading_model.predict_ensemble(features_df)
                predicted_win = prediction.get('prediction', 0) > 0.5
                actual_win = bool(trade['win'])

                if predicted_win == actual_win:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                continue

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"Model accuracy on real trades: {accuracy*100:.1f}% ({correct_predictions}/{total_predictions})")
        else:
            print("Could not validate - insufficient enriched data")

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║          COMPREHENSIVE ENHANCED MODEL TRAINING                     ║
║                                                                    ║
║  This will train advanced AI models using:                        ║
║  - 5 years of historical data (80+ symbols)                       ║
║  - Your 201 existing trades for validation                        ║
║  - 60+ technical features per data point                          ║
║  - Market regime detection (Bull/Bear/Sideways/Volatile)          ║
║  - Ensemble ML: RandomForest + XGBoost + Deep Learning            ║
║                                                                    ║
║  Estimated time: 10-20 minutes                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    trainer = ComprehensiveTrainer()
    results = trainer.train_comprehensive_models()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Models are ready to use in OPTIONS_BOT")
    print("2. Models will continue learning from live trades")
    print("3. Re-run this script periodically to retrain with more data")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()