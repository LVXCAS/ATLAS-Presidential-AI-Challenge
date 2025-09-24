"""
LIVE VALIDATION SYSTEM
======================
Track real-world performance of ML predictions vs actual market outcomes
Weekend setup for Monday market validation
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import os
from ml_enhanced_trading_system import MLEnhancedTradingSystem
from final_edge_strategy import FinalEdgeStrategy

class LiveValidationSystem:
    """System to track and validate ML predictions against real outcomes."""
    
    def __init__(self):
        self.db_path = "trading_validation.db"
        self.ml_system = MLEnhancedTradingSystem()
        self.edge_system = FinalEdgeStrategy()
        
        print("LIVE VALIDATION SYSTEM")
        print("=" * 50)
        print("Purpose: Prove ML models work in real markets")
        print("Starting: Monday market open")
        print("Goal: 30 days of validation data")
        print("=" * 50)
        
        self.setup_database()
    
    def setup_database(self):
        """Set up SQLite database for tracking."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            ml_confidence REAL,
            predicted_direction TEXT,
            current_price REAL,
            prediction_source TEXT,
            notes TEXT
        )
        """)
        
        # Outcomes table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            evaluation_date TEXT,
            outcome_price REAL,
            actual_move_pct REAL,
            prediction_correct INTEGER,
            profit_loss_pct REAL,
            days_held INTEGER,
            notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
        """)
        
        # Daily performance summary
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance (
            date TEXT PRIMARY KEY,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy_rate REAL,
            avg_profit_loss REAL,
            best_trade REAL,
            worst_trade REAL,
            notes TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        
        print("Validation database initialized")
    
    def log_prediction(self, symbol, strategy, ml_confidence, predicted_direction, 
                      current_price, prediction_source, notes=""):
        """Log a trading prediction."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO predictions 
        (timestamp, symbol, strategy, ml_confidence, predicted_direction, 
         current_price, prediction_source, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            symbol,
            strategy, 
            ml_confidence,
            predicted_direction,
            current_price,
            prediction_source,
            notes
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"LOGGED PREDICTION: {symbol} {strategy} (Confidence: {ml_confidence:.1%})")
        
        return prediction_id
    
    def evaluate_prediction(self, prediction_id, days_to_check=5):
        """Evaluate a prediction after specified days."""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get prediction details
        prediction = pd.read_sql_query("""
        SELECT * FROM predictions WHERE id = ?
        """, conn, params=(prediction_id,))
        
        if prediction.empty:
            conn.close()
            return
        
        pred = prediction.iloc[0]
        
        try:
            # Get current price
            ticker = yf.Ticker(pred['symbol'])
            current_data = ticker.history(period="5d")
            
            if current_data.empty:
                conn.close()
                return
            
            current_price = current_data['Close'].iloc[-1]
            entry_price = pred['current_price']
            
            # Calculate actual move
            actual_move_pct = (current_price - entry_price) / entry_price
            
            # Determine if prediction was correct
            prediction_correct = 0
            profit_loss_pct = 0
            
            if pred['strategy'] == 'ML_VOLATILITY_BREAKOUT':
                # For volatility, we need big moves (either direction)
                if abs(actual_move_pct) >= 0.05:  # 5% move threshold
                    prediction_correct = 1
                    profit_loss_pct = abs(actual_move_pct) * 100  # Straddle profits from big moves
                else:
                    profit_loss_pct = -30  # Typical straddle loss from time decay
            
            elif pred['strategy'] == 'ML_MOMENTUM':
                # For momentum, direction matters
                if pred['predicted_direction'] == 'BULLISH' and actual_move_pct > 0.03:
                    prediction_correct = 1
                    profit_loss_pct = actual_move_pct * 300  # Options leverage
                elif pred['predicted_direction'] == 'BEARISH' and actual_move_pct < -0.03:
                    prediction_correct = 1
                    profit_loss_pct = abs(actual_move_pct) * 300
                else:
                    profit_loss_pct = -50  # Typical options loss
            
            # Log outcome
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO outcomes 
            (prediction_id, evaluation_date, outcome_price, actual_move_pct, 
             prediction_correct, profit_loss_pct, days_held, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                datetime.now().isoformat(),
                current_price,
                actual_move_pct * 100,
                prediction_correct,
                profit_loss_pct,
                days_to_check,
                f"Move: {actual_move_pct*100:.1f}% in {days_to_check} days"
            ))
            
            conn.commit()
            
            print(f"EVALUATED: {pred['symbol']} - {'✓ CORRECT' if prediction_correct else '✗ WRONG'}")
            print(f"  Actual move: {actual_move_pct*100:.1f}%")
            print(f"  Est. P&L: {profit_loss_pct:.1f}%")
            
        except Exception as e:
            print(f"Error evaluating prediction {prediction_id}: {e}")
        
        conn.close()
    
    def run_daily_scan_and_log(self):
        """Run daily scan and log all predictions."""
        
        print(f"\nDAILY VALIDATION SCAN: {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        predictions_today = 0
        
        # Get ML predictions
        try:
            ml_signals = self.ml_system.generate_ml_trading_signals()
            
            for signal in ml_signals:
                prediction_id = self.log_prediction(
                    symbol=signal['symbol'],
                    strategy=signal['strategy'],
                    ml_confidence=signal['ml_confidence'],
                    predicted_direction=signal.get('direction', 'NEUTRAL'),
                    current_price=signal['current_price'],
                    prediction_source='ML_RandomForest_95%',
                    notes=f"Expected accuracy: {signal['expected_accuracy']:.1f}%"
                )
                predictions_today += 1
        
        except Exception as e:
            print(f"Error getting ML predictions: {e}")
        
        # Get traditional edge predictions
        try:
            edge_signals = self.edge_system.generate_live_trading_signals()
            
            for signal in edge_signals:
                prediction_id = self.log_prediction(
                    symbol=signal['symbol'],
                    strategy=signal['strategy'],
                    ml_confidence=signal.get('confidence', 0.76),  # Use baseline
                    predicted_direction=signal.get('direction', 'NEUTRAL'),
                    current_price=signal['current_price'],
                    prediction_source='Traditional_76%_Baseline',
                    notes=f"Edge score: {signal.get('edge_score', 0)}"
                )
                predictions_today += 1
        
        except Exception as e:
            print(f"Error getting edge predictions: {e}")
        
        print(f"\nTOTAL PREDICTIONS LOGGED TODAY: {predictions_today}")
        
        # Evaluate old predictions (5 days old)
        self.evaluate_old_predictions()
        
        return predictions_today
    
    def evaluate_old_predictions(self):
        """Evaluate predictions that are 5 days old."""
        
        five_days_ago = (datetime.now() - timedelta(days=5)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        
        # Find predictions to evaluate
        unevaluated = pd.read_sql_query("""
        SELECT p.id FROM predictions p
        LEFT JOIN outcomes o ON p.id = o.prediction_id
        WHERE p.timestamp < ? AND o.prediction_id IS NULL
        """, conn, params=(five_days_ago,))
        
        print(f"\nEVALUATING {len(unevaluated)} OLD PREDICTIONS...")
        
        for _, row in unevaluated.iterrows():
            self.evaluate_prediction(row['id'])
        
        conn.close()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        
        conn = sqlite3.connect(self.db_path)
        
        # Overall stats
        overall_stats = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_predictions,
            AVG(CASE WHEN o.prediction_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy_rate,
            AVG(o.profit_loss_pct) as avg_profit_loss,
            MAX(o.profit_loss_pct) as best_trade,
            MIN(o.profit_loss_pct) as worst_trade
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        """, conn)
        
        # By strategy
        strategy_stats = pd.read_sql_query("""
        SELECT 
            p.strategy,
            COUNT(*) as predictions,
            AVG(CASE WHEN o.prediction_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(o.profit_loss_pct) as avg_return
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        GROUP BY p.strategy
        ORDER BY accuracy DESC
        """, conn)
        
        # By confidence level
        confidence_stats = pd.read_sql_query("""
        SELECT 
            CASE 
                WHEN p.ml_confidence >= 0.95 THEN 'Very High (95%+)'
                WHEN p.ml_confidence >= 0.90 THEN 'High (90-95%)'
                WHEN p.ml_confidence >= 0.80 THEN 'Medium (80-90%)'
                ELSE 'Low (<80%)'
            END as confidence_bucket,
            COUNT(*) as predictions,
            AVG(CASE WHEN o.prediction_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        GROUP BY confidence_bucket
        ORDER BY accuracy DESC
        """, conn)
        
        conn.close()
        
        # Generate report
        report = f"""
LIVE VALIDATION PERFORMANCE REPORT
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE:
"""
        
        if not overall_stats.empty and overall_stats.iloc[0]['total_predictions'] > 0:
            stats = overall_stats.iloc[0]
            report += f"""
Total Predictions: {stats['total_predictions']:.0f}
Accuracy Rate: {(stats['accuracy_rate'] or 0)*100:.1f}%
Average Return: {stats['avg_profit_loss'] or 0:.1f}%
Best Trade: +{stats['best_trade'] or 0:.1f}%
Worst Trade: {stats['worst_trade'] or 0:.1f}%
"""
        else:
            report += "No completed predictions yet - need 5+ days of data"
        
        report += f"""

BY STRATEGY:
{strategy_stats.to_string(index=False) if not strategy_stats.empty else 'No data yet'}

BY CONFIDENCE LEVEL:
{confidence_stats.to_string(index=False) if not confidence_stats.empty else 'No data yet'}

VALIDATION STATUS:
- ML Model Claims: 95% accuracy
- Baseline Claims: 76% accuracy
- Actual Results: {'TBD - collecting data' if overall_stats.empty or overall_stats.iloc[0]['total_predictions'] == 0 else f"{(overall_stats.iloc[0]['accuracy_rate'] or 0)*100:.1f}%"}
"""
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'validation_report_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\nPERFORMANCE REPORT GENERATED: {filename}")
        print(report)
        
        return report
    
    def setup_monday_automation(self):
        """Create automation scripts for Monday morning."""
        
        # Daily runner script
        daily_script = """#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_validation_system import LiveValidationSystem

if __name__ == "__main__":
    print("MONDAY MARKET VALIDATION - STARTING...")
    
    validator = LiveValidationSystem()
    predictions = validator.run_daily_scan_and_log()
    
    if predictions > 0:
        print(f"SUCCESS: {predictions} predictions logged for validation")
    else:
        print("No high-confidence predictions today - models being selective")
    
    # Generate report if we have data
    validator.generate_performance_report()
"""
        
        with open('monday_validation_runner.py', 'w') as f:
            f.write(daily_script)
        
        # Create batch file for Windows automation
        batch_script = f"""@echo off
echo MONDAY MARKET VALIDATION STARTING...
cd /d "{os.getcwd()}"
python monday_validation_runner.py
echo VALIDATION COMPLETE - CHECK RESULTS
pause
"""
        
        with open('run_monday_validation.bat', 'w') as f:
            f.write(batch_script)
        
        print("MONDAY AUTOMATION CREATED:")
        print("  - monday_validation_runner.py")
        print("  - run_monday_validation.bat")
        print("\nRUN EVERY MORNING: python monday_validation_runner.py")

def main():
    """Set up live validation system for Monday."""
    
    validator = LiveValidationSystem()
    
    print("\nSETTING UP MONDAY MARKET VALIDATION...")
    
    # Set up automation
    validator.setup_monday_automation()
    
    # Generate initial report (will be empty but shows structure)
    validator.generate_performance_report()
    
    print(f"\nWEEKEND SETUP COMPLETE!")
    print(f"READY FOR MONDAY MARKET VALIDATION")
    print(f"")
    print(f"MONDAY MORNING TODO:")
    print(f"1. Run: python monday_validation_runner.py")
    print(f"2. Track predictions vs actual outcomes")
    print(f"3. Build proof database over 30 days")
    print(f"")
    print(f"GOAL: Prove if 95% ML accuracy claims are real!")

if __name__ == "__main__":
    main()