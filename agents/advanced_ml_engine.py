#!/usr/bin/env python3
"""
Advanced Machine Learning Engine for Options Trading
Integrates technical analysis, volatility surfaces, macro events, and options flow
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import json
import requests
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries (install if needed)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not installed. Run: pip install ta-lib")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")

@dataclass
class TechnicalFeatures:
    """Technical analysis features"""
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5  # Where price sits in BB bands
    price_momentum_5d: float = 0.0
    price_momentum_20d: float = 0.0
    volume_ratio_5d: float = 1.0
    volatility_rank: float = 50.0  # Percentile rank of current volatility

@dataclass
class VolatilityFeatures:
    """Volatility surface features"""
    iv_rank: float = 50.0  # Implied volatility rank
    iv_percentile: float = 50.0  # IV percentile
    hv_iv_ratio: float = 1.0  # Historical vs Implied volatility
    term_structure_slope: float = 0.0  # Short vs long term IV
    skew_25delta: float = 0.0  # 25-delta skew
    smile_convexity: float = 0.0  # Volatility smile shape

@dataclass
class MacroFeatures:
    """Macro and event features"""
    days_to_earnings: int = 999  # Days until earnings
    fed_meeting_proximity: int = 999  # Days to Fed meeting
    earnings_surprise_history: float = 0.0  # Historical earnings surprise
    market_stress_index: float = 0.0  # VIX relative level
    sector_relative_strength: float = 0.0  # vs SPY performance

@dataclass
class OptionsFlowFeatures:
    """Options flow features"""
    put_call_ratio: float = 1.0
    unusual_call_volume: bool = False
    unusual_put_volume: bool = False
    net_gamma_exposure: float = 0.0
    dealer_positioning: float = 0.0  # Estimated dealer positioning
    flow_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL

class AdvancedMLEngine:
    """Advanced machine learning engine with comprehensive features"""
    
    def __init__(self, db_path: str = "advanced_ml_data.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_importance = {}
        self.last_prediction = None
        
        # API keys (set these as environment variables)
        self.alpha_vantage_key = None  # Set via environment variable
        self.news_api_key = None       # Set via environment variable
        
        self.initialize_database()
        self.load_model()
    
    def initialize_database(self):
        """Initialize database for ML features and predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_features (
                timestamp TEXT,
                symbol TEXT,
                
                -- Technical features
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                bb_position REAL,
                price_momentum_5d REAL,
                price_momentum_20d REAL,
                volume_ratio_5d REAL,
                volatility_rank REAL,
                
                -- Volatility features
                iv_rank REAL,
                hv_iv_ratio REAL,
                term_structure_slope REAL,
                skew_25delta REAL,
                
                -- Macro features
                days_to_earnings INTEGER,
                fed_meeting_proximity INTEGER,
                market_stress_index REAL,
                
                -- Options flow features
                put_call_ratio REAL,
                unusual_call_volume INTEGER,
                unusual_put_volume INTEGER,
                flow_sentiment TEXT,
                
                -- Target variables
                actual_move_1d REAL,
                actual_move_5d REAL,
                profitable_trade INTEGER,
                
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                date TEXT PRIMARY KEY,
                accuracy REAL,
                precision REAL,
                recall REAL,
                feature_importance TEXT,
                model_version INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_technical_features(self, symbol: str, lookback_days: int = 100) -> TechnicalFeatures:
        """Extract technical analysis features"""
        try:
            # Get price data (using yfinance as fallback)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{lookback_days}d")
            
            if data.empty:
                return TechnicalFeatures()
            
            prices = data['Close'].values
            volumes = data['Volume'].values
            high = data['High'].values
            low = data['Low'].values
            
            features = TechnicalFeatures()
            
            if TALIB_AVAILABLE and len(prices) >= 50:
                try:
                    # RSI
                    rsi = talib.RSI(prices, timeperiod=14)
                    features.rsi_14 = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
                    
                    # MACD
                    macd, macd_signal, macd_hist = talib.MACD(prices)
                    features.macd = macd[-1] if not np.isnan(macd[-1]) else 0.0
                    features.macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
                    features.macd_histogram = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                    if not np.isnan(bb_upper[-1]):
                        features.bb_upper = bb_upper[-1]
                        features.bb_middle = bb_middle[-1]
                        features.bb_lower = bb_lower[-1]
                        # Calculate position within bands
                        bb_range = bb_upper[-1] - bb_lower[-1]
                        if bb_range > 0:
                            features.bb_position = (prices[-1] - bb_lower[-1]) / bb_range
                    
                except Exception as ta_error:
                    print(f"TA-Lib error for {symbol}: {ta_error}")
            
            # Price momentum (manual calculation)
            if len(prices) >= 20:
                features.price_momentum_5d = (prices[-1] / prices[-6] - 1.0) if len(prices) >= 6 else 0.0
                features.price_momentum_20d = (prices[-1] / prices[-21] - 1.0)
            
            # Volume analysis
            if len(volumes) >= 5:
                recent_avg_volume = np.mean(volumes[-5:])
                historical_avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else recent_avg_volume
                features.volume_ratio_5d = recent_avg_volume / historical_avg_volume if historical_avg_volume > 0 else 1.0
            
            # Volatility rank (simplified)
            if len(prices) >= 50:
                returns = np.diff(np.log(prices))
                current_vol = np.std(returns[-20:]) * np.sqrt(252)
                historical_vols = [np.std(returns[i:i+20]) * np.sqrt(252) 
                                 for i in range(len(returns)-40, len(returns)-20)]
                if historical_vols:
                    features.volatility_rank = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols)) * 100
            
            return features
            
        except Exception as e:
            print(f"Error extracting technical features for {symbol}: {e}")
            return TechnicalFeatures()
    
    def extract_volatility_features(self, symbol: str) -> VolatilityFeatures:
        """Extract volatility surface features"""
        features = VolatilityFeatures()
        
        try:
            # This would integrate with options data provider
            # For now, using simplified calculations
            
            # Get current IV from options chain (simplified)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Try to get options data
            try:
                expiry_dates = ticker.options
                if expiry_dates:
                    # Get nearest expiry
                    options_chain = ticker.option_chain(expiry_dates[0])
                    calls = options_chain.calls
                    
                    if not calls.empty:
                        # Calculate average IV
                        avg_iv = calls['impliedVolatility'].mean()
                        features.iv_rank = min(100, max(0, avg_iv * 100))  # Simplified
                        
                        # HV vs IV (simplified)
                        hist_data = ticker.history(period="30d")
                        if len(hist_data) >= 20:
                            returns = np.diff(np.log(hist_data['Close'].values))
                            hv = np.std(returns) * np.sqrt(252)
                            features.hv_iv_ratio = hv / avg_iv if avg_iv > 0 else 1.0
                
            except Exception as options_error:
                pass  # Options data not available
                
        except Exception as e:
            print(f"Error extracting volatility features for {symbol}: {e}")
        
        return features
    
    def extract_macro_features(self, symbol: str) -> MacroFeatures:
        """Extract macro and event-driven features"""
        features = MacroFeatures()
        
        try:
            # Earnings proximity (simplified - would use earnings calendar API)
            # For now, estimate based on typical quarterly reporting
            today = datetime.now()
            
            # Simplified earnings estimation (every ~90 days)
            # Real implementation would use earnings calendar API
            days_since_year_start = (today - datetime(today.year, 1, 1)).days
            estimated_earnings_cycle = days_since_year_start % 90
            features.days_to_earnings = min(estimated_earnings_cycle, 90 - estimated_earnings_cycle)
            
            # Fed meeting proximity (simplified - 8 meetings per year)
            # Real implementation would use Fed calendar API
            fed_meeting_cycle = days_since_year_start % 45  # Approximate
            features.fed_meeting_proximity = min(fed_meeting_cycle, 45 - fed_meeting_cycle)
            
            # Market stress index (using VIX proxy)
            try:
                import yfinance as yf
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="5d")
                if not vix_data.empty:
                    current_vix = vix_data['Close'][-1]
                    # VIX levels: <20 (calm), 20-30 (elevated), >30 (high stress)
                    features.market_stress_index = min(100, max(0, (current_vix - 10) / 40 * 100))
            except:
                features.market_stress_index = 25.0  # Default moderate stress
            
        except Exception as e:
            print(f"Error extracting macro features: {e}")
        
        return features
    
    def extract_options_flow_features(self, symbol: str) -> OptionsFlowFeatures:
        """Extract options flow and sentiment features"""
        features = OptionsFlowFeatures()
        
        try:
            # This would integrate with options flow data provider
            # For now, using simplified calculations from available data
            
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            try:
                expiry_dates = ticker.options
                if expiry_dates:
                    options_chain = ticker.option_chain(expiry_dates[0])
                    calls = options_chain.calls
                    puts = options_chain.puts
                    
                    if not calls.empty and not puts.empty:
                        # Put/Call ratio
                        call_volume = calls['volume'].sum()
                        put_volume = puts['volume'].sum()
                        if call_volume > 0:
                            features.put_call_ratio = put_volume / call_volume
                        
                        # Unusual volume detection (simplified)
                        avg_call_volume = calls['volume'].mean()
                        avg_put_volume = puts['volume'].mean()
                        
                        # Check for unusual activity (>2x average)
                        features.unusual_call_volume = call_volume > (avg_call_volume * 2)
                        features.unusual_put_volume = put_volume > (avg_put_volume * 2)
                        
                        # Flow sentiment
                        if features.put_call_ratio < 0.7:
                            features.flow_sentiment = "BULLISH"
                        elif features.put_call_ratio > 1.3:
                            features.flow_sentiment = "BEARISH"
                        else:
                            features.flow_sentiment = "NEUTRAL"
            
            except Exception as options_error:
                pass  # Options data not available
                
        except Exception as e:
            print(f"Error extracting options flow features for {symbol}: {e}")
        
        return features
    
    def create_feature_vector(self, symbol: str) -> np.ndarray:
        """Create comprehensive feature vector for ML model"""
        # Extract all feature types
        tech_features = self.extract_technical_features(symbol)
        vol_features = self.extract_volatility_features(symbol)
        macro_features = self.extract_macro_features(symbol)
        flow_features = self.extract_options_flow_features(symbol)
        
        # Combine into feature vector
        feature_vector = [
            # Technical features
            tech_features.rsi_14,
            tech_features.macd,
            tech_features.macd_signal,
            tech_features.bb_position,
            tech_features.price_momentum_5d,
            tech_features.price_momentum_20d,
            tech_features.volume_ratio_5d,
            tech_features.volatility_rank,
            
            # Volatility features
            vol_features.iv_rank,
            vol_features.hv_iv_ratio,
            vol_features.term_structure_slope,
            
            # Macro features
            min(30, macro_features.days_to_earnings) / 30.0,  # Normalize
            min(30, macro_features.fed_meeting_proximity) / 30.0,  # Normalize
            macro_features.market_stress_index / 100.0,  # Normalize
            
            # Options flow features
            min(3.0, flow_features.put_call_ratio) / 3.0,  # Normalize and cap
            1.0 if flow_features.unusual_call_volume else 0.0,
            1.0 if flow_features.unusual_put_volume else 0.0,
            {"BULLISH": 1.0, "NEUTRAL": 0.5, "BEARISH": 0.0}[flow_features.flow_sentiment]
        ]
        
        return np.array(feature_vector)
    
    def predict_trade_success(self, symbol: str, strategy: str, confidence: float) -> Tuple[float, Dict]:
        """Predict trade success probability using ML model"""
        try:
            if not SKLEARN_AVAILABLE or self.model is None:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(symbol, strategy, confidence)
            
            # Get feature vector
            features = self.create_feature_vector(symbol)
            features_scaled = self.scaler.transform([features]) if self.scaler else [features]
            
            # Predict probability
            prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of success
            
            # Create explanation
            feature_names = [
                'rsi_14', 'macd', 'macd_signal', 'bb_position', 'momentum_5d', 'momentum_20d',
                'volume_ratio', 'vol_rank', 'iv_rank', 'hv_iv_ratio', 'term_structure',
                'days_to_earnings', 'fed_proximity', 'market_stress', 'put_call_ratio',
                'unusual_calls', 'unusual_puts', 'flow_sentiment'
            ]
            
            explanation = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(feature_names):
                        explanation[feature_names[i]] = {
                            'value': features[i],
                            'importance': importance
                        }
            
            self.last_prediction = {
                'symbol': symbol,
                'strategy': strategy,
                'predicted_prob': prob,
                'features': features,
                'explanation': explanation
            }
            
            return prob, explanation
            
        except Exception as e:
            print(f"ML prediction error for {symbol}: {e}")
            return self._rule_based_prediction(symbol, strategy, confidence)
    
    def _rule_based_prediction(self, symbol: str, strategy: str, confidence: float) -> Tuple[float, Dict]:
        """Fallback rule-based prediction when ML is not available"""
        # Simple rule-based system
        base_prob = confidence
        
        try:
            # Get basic technical features
            tech_features = self.extract_technical_features(symbol)
            
            # RSI adjustment
            if tech_features.rsi_14 > 70:  # Overbought
                base_prob *= 0.9 if strategy == "LONG_CALL" else 1.1
            elif tech_features.rsi_14 < 30:  # Oversold
                base_prob *= 1.1 if strategy == "LONG_CALL" else 0.9
            
            # Momentum adjustment
            if tech_features.price_momentum_5d > 0.02:  # Strong upward momentum
                base_prob *= 1.1 if strategy == "LONG_CALL" else 0.9
            elif tech_features.price_momentum_5d < -0.02:  # Strong downward momentum
                base_prob *= 0.9 if strategy == "LONG_CALL" else 1.1
            
            explanation = {
                'method': 'rule_based',
                'rsi_14': tech_features.rsi_14,
                'momentum_5d': tech_features.price_momentum_5d
            }
            
            return min(0.95, max(0.05, base_prob)), explanation
            
        except Exception:
            return confidence, {'method': 'fallback'}
    
    def train_model(self, min_samples: int = 50):
        """Train ML model on historical data"""
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available. Install with: pip install scikit-learn")
            return False
        
        try:
            # Load training data from database
            conn = sqlite3.connect(self.db_path)
            
            # Get training data
            query = '''
                SELECT * FROM ml_features 
                WHERE profitable_trade IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 1000
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < min_samples:
                print(f"Not enough training data: {len(df)} samples (need {min_samples})")
                return False
            
            # Prepare features and targets
            feature_columns = [
                'rsi_14', 'macd', 'macd_signal', 'bb_position', 'price_momentum_5d',
                'price_momentum_20d', 'volume_ratio_5d', 'volatility_rank', 'iv_rank',
                'hv_iv_ratio', 'term_structure_slope', 'days_to_earnings',
                'fed_meeting_proximity', 'market_stress_index', 'put_call_ratio',
                'unusual_call_volume', 'unusual_put_volume'
            ]
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) < 5:
                print("Not enough feature columns available")
                return False
            
            X = df[available_columns].fillna(0)
            y = df['profitable_trade']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model trained successfully!")
            print(f"Training samples: {len(X_train)}")
            print(f"Test accuracy: {accuracy:.3f}")
            print(f"Features used: {len(available_columns)}")
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(available_columns, self.model.feature_importances_))
                print("\nTop 5 most important features:")
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:5]:
                    print(f"  {feature}: {importance:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def load_model(self):
        """Load saved model if available"""
        # For now, we'll train a new model each time
        # In production, you'd save/load the model from disk
        pass
    
    def get_feature_analysis(self, symbol: str) -> Dict:
        """Get detailed feature analysis for a symbol"""
        try:
            tech_features = self.extract_technical_features(symbol)
            vol_features = self.extract_volatility_features(symbol)
            macro_features = self.extract_macro_features(symbol)
            flow_features = self.extract_options_flow_features(symbol)
            
            return {
                'technical': {
                    'rsi_14': tech_features.rsi_14,
                    'momentum_5d': tech_features.price_momentum_5d * 100,  # As percentage
                    'momentum_20d': tech_features.price_momentum_20d * 100,
                    'bb_position': tech_features.bb_position,
                    'volume_ratio': tech_features.volume_ratio_5d,
                    'volatility_rank': tech_features.volatility_rank
                },
                'volatility': {
                    'iv_rank': vol_features.iv_rank,
                    'hv_iv_ratio': vol_features.hv_iv_ratio
                },
                'macro': {
                    'days_to_earnings': macro_features.days_to_earnings,
                    'fed_proximity': macro_features.fed_meeting_proximity,
                    'market_stress': macro_features.market_stress_index
                },
                'options_flow': {
                    'put_call_ratio': flow_features.put_call_ratio,
                    'unusual_calls': flow_features.unusual_call_volume,
                    'unusual_puts': flow_features.unusual_put_volume,
                    'sentiment': flow_features.flow_sentiment
                }
            }
            
        except Exception as e:
            print(f"Feature analysis error for {symbol}: {e}")
            return {}

# Global instance
advanced_ml_engine = AdvancedMLEngine()