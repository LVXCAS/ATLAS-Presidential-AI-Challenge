import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ForexMLBot:
    def __init__(self, lookback=60, prediction_horizon=1):
        """
        Initialize the Forex ML Bot
        
        Args:
            lookback: Number of previous timesteps to use for prediction
            prediction_horizon: Number of timesteps ahead to predict
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def create_features(self, df):
        """
        Create technical indicators and features
        """
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
        
        # Volatility
        data['volatility_10'] = data['returns'].rolling(window=10).std()
        data['volatility_30'] = data['returns'].rolling(window=30).std()
        
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Price position relative to bands
        data['price_to_bb'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Momentum
        data['momentum_5'] = data['close'] - data['close'].shift(5)
        data['momentum_10'] = data['close'] - data['close'].shift(10)
        
        # Volume features (if available)
        if 'volume' in data.columns:
            data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_10']
        
        return data
    
    def prepare_data(self, df):
        """
        Prepare sequences for LSTM training
        """
        # Create features
        data = self.create_features(df)
        
        # Drop NaN values
        data = data.dropna()
        
        # Select features for training (exclude timestamp and OHLCV columns)
        exclude_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(data[feature_columns])
        
        # Scale target (close price)
        target_scaled = self.scaler.fit_transform(data[['close']])
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(data) - self.prediction_horizon):
            X.append(features_scaled[i-self.lookback:i])
            y.append(target_scaled[i + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y), data
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, df, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model
        """
        print("Preparing data...")
        X, y, processed_data = self.prepare_data(df)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        print("Building model...")
        self.model = self.build_model((X.shape[1], X.shape[2]))
        print(self.model.summary())
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history, processed_data
    
    def predict(self, X):
        """
        Make predictions
        """
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        return predictions
    
    def backtest(self, df, initial_balance=10000, position_size=0.1):
        """
        Simple backtest strategy
        """
        X, y, processed_data = self.prepare_data(df)
        
        # Make predictions
        predictions = self.predict(X)
        actual = self.scaler.inverse_transform(y)
        
        # Trading logic
        balance = initial_balance
        position = 0
        trades = []
        
        for i in range(1, len(predictions)):
            # Simple strategy: buy if predicted price > current, sell if <
            current_price = actual[i-1][0]
            predicted_price = predictions[i][0]
            
            # Calculate signal strength
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.001 and position <= 0:  # Buy signal
                position = (balance * position_size) / current_price
                balance -= position * current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'position': position,
                    'balance': balance
                })
                
            elif price_change < -0.001 and position > 0:  # Sell signal
                balance += position * current_price
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'position': position,
                    'balance': balance
                })
                position = 0
        
        # Close any open position
        if position > 0:
            balance += position * actual[-1][0]
            position = 0
        
        final_balance = balance
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'num_trades': len(trades),
            'predictions': predictions,
            'actual': actual,
            'trades': trades
        }
    
    def plot_results(self, history, backtest_results):
        """
        Visualize training and backtest results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Predictions vs Actual
        predictions = backtest_results['predictions'][-500:]
        actual = backtest_results['actual'][-500:]
        axes[0, 1].plot(actual, label='Actual', alpha=0.7)
        axes[0, 1].plot(predictions, label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual (Last 500 points)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Prediction Error
        error = predictions - actual
        axes[1, 0].hist(error, bins=50, edgecolor='black')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Portfolio Value
        trades = backtest_results['trades']
        if trades:
            trade_balances = [t['balance'] for t in trades]
            axes[1, 1].plot(trade_balances)
            axes[1, 1].set_title('Portfolio Value Over Trades')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Balance ($)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print backtest results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Balance: ${backtest_results['initial_balance']:,.2f}")
        print(f"Final Balance: ${backtest_results['final_balance']:,.2f}")
        print(f"Total Return: {backtest_results['total_return']:.2f}%")
        print(f"Number of Trades: {backtest_results['num_trades']}")
        print("="*50)

# Example usage with synthetic data (replace with real forex data)
def generate_sample_data(n_samples=10000):
    """
    Generate sample forex data for demonstration
    Replace this with real forex data from Kaggle datasets
    """
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    # Simulate price movement
    returns = np.random.normal(0.0001, 0.01, n_samples)
    price = 1.1000  # Starting EUR/USD price
    prices = [price]
    
    for ret in returns[1:]:
        price = price * (1 + ret)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return df

# Main execution
if __name__ == "__main__":
    print("Forex ML Trading Bot")
    print("="*50)
    
    # Generate or load data
    # For Kaggle, you would load real forex data like:
    # df = pd.read_csv('/kaggle/input/forex-data/eurusd.csv')
    df = generate_sample_data(10000)
    
    print(f"Data loaded: {len(df)} samples")
    
    # Initialize bot
    bot = ForexMLBot(lookback=60, prediction_horizon=1)
    
    # Train model
    history, processed_data = bot.train(df, epochs=50, batch_size=64)
    
    # Backtest
    backtest_results = bot.backtest(df)
    
    # Visualize results
    bot.plot_results(history, backtest_results)
    
    # Save model
    bot.model.save('forex_ml_model.h5')
    print("\nModel saved as 'forex_ml_model.h5'")