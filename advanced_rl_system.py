#!/usr/bin/env python3
"""
HIVE TRADE - Advanced RL Trading System
Deep reinforcement learning with proper neural networks and live data integration
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import asyncio
from datetime import datetime, timedelta
from collections import deque
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class MarketState:
    """Comprehensive market state representation"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    sma_5: float
    sma_10: float  
    sma_20: float
    rsi: float
    volatility: float
    momentum_5: float
    momentum_10: float
    market_trend: str
    volume_trend: float
    price_position: float

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size=14, hidden_size=256, num_actions=3):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_actions)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AdvancedRLSystem:
    """Advanced Deep RL Trading System"""
    
    def __init__(self, initial_balance=100000):
        print("=" * 70)
        print("HIVE TRADE - ADVANCED RL TRADING SYSTEM")
        print("Deep Reinforcement Learning with Neural Networks")
        print("=" * 70)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Using device: {self.device}")
        
        # Environment setup
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
        # RL Parameters
        self.state_size = 14  # Enhanced feature set
        self.action_size = 3  # BUY, SELL, HOLD
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory_size = 10000
        
        # Neural Networks
        self.main_network = DQNNetwork(self.state_size, 256, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, 256, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Training data
        self.symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.current_episode = 0
        self.total_reward = 0
        self.performance_history = []
        
        # Load training data
        self.training_data = self.load_training_data()
        
        print(f">> Advanced RL System Initialized")
        print(f"   Neural Network: DQN with {self.state_size} inputs")
        print(f"   Training Data: {len(self.training_data)} samples")
        print(f"   Virtual Balance: ${self.balance:,.2f}")
        
    def load_training_data(self):
        """Load comprehensive training data"""
        data = []
        
        # Load massive stock dataset
        try:
            with open('massive_stock_dataset.json', 'r') as f:
                stock_data = json.load(f)
                data.extend(stock_data['samples'])
                print(f"   Loaded {len(stock_data['samples'])} stock samples")
        except FileNotFoundError:
            print("   Warning: No massive stock dataset found")
        
        # Load enhanced crypto/stock data
        try:
            with open('enhanced_training_data.json', 'r') as f:
                enhanced_data = json.load(f)
                data.extend(enhanced_data['samples'])
                print(f"   Loaded {len(enhanced_data['samples'])} enhanced samples")
        except FileNotFoundError:
            print("   Warning: No enhanced training data found")
        
        return data
    
    def get_live_market_state(self, symbol):
        """Get comprehensive market state with proper feature engineering"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if len(hist) < 20:
                return None
            
            # Calculate technical indicators
            prices = hist['Close']
            volumes = hist['Volume']
            
            # Moving averages
            sma_5 = prices.rolling(5).mean().iloc[-1]
            sma_10 = prices.rolling(10).mean().iloc[-1] 
            sma_20 = prices.rolling(20).mean().iloc[-1]
            
            # RSI
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Volatility
            volatility = prices.rolling(10).std().iloc[-1] / prices.iloc[-1]
            
            # Momentum
            momentum_5 = (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]
            momentum_10 = (prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11]
            
            # Volume analysis
            volume_sma = volumes.rolling(10).mean().iloc[-1]
            volume_trend = volumes.iloc[-1] / volume_sma
            
            # Price position in range
            high_20 = hist['High'].rolling(20).max().iloc[-1]
            low_20 = hist['Low'].rolling(20).min().iloc[-1]
            price_position = (prices.iloc[-1] - low_20) / (high_20 - low_20)
            
            # Market trend classification
            if sma_5 > sma_10 > sma_20:
                market_trend = 1.0  # Strong uptrend
            elif sma_5 > sma_10:
                market_trend = 0.5  # Mild uptrend
            elif sma_5 < sma_10 < sma_20:
                market_trend = -1.0  # Strong downtrend
            else:
                market_trend = -0.5  # Mild downtrend
            
            return MarketState(
                timestamp=datetime.now(),
                symbol=symbol,
                price=prices.iloc[-1],
                volume=volumes.iloc[-1],
                sma_5=sma_5,
                sma_10=sma_10,
                sma_20=sma_20,
                rsi=rsi,
                volatility=volatility,
                momentum_5=momentum_5,
                momentum_10=momentum_10,
                market_trend=market_trend,
                volume_trend=volume_trend,
                price_position=price_position
            )
            
        except Exception as e:
            print(f"   Market state error for {symbol}: {e}")
            return None
    
    def state_to_vector(self, state: MarketState) -> np.ndarray:
        """Convert market state to normalized feature vector"""
        if state is None:
            return np.zeros(self.state_size)
        
        # Normalize all features to [-1, 1] range
        vector = np.array([
            np.clip((state.price / 50000) - 1, -1, 1),  # Normalized price
            np.clip((state.sma_5 / state.price) - 1, -0.5, 0.5),  # SMA5 ratio
            np.clip((state.sma_10 / state.price) - 1, -0.5, 0.5),  # SMA10 ratio  
            np.clip((state.sma_20 / state.price) - 1, -0.5, 0.5),  # SMA20 ratio
            np.clip((state.rsi / 100) - 0.5, -0.5, 0.5),  # RSI normalized
            np.clip(state.volatility * 10, 0, 1),  # Volatility
            np.clip(state.momentum_5 * 5, -1, 1),  # 5-day momentum
            np.clip(state.momentum_10 * 3, -1, 1),  # 10-day momentum
            state.market_trend,  # Trend indicator
            np.clip(state.volume_trend - 1, -2, 2),  # Volume trend
            state.price_position,  # Price position in range
            np.clip(state.volume / 1e7, 0, 1),  # Normalized volume
            # Portfolio features
            self.calculate_portfolio_pnl() / 10000,  # Portfolio P&L
            len(self.positions) / 10  # Number of positions
        ])
        
        return vector.astype(np.float32)
    
    def calculate_portfolio_pnl(self):
        """Calculate current portfolio P&L"""
        total_pnl = 0
        for symbol, position in self.positions.items():
            current_state = self.get_live_market_state(symbol)
            if current_state:
                current_pnl = (current_state.price - position['entry_price']) * position['qty']
                total_pnl += current_pnl
        return total_pnl
    
    def get_action(self, state_vector):
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()
    
    def execute_virtual_trade(self, symbol, action, amount=1000):
        """Execute trade in virtual environment"""
        state = self.get_live_market_state(symbol)
        if state is None:
            return 0
        
        reward = 0
        
        if action == 1:  # BUY
            if self.balance >= amount:
                qty = amount / state.price
                self.positions[symbol] = {
                    'qty': qty,
                    'entry_price': state.price,
                    'timestamp': datetime.now()
                }
                self.balance -= amount
                reward = 0.1  # Small reward for taking action
                
        elif action == 2:  # SELL
            if symbol in self.positions:
                position = self.positions[symbol]
                sale_value = position['qty'] * state.price
                pnl = sale_value - (position['qty'] * position['entry_price'])
                
                self.balance += sale_value
                reward = pnl / 1000  # Reward based on profit
                
                self.trade_history.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'pnl': pnl,
                    'timestamp': datetime.now()
                })
                
                del self.positions[symbol]
        
        # Risk penalty for too many positions
        if len(self.positions) > 5:
            reward -= 0.1
            
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay_experience(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    async def training_episode(self, episode_num):
        """Run one training episode"""
        print(f"\\n>> EPISODE {episode_num}")
        
        episode_reward = 0
        episode_trades = 0
        
        # Reset environment
        self.balance = self.initial_balance
        self.positions = {}
        
        for step in range(50):  # 50 steps per episode
            symbol = random.choice(self.symbols)
            state = self.get_live_market_state(symbol)
            
            if state is None:
                continue
                
            state_vector = self.state_to_vector(state)
            action = self.get_action(state_vector)
            
            reward = self.execute_virtual_trade(symbol, action, 1000)
            episode_reward += reward
            
            if action != 0:  # If not HOLD
                episode_trades += 1
            
            # Get next state
            await asyncio.sleep(0.1)  # Small delay
            next_state = self.get_live_market_state(symbol)
            next_state_vector = self.state_to_vector(next_state)
            
            # Store experience
            done = (step == 49)
            self.remember(state_vector, action, reward, next_state_vector, done)
            
            # Train the network
            if len(self.memory) > self.batch_size:
                loss = self.replay_experience()
        
        # Update target network every 10 episodes
        if episode_num % 10 == 0:
            self.update_target_network()
            
        # Calculate final portfolio value
        portfolio_value = self.balance + sum([
            pos['qty'] * self.get_live_market_state(symbol).price 
            for symbol, pos in self.positions.items()
            if self.get_live_market_state(symbol) is not None
        ])
        
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        print(f"   Episode Reward: {episode_reward:.2f}")
        print(f"   Trades Executed: {episode_trades}")
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Epsilon: {self.epsilon:.3f}")
        
        self.performance_history.append({
            'episode': episode_num,
            'reward': episode_reward,
            'trades': episode_trades,
            'portfolio_value': portfolio_value,
            'return': total_return,
            'timestamp': datetime.now().isoformat()
        })
        
        return total_return
    
    def get_model_confidence(self, state_vector):
        """Get model confidence for current prediction"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            softmax_probs = F.softmax(q_values, dim=1)
            confidence = torch.max(softmax_probs).item()
            return confidence
    
    def get_live_prediction(self, symbol):
        """Get live trading prediction for production use"""
        state = self.get_live_market_state(symbol)
        if state is None:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        state_vector = self.state_to_vector(state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            action_idx = q_values.argmax().item()
            confidence = self.get_model_confidence(state_vector)
        
        actions = ['HOLD', 'BUY', 'SELL']
        return {
            'action': actions[action_idx],
            'confidence': confidence,
            'q_values': q_values.cpu().numpy().tolist()[0],
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
    
    async def advanced_training_loop(self, num_episodes=100):
        """Main training loop"""
        print(f"\\n>> STARTING ADVANCED RL TRAINING")
        print(f"   Episodes: {num_episodes}")
        print(f"   Neural Network: Deep Q-Network")
        print(f"   Experience Replay: {self.memory_size} buffer")
        
        best_return = -float('inf')
        
        for episode in range(num_episodes):
            episode_return = await self.training_episode(episode + 1)
            
            if episode_return > best_return:
                best_return = episode_return
                # Save best model
                torch.save(self.main_network.state_dict(), 'best_rl_model.pth')
                print(f"   >> New best model saved! Return: {episode_return:.2%}")
            
            # Save progress
            if (episode + 1) % 10 == 0:
                self.save_training_progress()
                
        print(f"\\n>> TRAINING COMPLETE!")
        print(f"   Best Return: {best_return:.2%}")
        print(f"   Total Episodes: {num_episodes}")
        print(f"   Model Confidence: {self.get_model_confidence(np.random.randn(self.state_size)):.1%}")
        
    def save_training_progress(self):
        """Save training progress and model"""
        progress_data = {
            'performance_history': self.performance_history,
            'training_episodes': len(self.performance_history),
            'best_return': max([p['return'] for p in self.performance_history]) if self.performance_history else 0,
            'current_epsilon': self.epsilon,
            'model_state': 'saved_to_best_rl_model.pth',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('rl_training_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)

async def main():
    """Main function"""
    rl_system = AdvancedRLSystem()
    
    # Run training
    await rl_system.advanced_training_loop(num_episodes=50)
    
    # Test live predictions
    print("\\n>> TESTING LIVE PREDICTIONS:")
    for symbol in ['BTC-USD', 'ETH-USD', 'AAPL']:
        prediction = rl_system.get_live_prediction(symbol)
        print(f"   {symbol}: {prediction['action']} (Confidence: {prediction['confidence']:.1%})")

if __name__ == "__main__":
    asyncio.run(main())