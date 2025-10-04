"""
GPU AI TRADING AGENT
Self-learning reinforcement learning system powered by GTX 1660 Super
Adapts to market conditions and evolves trading strategies
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging
from dataclasses import dataclass
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configure GPU for RL training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_ai_trading_agent.log'),
        logging.StreamHandler()
    ]
)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TradingAction:
    """Trading action representation"""
    action_type: str  # 'buy', 'sell', 'hold'
    position_size: float  # 0.0 to 1.0 (percentage of portfolio)
    confidence: float
    reasoning: str

@dataclass
class MarketState:
    """Market state representation"""
    price_features: np.ndarray
    technical_indicators: np.ndarray
    volume_features: np.ndarray
    sentiment_features: np.ndarray
    portfolio_state: np.ndarray
    timestamp: datetime

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""

    def __init__(self, state_size=100, action_size=21, hidden_size=512):
        super(DQNNetwork, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.1)
        )

        # Dueling DQN architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous action space"""

    def __init__(self, state_size=100, action_size=3, hidden_size=512):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor output: [position_change, confidence, risk_tolerance]
        action = self.actor(shared_features)

        # Critic output: state value
        value = self.critic(shared_features)

        return action, value

class ReplayBuffer:
    """Experience replay buffer for stable learning"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """Trading environment for RL training"""

    def __init__(self, data: pd.DataFrame, initial_balance=100000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        # State configuration
        self.lookback_window = 20
        self.state_size = 100

        self.reset()

        # Action space: 21 discrete actions (-1.0 to 1.0 in 0.1 increments)
        self.action_space_size = 21
        self.actions = np.linspace(-1.0, 1.0, self.action_space_size)

        # Trading parameters
        self.transaction_cost = 0.001  # 0.1% per trade
        self.max_position = 1.0  # 100% of portfolio

    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # Current position (-1 to 1)
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.max_portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current market state as feature vector"""
        if self.current_step < self.lookback_window:
            return np.zeros(self.state_size)

        # Get recent data
        recent_data = self.data.iloc[self.current_step - self.lookback_window:self.current_step]

        # Price features
        prices = recent_data['Close'].values
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.log(prices[1:] / prices[:-1])

        # Technical indicators
        sma_5 = prices[-5:].mean() / prices[-1] if len(prices) >= 5 else 1.0
        sma_10 = prices[-10:].mean() / prices[-1] if len(prices) >= 10 else 1.0
        sma_20 = prices[-20:].mean() / prices[-1] if len(prices) >= 20 else 1.0

        # Volatility
        volatility = np.std(returns) if len(returns) > 1 else 0.0

        # Volume features
        volumes = recent_data['Volume'].values
        volume_ma = volumes.mean()
        volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1.0

        # RSI
        rsi = self._calculate_rsi(prices)

        # MACD
        macd, macd_signal = self._calculate_macd(prices)

        # Price position features
        high_20 = recent_data['High'].max()
        low_20 = recent_data['Low'].min()
        price_position = (prices[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5

        # Portfolio features
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0

        # Momentum features
        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

        # Combine all features
        state_features = []

        # Price features (20)
        if len(returns) >= 10:
            state_features.extend(returns[-10:])
        else:
            state_features.extend([0] * 10)

        if len(log_returns) >= 5:
            state_features.extend(log_returns[-5:])
        else:
            state_features.extend([0] * 5)

        state_features.extend([sma_5, sma_10, sma_20, momentum_5, momentum_10])

        # Technical indicators (15)
        state_features.extend([
            rsi / 100, volatility, macd, macd_signal, price_position,
            volume_ratio, np.log(volume_ratio + 1),
            (prices[-1] - prices.min()) / (prices.max() - prices.min()) if prices.max() != prices.min() else 0.5,
            len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.5,
            np.mean(returns) if len(returns) > 0 else 0,
            np.std(returns) if len(returns) > 1 else 0,
            np.max(returns) if len(returns) > 0 else 0,
            np.min(returns) if len(returns) > 0 else 0,
            np.percentile(returns, 75) if len(returns) > 0 else 0,
            np.percentile(returns, 25) if len(returns) > 0 else 0
        ])

        # Portfolio state (10)
        state_features.extend([
            self.position,
            portfolio_return,
            drawdown,
            self.portfolio_value / self.initial_balance,
            self.balance / self.initial_balance,
            len(self.trade_history) / 100,  # Normalized trade count
            1 if self.position > 0 else 0,   # Long position
            1 if self.position < 0 else 0,   # Short position
            abs(self.position),              # Position magnitude
            (self.current_step - self.lookback_window) / len(self.data)  # Time progress
        ])

        # Market microstructure (20)
        state_features.extend([
            recent_data['High'].iloc[-1] / prices[-1],
            recent_data['Low'].iloc[-1] / prices[-1],
            recent_data['Open'].iloc[-1] / prices[-1],
            (recent_data['High'].iloc[-1] - recent_data['Low'].iloc[-1]) / prices[-1],
            abs(recent_data['Close'].iloc[-1] - recent_data['Open'].iloc[-1]) / prices[-1]
        ])

        # Normalized recent prices
        state_features.extend([p / prices[-1] for p in prices[-10:]])

        # Normalized volumes
        if volume_ma > 0:
            state_features.extend([v / volume_ma for v in volumes[-5:]])
        else:
            state_features.extend([1] * 5)

        # Time-based features (5)
        state_features.extend([
            self.current_step / len(self.data),
            (self.current_step % 20) / 20,  # Intraday cycle
            np.sin(2 * np.pi * self.current_step / 252),  # Annual cycle
            np.cos(2 * np.pi * self.current_step / 252),
            1.0  # Bias term
        ])

        # Ensure exactly 100 features
        while len(state_features) < self.state_size:
            state_features.append(0.0)

        state_features = state_features[:self.state_size]

        # Replace any NaN values
        state_features = [float(x) if np.isfinite(x) else 0.0 for x in state_features]

        return np.array(state_features, dtype=np.float32)

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices):
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0, 0.0

        ema_12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26

        macd_series = pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean()
        macd_signal = macd_series.ewm(span=9).mean().iloc[-1]

        return macd / prices[-1], macd_signal / prices[-1]  # Normalized

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute trading action and return next state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}

        # Convert action index to position change
        target_position = self.actions[action_index]
        position_change = target_position - self.position

        # Execute trade
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']

        # Calculate transaction costs
        transaction_cost = abs(position_change) * self.transaction_cost

        # Update position
        self.position = target_position

        # Calculate portfolio value change
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return - transaction_cost

        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return)
        self.balance = self.portfolio_value * (1 - abs(self.position))

        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # Record trade
        if abs(position_change) > 0.01:  # Significant position change
            self.trade_history.append({
                'step': self.current_step,
                'action': target_position,
                'price': current_price,
                'portfolio_value': self.portfolio_value
            })

        # Move to next step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(portfolio_return, position_change, transaction_cost)

        # Check if done
        done = (self.current_step >= len(self.data) - 1) or (self.portfolio_value < self.initial_balance * 0.5)

        # Get next state
        next_state = self._get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'price_return': price_return,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost
        }

        return next_state, reward, done, info

    def _calculate_reward(self, portfolio_return: float, position_change: float, transaction_cost: float) -> float:
        """Calculate reward for the action"""
        # Base reward: portfolio return
        reward = portfolio_return * 100  # Scale up

        # Penalty for excessive trading
        trading_penalty = abs(position_change) * 0.1

        # Risk-adjusted reward
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        risk_penalty = drawdown * 2

        # Bonus for consistent performance
        if len(self.trade_history) > 10:
            recent_returns = [t.get('portfolio_return', 0) for t in self.trade_history[-10:]]
            if all(r > -0.02 for r in recent_returns):  # Consistent small losses or gains
                reward += 0.1

        total_reward = reward - trading_penalty - risk_penalty

        return total_reward

class GPUAITradingAgent:
    """GPU-accelerated AI trading agent with reinforcement learning"""

    def __init__(self, state_size=100, action_size=21, learning_rate=0.001):
        self.device = device
        self.logger = logging.getLogger('AITradingAgent')

        self.state_size = state_size
        self.action_size = action_size

        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.actor_critic = ActorCriticNetwork(state_size).to(self.device)

        # Optimizers
        self.dqn_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # RL parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.tau = 0.005   # Soft update parameter

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = 256 if self.device.type == 'cuda' else 64

        # Training tracking
        self.training_history = []
        self.performance_metrics = {
            'total_episodes': 0,
            'avg_reward': 0,
            'win_rate': 0,
            'sharpe_ratio': 0
        }

        if self.device.type == 'cuda':
            self.logger.info(f">> AI Trading Agent: {torch.cuda.get_device_name(0)}")
            self.logger.info(f">> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.info(">> CPU AI Trading Agent")

        self.logger.info(f">> Batch size: {self.batch_size}")
        self.logger.info(f">> DQN + Actor-Critic architecture ready")

        # Initialize target network
        self._soft_update(tau=1.0)

    def _soft_update(self, tau=None):
        """Soft update target network"""
        tau = tau or self.tau

        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state: np.ndarray, training=True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

    def train_dqn(self) -> float:
        """Train DQN network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.dqn_optimizer.step()

        # Soft update target network
        self._soft_update()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Train agent for one episode"""
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        steps = 0

        while True:
            # Select action
            action = self.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            self.store_experience(state, action, reward, next_state, done)

            # Train network
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.train_dqn()
                episode_loss += loss

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Update metrics
        self.performance_metrics['total_episodes'] += 1

        episode_metrics = {
            'total_reward': total_reward,
            'steps': steps,
            'final_portfolio_value': info.get('portfolio_value', 0),
            'average_loss': episode_loss / steps if steps > 0 else 0,
            'epsilon': self.epsilon
        }

        return episode_metrics

    def train_multiple_episodes(self, data: pd.DataFrame, num_episodes: int = 100) -> Dict[str, Any]:
        """Train agent for multiple episodes"""
        self.logger.info(f">> Starting AI training for {num_episodes} episodes...")
        start_time = datetime.now()

        episode_rewards = []
        episode_portfolios = []

        for episode in range(num_episodes):
            env = TradingEnvironment(data)
            metrics = self.train_episode(env)

            episode_rewards.append(metrics['total_reward'])
            episode_portfolios.append(metrics['final_portfolio_value'])

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_portfolio = np.mean(episode_portfolios[-10:])
                self.logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Avg Portfolio = ${avg_portfolio:,.0f}")

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # Calculate final metrics
        returns = [(p - 100000) / 100000 for p in episode_portfolios]
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        training_summary = {
            'training_timestamp': start_time,
            'completion_timestamp': end_time,
            'training_time_seconds': training_time,
            'episodes_completed': num_episodes,
            'gpu_accelerated': self.device.type == 'cuda',
            'final_metrics': {
                'average_reward': np.mean(episode_rewards),
                'average_portfolio_value': np.mean(episode_portfolios),
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'best_portfolio': max(episode_portfolios),
                'worst_portfolio': min(episode_portfolios)
            },
            'learning_progress': {
                'initial_avg_reward': np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else 0,
                'final_avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0,
                'improvement': np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10]) if len(episode_rewards) >= 20 else 0
            },
            'performance_metrics': {
                'episodes_per_second': num_episodes / training_time if training_time > 0 else 0,
                'gpu_memory_used_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            }
        }

        self.logger.info(f">> AI Training Complete!")
        self.logger.info(f">> Training time: {training_time:.1f}s")
        self.logger.info(f">> Performance: {training_summary['performance_metrics']['episodes_per_second']:.2f} episodes/second")
        self.logger.info(f">> Final Sharpe ratio: {sharpe_ratio:.3f}")
        self.logger.info(f">> Win rate: {win_rate:.1%}")

        return training_summary

    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'dqn_optimizer_state_dict': self.dqn_optimizer.state_dict(),
            'ac_optimizer_state_dict': self.ac_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'performance_metrics': self.performance_metrics
        }, filepath)

        self.logger.info(f">> Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.dqn_optimizer.load_state_dict(checkpoint['dqn_optimizer_state_dict'])
        self.ac_optimizer.load_state_dict(checkpoint['ac_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.performance_metrics = checkpoint['performance_metrics']

        self.logger.info(f">> Model loaded from {filepath}")

if __name__ == "__main__":
    # Initialize AI trading agent
    agent = GPUAITradingAgent()

    # Load sample data for training
    try:
        # Download sample data
        ticker = yf.Ticker('SPY')
        data = ticker.history(period='2y', interval='1d')

        if not data.empty:
            # Train the agent
            results = agent.train_multiple_episodes(data, num_episodes=50)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'ai_training_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save model
            agent.save_model(f'ai_trading_model_{timestamp}.pth')

            print(f"\n>> AI TRADING AGENT TRAINING COMPLETE!")
            print(f">> Episodes: {results['episodes_completed']}")
            print(f">> Training time: {results['training_time_seconds']:.1f}s")
            print(f">> Performance: {results['performance_metrics']['episodes_per_second']:.2f} episodes/second")
            print(f">> Final Sharpe ratio: {results['final_metrics']['sharpe_ratio']:.3f}")
            print(f">> Win rate: {results['final_metrics']['win_rate']:.1%}")
            print(f">> Best portfolio: ${results['final_metrics']['best_portfolio']:,.0f}")

            if torch.cuda.is_available():
                print(f">> GPU memory used: {results['performance_metrics']['gpu_memory_used_gb']:.2f} GB")

            print(f">> AI Agent ready for live trading! 🤖")

        else:
            print(">> Failed to download training data")

    except Exception as e:
        print(f">> Error in AI training: {e}")
        print(">> Check internet connection and try again")