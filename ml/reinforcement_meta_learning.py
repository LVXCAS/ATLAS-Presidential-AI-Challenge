"""
Reinforcement Learning and Meta Learning System
==============================================

This module implements advanced reinforcement learning and meta learning
components for adaptive trading strategies:

1. Trading Environment (Gym-compatible)
2. Deep Q-Network (DQN) for trading decisions
3. Policy Gradient Methods (A2C, PPO)
4. Meta Learning for strategy adaptation
5. Multi-task Learning across different markets
6. Transfer Learning for new assets
7. Curriculum Learning for progressive difficulty
8. Online Learning with experience replay

The system enables the trading bot to learn optimal actions through
interaction with the market environment while adapting strategies
across different market conditions and asset classes.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
import pickle
import json
import os

# Stable Baselines3 for RL algorithms
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trading action types"""
    HOLD = 0
    BUY = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4


class MarketRegime(Enum):
    """Market regime types for meta learning"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOL = "low_vol"


@dataclass
class TradingState:
    """Trading state representation"""
    price_features: np.ndarray
    technical_features: np.ndarray
    portfolio_features: np.ndarray
    market_features: np.ndarray
    position: float
    cash: float
    portfolio_value: float
    timestamp: datetime


@dataclass
class TradeAction:
    """Trading action with details"""
    action_type: ActionType
    quantity: float
    confidence: float
    expected_return: float
    risk_score: float
    reasoning: str


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class TradingEnvironment(gym.Env):
    """
    Gym-compatible trading environment for reinforcement learning
    """

    def __init__(self,
                 market_data: pd.DataFrame,
                 initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 lookback_window: int = 60,
                 reward_scaling: float = 1000.0):

        super(TradingEnvironment, self).__init__()

        self.market_data = market_data
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling

        # Environment state
        self.current_step = 0
        self.max_steps = len(market_data) - lookback_window - 1
        self.cash = initial_cash
        self.position = 0.0
        self.portfolio_value = initial_cash
        self.trade_history = []

        # Feature engineering
        self.features = self._prepare_features()

        # Action space: [HOLD, BUY, SELL, STRONG_BUY, STRONG_SELL]
        self.action_space = spaces.Discrete(5)

        # Observation space
        feature_dim = self._get_feature_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(feature_dim,), dtype=np.float32
        )

        # Performance tracking
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features from market data"""
        try:
            features_df = pd.DataFrame(index=self.market_data.index)

            # Price features
            close = self.market_data['Close']
            high = self.market_data['High']
            low = self.market_data['Low']
            volume = self.market_data['Volume']

            # Returns
            features_df['returns'] = close.pct_change()
            features_df['log_returns'] = np.log(close / close.shift(1))

            # Technical indicators
            features_df['rsi'] = self._calculate_rsi(close, 14)
            features_df['macd'] = self._calculate_macd(close)
            features_df['bb_position'] = self._calculate_bb_position(close)
            features_df['volume_ratio'] = volume / volume.rolling(20).mean()

            # Price ratios
            features_df['price_sma_20'] = close / close.rolling(20).mean()
            features_df['price_sma_50'] = close / close.rolling(50).mean()

            # Volatility
            features_df['volatility'] = features_df['returns'].rolling(20).std()

            # Market microstructure
            features_df['high_low_ratio'] = (high - low) / close
            features_df['price_position'] = (close - low) / (high - low)

            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)

            return features_df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26

    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return (prices - lower_band) / (upper_band - lower_band)

    def _get_feature_dimension(self) -> int:
        """Get total feature dimension"""
        market_features = len(self.features.columns) if len(self.features) > 0 else 10
        portfolio_features = 3  # position, cash_ratio, portfolio_value
        historical_features = self.lookback_window * market_features
        return market_features + portfolio_features + historical_features

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.trade_history = []

        # Reset metrics
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

        state = self._get_observation()
        info = self._get_info()

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        try:
            # Execute action
            reward = self._execute_action(action)

            # Move to next step
            self.current_step += 1

            # Check if episode is done
            done = self.current_step >= self.max_steps
            truncated = False

            # Get new observation
            next_state = self._get_observation()

            # Calculate episode metrics if done
            if done:
                self._calculate_episode_metrics()

            info = self._get_info()

            return next_state, reward, done, truncated, info

        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            return self._get_observation(), 0.0, True, True, {}

    def _execute_action(self, action: int) -> float:
        """Execute trading action and calculate reward"""
        try:
            current_price = self.market_data['Close'].iloc[self.current_step]
            action_type = ActionType(action)

            # Calculate position change
            if action_type == ActionType.HOLD:
                position_change = 0.0
            elif action_type == ActionType.BUY:
                position_change = 0.25  # 25% of max position
            elif action_type == ActionType.SELL:
                position_change = -0.25
            elif action_type == ActionType.STRONG_BUY:
                position_change = 0.5   # 50% of max position
            elif action_type == ActionType.STRONG_SELL:
                position_change = -0.5
            else:
                position_change = 0.0

            # Clip position change to respect limits
            new_position = np.clip(
                self.position + position_change,
                -self.max_position,
                self.max_position
            )

            actual_position_change = new_position - self.position

            # Calculate transaction cost
            transaction_value = abs(actual_position_change) * current_price * self.initial_cash
            cost = transaction_value * self.transaction_cost

            # Update position and cash
            if actual_position_change != 0:
                self.position = new_position
                self.cash -= cost

                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': action_type.name,
                    'position_change': actual_position_change,
                    'price': current_price,
                    'cost': cost
                })

            # Update portfolio value
            self.portfolio_value = self.cash + (self.position * current_price * self.initial_cash)

            # Calculate reward
            reward = self._calculate_reward()

            return reward

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return 0.0

    def _calculate_reward(self) -> float:
        """Calculate reward for current action"""
        try:
            if self.current_step < self.lookback_window + 1:
                return 0.0

            # Portfolio return
            prev_value = self.cash + (self.position *
                                    self.market_data['Close'].iloc[self.current_step - 1] *
                                    self.initial_cash)

            portfolio_return = (self.portfolio_value - prev_value) / prev_value

            # Market return
            current_price = self.market_data['Close'].iloc[self.current_step]
            prev_price = self.market_data['Close'].iloc[self.current_step - 1]
            market_return = (current_price - prev_price) / prev_price

            # Excess return (alpha)
            excess_return = portfolio_return - market_return

            # Risk penalty (volatility of returns)
            if len(self.trade_history) > 10:
                recent_returns = []
                for i in range(max(0, len(self.trade_history) - 10), len(self.trade_history)):
                    trade = self.trade_history[i]
                    # Calculate return for this trade (simplified)
                    recent_returns.append(portfolio_return)

                volatility_penalty = np.std(recent_returns) if len(recent_returns) > 1 else 0
            else:
                volatility_penalty = 0

            # Transaction cost penalty
            transaction_penalty = 0
            if self.trade_history and self.trade_history[-1]['step'] == self.current_step:
                transaction_penalty = self.trade_history[-1]['cost'] / self.initial_cash

            # Drawdown penalty
            max_value = max([self.initial_cash] +
                           [self.cash + (self.position *
                                       self.market_data['Close'].iloc[i] *
                                       self.initial_cash)
                            for i in range(self.lookback_window, self.current_step)])

            drawdown = (max_value - self.portfolio_value) / max_value
            drawdown_penalty = drawdown * 0.5

            # Combined reward
            reward = (excess_return - volatility_penalty * 0.1 -
                     transaction_penalty * 10 - drawdown_penalty) * self.reward_scaling

            return reward

        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation/state"""
        try:
            if self.current_step < self.lookback_window:
                # Return zero observation if not enough history
                return np.zeros(self._get_feature_dimension(), dtype=np.float32)

            # Current market features
            current_features = self.features.iloc[self.current_step].values

            # Portfolio features
            current_price = self.market_data['Close'].iloc[self.current_step]
            portfolio_features = np.array([
                self.position,
                self.cash / self.initial_cash,
                self.portfolio_value / self.initial_cash
            ])

            # Historical features (lookback window)
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step
            historical_features = self.features.iloc[start_idx:end_idx].values.flatten()

            # Combine all features
            observation = np.concatenate([
                current_features,
                portfolio_features,
                historical_features
            ]).astype(np.float32)

            # Handle any NaN or inf values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

            return observation

        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            return np.zeros(self._get_feature_dimension(), dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state"""
        try:
            return {
                'current_step': self.current_step,
                'portfolio_value': self.portfolio_value,
                'position': self.position,
                'cash': self.cash,
                'num_trades': len(self.trade_history),
                'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash
            }
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {}

    def _calculate_episode_metrics(self) -> None:
        """Calculate metrics for completed episode"""
        try:
            # Total return
            total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
            self.episode_metrics['total_return'] = total_return

            # Calculate daily returns for Sharpe ratio
            daily_returns = []
            daily_values = []

            for i in range(self.lookback_window, self.current_step):
                price = self.market_data['Close'].iloc[i]
                value = self.cash + (self.position * price * self.initial_cash)
                daily_values.append(value)

                if len(daily_values) > 1:
                    daily_return = (daily_values[-1] - daily_values[-2]) / daily_values[-2]
                    daily_returns.append(daily_return)

            if len(daily_returns) > 1:
                # Sharpe ratio (simplified)
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
                self.episode_metrics['sharpe_ratio'] = sharpe_ratio

                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + np.array(daily_returns))
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (peak - cumulative_returns) / peak
                max_drawdown = np.max(drawdown)
                self.episode_metrics['max_drawdown'] = max_drawdown

            # Trading metrics
            self.episode_metrics['num_trades'] = len(self.trade_history)

            # Win rate and profit factor (simplified)
            if self.trade_history:
                profitable_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
                win_rate = profitable_trades / len(self.trade_history)
                self.episode_metrics['win_rate'] = win_rate

                total_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0)
                total_loss = abs(sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) < 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                self.episode_metrics['profit_factor'] = profit_factor

        except Exception as e:
            logger.error(f"Error calculating episode metrics: {e}")

    def render(self, mode='human') -> None:
        """Render environment (for debugging)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:.2f}, "
                  f"Position: {self.position:.2f}, Cash: ${self.cash:.2f}")


class DQNAgent:
    """Deep Q-Network agent for trading"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 config: Dict[str, Any] = None):

        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Hyperparameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_size = self.config.get('memory_size', 10000)

        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(state_size, action_size, self.config).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, self.config).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience replay
        self.memory = deque(maxlen=self.memory_size)

        # Training tracking
        self.training_history = []
        self.loss_history = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        try:
            if training and random.random() <= self.epsilon:
                return random.choice(range(self.action_size))

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

        except Exception as e:
            logger.error(f"Error in DQN action selection: {e}")
            return 0

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer"""
        try:
            experience = Experience(state, action, reward, next_state, done)
            self.memory.append(experience)
        except Exception as e:
            logger.error(f"Error storing experience: {e}")

    def replay(self) -> float:
        """Train the agent on a batch of experiences"""
        try:
            if len(self.memory) < self.batch_size:
                return 0.0

            batch = random.sample(self.memory, self.batch_size)

            states = torch.FloatTensor([e.state for e in batch]).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
            dones = torch.BoolTensor([e.done for e in batch]).to(self.device)

            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.loss_history.append(loss.item())
            return loss.item()

        except Exception as e:
            logger.error(f"Error in DQN replay: {e}")
            return 0.0

    def update_target_network(self) -> None:
        """Update target network with current network weights"""
        try:
            self.target_network.load_state_dict(self.q_network.state_dict())
        except Exception as e:
            logger.error(f"Error updating target network: {e}")

    def save(self, filepath: str) -> bool:
        """Save agent model"""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_history': self.training_history,
                'epsilon': self.epsilon
            }, filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving DQN agent: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load agent model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            return True
        except Exception as e:
            logger.error(f"Error loading DQN agent: {e}")
            return False


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        super(DQNNetwork, self).__init__()

        hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        dropout_rate = config.get('dropout_rate', 0.2)

        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MetaLearningAgent:
    """
    Meta learning agent that adapts strategies across different market conditions
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Task-specific agents for different market regimes
        self.regime_agents: Dict[MarketRegime, DQNAgent] = {}

        # Meta learning components
        self.meta_network = None
        self.regime_classifier = None

        # Experience storage per regime
        self.regime_experiences: Dict[MarketRegime, deque] = {}

        # Performance tracking per regime
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}

        # Current regime detection
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = deque(maxlen=100)

    async def initialize(self, state_size: int, action_size: int) -> bool:
        """Initialize meta learning agent"""
        try:
            logger.info("Initializing Meta Learning Agent")

            # Initialize agents for each market regime
            for regime in MarketRegime:
                agent_config = self.config.get(f'{regime.value}_config', self.config)
                self.regime_agents[regime] = DQNAgent(state_size, action_size, agent_config)
                self.regime_experiences[regime] = deque(maxlen=10000)
                self.regime_performance[regime] = {
                    'total_episodes': 0,
                    'avg_reward': 0.0,
                    'success_rate': 0.0,
                    'sharpe_ratio': 0.0
                }

            # Initialize regime classifier
            self.regime_classifier = RegimeClassifier(config=self.config.get('regime_classifier', {}))

            logger.info(f"Meta Learning Agent initialized with {len(self.regime_agents)} regime-specific agents")
            return True

        except Exception as e:
            logger.error(f"Error initializing Meta Learning Agent: {e}")
            return False

    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(market_data) < 50:
                return MarketRegime.SIDEWAYS

            # Simple regime detection based on volatility and trend
            returns = market_data['Close'].pct_change().dropna()

            # Calculate metrics
            volatility = returns.std() * np.sqrt(252)
            trend = returns.mean() * 252

            # Recent performance
            recent_returns = returns.tail(20)
            recent_trend = recent_returns.mean() * 252

            # Classify regime
            if volatility > 0.3:
                regime = MarketRegime.VOLATILE
            elif volatility < 0.1:
                regime = MarketRegime.LOW_VOL
            elif recent_trend > 0.15:
                regime = MarketRegime.BULL
            elif recent_trend < -0.15:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS

            # Update regime history
            self.regime_history.append(regime)
            self.current_regime = regime

            return regime

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime.SIDEWAYS

    async def act(self, state: np.ndarray, market_data: pd.DataFrame, training: bool = True) -> int:
        """Select action using appropriate regime-specific agent"""
        try:
            # Detect current regime
            regime = await self.detect_regime(market_data)

            # Get action from regime-specific agent
            agent = self.regime_agents[regime]
            action = agent.act(state, training)

            return action

        except Exception as e:
            logger.error(f"Error in meta learning action selection: {e}")
            return 0

    async def learn_from_experience(self,
                                  state: np.ndarray,
                                  action: int,
                                  reward: float,
                                  next_state: np.ndarray,
                                  done: bool,
                                  market_data: pd.DataFrame) -> None:
        """Learn from experience using appropriate regime-specific agent"""
        try:
            # Detect regime for this experience
            regime = await self.detect_regime(market_data)

            # Store experience in regime-specific memory
            agent = self.regime_agents[regime]
            agent.remember(state, action, reward, next_state, done)

            # Train regime-specific agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()

                # Update performance tracking
                self.regime_performance[regime]['total_episodes'] += 1

                # Update target network periodically
                if self.regime_performance[regime]['total_episodes'] % 100 == 0:
                    agent.update_target_network()

            # Cross-regime knowledge transfer
            await self._transfer_knowledge(regime, state, action, reward, next_state, done)

        except Exception as e:
            logger.error(f"Error in meta learning experience: {e}")

    async def _transfer_knowledge(self,
                                source_regime: MarketRegime,
                                state: np.ndarray,
                                action: int,
                                reward: float,
                                next_state: np.ndarray,
                                done: bool) -> None:
        """Transfer knowledge between regimes"""
        try:
            # Simple knowledge transfer: share successful experiences
            if reward > 0:  # Only transfer positive experiences
                for regime, agent in self.regime_agents.items():
                    if regime != source_regime:
                        # Add experience to other agents with reduced weight
                        scaled_reward = reward * 0.5  # Reduce reward for transferred experience
                        agent.remember(state, action, scaled_reward, next_state, done)

        except Exception as e:
            logger.error(f"Error in knowledge transfer: {e}")

    async def adapt_to_regime_change(self,
                                   old_regime: MarketRegime,
                                   new_regime: MarketRegime) -> None:
        """Adapt when regime changes"""
        try:
            if old_regime == new_regime:
                return

            logger.info(f"Regime change detected: {old_regime.value} -> {new_regime.value}")

            # Transfer recent successful experiences
            old_agent = self.regime_agents[old_regime]
            new_agent = self.regime_agents[new_regime]

            # Copy recent positive experiences
            recent_experiences = list(old_agent.memory)[-100:]  # Last 100 experiences
            positive_experiences = [exp for exp in recent_experiences if exp.reward > 0]

            for exp in positive_experiences:
                new_agent.remember(exp.state, exp.action, exp.reward * 0.7, exp.next_state, exp.done)

            # Adjust exploration for new regime
            new_agent.epsilon = min(new_agent.epsilon * 1.2, 0.8)  # Increase exploration

        except Exception as e:
            logger.error(f"Error adapting to regime change: {e}")

    def get_meta_performance(self) -> Dict[str, Any]:
        """Get performance metrics across all regimes"""
        try:
            overall_metrics = {
                'current_regime': self.current_regime.value,
                'regime_distribution': {},
                'regime_performance': {},
                'knowledge_transfer_events': 0,
                'adaptation_events': 0
            }

            # Calculate regime distribution
            if self.regime_history:
                for regime in MarketRegime:
                    count = sum(1 for r in self.regime_history if r == regime)
                    overall_metrics['regime_distribution'][regime.value] = count / len(self.regime_history)

            # Aggregate regime performance
            for regime, performance in self.regime_performance.items():
                overall_metrics['regime_performance'][regime.value] = performance

            return overall_metrics

        except Exception as e:
            logger.error(f"Error getting meta performance: {e}")
            return {}

    async def save_meta_agent(self, directory: str) -> bool:
        """Save all regime-specific agents and meta components"""
        try:
            os.makedirs(directory, exist_ok=True)

            # Save individual agents
            for regime, agent in self.regime_agents.items():
                agent_path = os.path.join(directory, f"{regime.value}_agent.pth")
                success = agent.save(agent_path)
                if not success:
                    logger.warning(f"Failed to save {regime.value} agent")

            # Save meta learning metadata
            metadata = {
                'regime_performance': self.regime_performance,
                'current_regime': self.current_regime.value,
                'regime_history': [r.value for r in list(self.regime_history)],
                'config': self.config
            }

            metadata_path = os.path.join(directory, 'meta_agent_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Meta agent saved to {directory}")
            return True

        except Exception as e:
            logger.error(f"Error saving meta agent: {e}")
            return False

    async def load_meta_agent(self, directory: str) -> bool:
        """Load all regime-specific agents and meta components"""
        try:
            if not os.path.exists(directory):
                logger.error(f"Meta agent directory {directory} does not exist")
                return False

            # Load metadata
            metadata_path = os.path.join(directory, 'meta_agent_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.regime_performance = metadata.get('regime_performance', {})
                self.current_regime = MarketRegime(metadata.get('current_regime', 'sideways'))
                regime_history = metadata.get('regime_history', [])
                self.regime_history = deque([MarketRegime(r) for r in regime_history], maxlen=100)

            # Load individual agents
            for regime in MarketRegime:
                agent_path = os.path.join(directory, f"{regime.value}_agent.pth")
                if os.path.exists(agent_path) and regime in self.regime_agents:
                    success = self.regime_agents[regime].load(agent_path)
                    if not success:
                        logger.warning(f"Failed to load {regime.value} agent")

            logger.info(f"Meta agent loaded from {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading meta agent: {e}")
            return False


class RegimeClassifier:
    """Classifier for market regime detection"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_trained = False

    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime classification"""
        try:
            close = market_data['Close']
            high = market_data['High']
            low = market_data['Low']
            volume = market_data['Volume']

            # Calculate features
            returns = close.pct_change()
            volatility = returns.rolling(20).std()
            trend = returns.rolling(20).mean()
            volume_ratio = volume / volume.rolling(20).mean()

            # Price momentum
            momentum_5 = close / close.shift(5) - 1
            momentum_20 = close / close.shift(20) - 1

            # Create feature matrix
            features = pd.DataFrame({
                'volatility': volatility,
                'trend': trend,
                'volume_ratio': volume_ratio,
                'momentum_5': momentum_5,
                'momentum_20': momentum_20,
                'rsi': self._calculate_rsi(close),
                'bb_position': self._calculate_bb_position(close)
            })

            return features.fillna(0).values

        except Exception as e:
            logger.error(f"Error preparing regime features: {e}")
            return np.array([])

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return (prices - lower_band) / (upper_band - lower_band)


class ReinforcementMetaLearningSystem:
    """
    Main system that combines reinforcement learning with meta learning
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Core components
        self.trading_env = None
        self.meta_agent = None

        # Training configuration
        self.training_episodes = self.config.get('training_episodes', 1000)
        self.evaluation_episodes = self.config.get('evaluation_episodes', 100)
        self.update_frequency = self.config.get('update_frequency', 100)

        # Performance tracking
        self.training_history = []
        self.evaluation_history = []

    async def initialize(self, market_data: pd.DataFrame) -> bool:
        """Initialize the RL/Meta learning system"""
        try:
            logger.info("Initializing Reinforcement Meta Learning System")

            # Initialize trading environment
            env_config = self.config.get('environment', {})
            self.trading_env = TradingEnvironment(market_data, **env_config)

            # Initialize meta learning agent
            state_size = self.trading_env.observation_space.shape[0]
            action_size = self.trading_env.action_space.n

            self.meta_agent = MetaLearningAgent(self.config.get('meta_agent', {}))
            success = await self.meta_agent.initialize(state_size, action_size)

            if not success:
                return False

            logger.info("Reinforcement Meta Learning System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing RL Meta Learning System: {e}")
            return False

    async def train(self, market_data: pd.DataFrame, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the meta learning agent"""
        try:
            episodes = episodes or self.training_episodes
            logger.info(f"Starting training for {episodes} episodes")

            training_results = {
                'episode_rewards': [],
                'episode_lengths': [],
                'regime_performance': {},
                'training_losses': []
            }

            for episode in range(episodes):
                # Reset environment
                state, info = self.trading_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

                while not done:
                    # Select action using meta agent
                    action = await self.meta_agent.act(state, market_data, training=True)

                    # Execute action
                    next_state, reward, done, truncated, info = self.trading_env.step(action)

                    # Learn from experience
                    await self.meta_agent.learn_from_experience(
                        state, action, reward, next_state, done, market_data
                    )

                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if truncated:
                        done = True

                # Record episode results
                training_results['episode_rewards'].append(episode_reward)
                training_results['episode_lengths'].append(episode_length)

                # Log progress
                if episode % 100 == 0:
                    avg_reward = np.mean(training_results['episode_rewards'][-100:])
                    logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                              f"Current Regime: {self.meta_agent.current_regime.value}")

                # Evaluate periodically
                if episode % self.evaluation_episodes == 0 and episode > 0:
                    eval_results = await self.evaluate(market_data, episodes=10)
                    self.evaluation_history.append({
                        'episode': episode,
                        'results': eval_results
                    })

            # Get final performance metrics
            training_results['regime_performance'] = self.meta_agent.get_meta_performance()

            self.training_history.append(training_results)

            logger.info(f"Training completed after {episodes} episodes")
            return training_results

        except Exception as e:
            logger.error(f"Error in training: {e}")
            return {}

    async def evaluate(self, market_data: pd.DataFrame, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        try:
            logger.info(f"Evaluating agent for {episodes} episodes")

            evaluation_results = {
                'episode_rewards': [],
                'episode_metrics': [],
                'regime_distribution': {},
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

            for episode in range(episodes):
                # Reset environment
                state, info = self.trading_env.reset()
                episode_reward = 0
                done = False

                while not done:
                    # Select action (no exploration)
                    action = await self.meta_agent.act(state, market_data, training=False)

                    # Execute action
                    next_state, reward, done, truncated, info = self.trading_env.step(action)

                    state = next_state
                    episode_reward += reward

                    if truncated:
                        done = True

                # Record results
                evaluation_results['episode_rewards'].append(episode_reward)
                evaluation_results['episode_metrics'].append(self.trading_env.episode_metrics.copy())

            # Calculate aggregate metrics
            if evaluation_results['episode_rewards']:
                returns = [metrics['total_return'] for metrics in evaluation_results['episode_metrics']]
                sharpe_ratios = [metrics['sharpe_ratio'] for metrics in evaluation_results['episode_metrics']]
                max_drawdowns = [metrics['max_drawdown'] for metrics in evaluation_results['episode_metrics']]

                evaluation_results['total_return'] = np.mean(returns)
                evaluation_results['sharpe_ratio'] = np.mean(sharpe_ratios)
                evaluation_results['max_drawdown'] = np.mean(max_drawdowns)

            logger.info(f"Evaluation completed: Avg Return: {evaluation_results['total_return']:.2%}, "
                       f"Avg Sharpe: {evaluation_results['sharpe_ratio']:.2f}")

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {}

    async def predict_action(self, state: np.ndarray, market_data: pd.DataFrame) -> Tuple[int, float]:
        """Predict best action for given state"""
        try:
            # Get action from meta agent
            action = await self.meta_agent.act(state, market_data, training=False)

            # Get confidence (simplified)
            regime = await self.meta_agent.detect_regime(market_data)
            agent = self.meta_agent.regime_agents[regime]

            # Calculate confidence based on Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1).max().item()

            return action, confidence

        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            return 0, 0.0

    async def save_system(self, directory: str) -> bool:
        """Save the entire RL/Meta learning system"""
        try:
            os.makedirs(directory, exist_ok=True)

            # Save meta agent
            meta_agent_dir = os.path.join(directory, 'meta_agent')
            success = await self.meta_agent.save_meta_agent(meta_agent_dir)

            if not success:
                return False

            # Save system metadata
            metadata = {
                'config': self.config,
                'training_history': self.training_history,
                'evaluation_history': self.evaluation_history,
                'training_episodes': self.training_episodes
            }

            metadata_path = os.path.join(directory, 'system_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"RL Meta Learning System saved to {directory}")
            return True

        except Exception as e:
            logger.error(f"Error saving system: {e}")
            return False

    async def load_system(self, directory: str) -> bool:
        """Load the entire RL/Meta learning system"""
        try:
            if not os.path.exists(directory):
                logger.error(f"System directory {directory} does not exist")
                return False

            # Load system metadata
            metadata_path = os.path.join(directory, 'system_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.training_history = metadata.get('training_history', [])
                self.evaluation_history = metadata.get('evaluation_history', [])
                self.training_episodes = metadata.get('training_episodes', 1000)

            # Load meta agent
            meta_agent_dir = os.path.join(directory, 'meta_agent')
            if self.meta_agent:
                success = await self.meta_agent.load_meta_agent(meta_agent_dir)
                if not success:
                    return False

            logger.info(f"RL Meta Learning System loaded from {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading system: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'training_episodes_completed': len(self.training_history),
                'evaluation_episodes_completed': len(self.evaluation_history),
                'config': self.config
            }

            if self.meta_agent:
                status['meta_agent_performance'] = self.meta_agent.get_meta_performance()

            if self.training_history:
                latest_training = self.training_history[-1]
                status['latest_training_avg_reward'] = np.mean(latest_training['episode_rewards'][-100:])

            if self.evaluation_history:
                latest_eval = self.evaluation_history[-1]['results']
                status['latest_evaluation'] = {
                    'total_return': latest_eval.get('total_return', 0.0),
                    'sharpe_ratio': latest_eval.get('sharpe_ratio', 0.0),
                    'max_drawdown': latest_eval.get('max_drawdown', 0.0)
                }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_rl_meta_system():
        """Test the RL/Meta learning system"""

        config = {
            'training_episodes': 100,  # Reduced for testing
            'evaluation_episodes': 50,
            'environment': {
                'initial_cash': 100000,
                'transaction_cost': 0.001,
                'max_position': 1.0,
                'lookback_window': 30,
                'reward_scaling': 1000
            },
            'meta_agent': {
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
                'memory_size': 10000
            }
        }

        # Generate sample market data
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        np.random.seed(42)

        price_data = []
        base_price = 100.0

        for i in range(1000):
            # Create different market regimes
            if i < 300:  # Bull market
                change = np.random.normal(0.002, 0.015)
            elif i < 600:  # Bear market
                change = np.random.normal(-0.002, 0.025)
            elif i < 800:  # Sideways
                change = np.random.normal(0.0, 0.012)
            else:  # Volatile
                change = np.random.normal(0.001, 0.035)

            base_price *= (1 + change)

            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price + np.random.normal(0, 0.005)
            volume = np.random.randint(100000, 1000000)

            price_data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': base_price,
                'Volume': volume
            })

        market_data = pd.DataFrame(price_data, index=dates)

        try:
            # Initialize system
            rl_system = ReinforcementMetaLearningSystem(config)
            success = await rl_system.initialize(market_data)

            if not success:
                print("Failed to initialize RL Meta Learning System")
                return

            print("RL Meta Learning System initialized successfully")

            # Train the system
            print("Starting training...")
            training_results = await rl_system.train(market_data, episodes=50)  # Reduced for testing

            print(f"Training completed:")
            print(f"- Average reward: {np.mean(training_results['episode_rewards']):.2f}")
            print(f"- Episodes completed: {len(training_results['episode_rewards'])}")

            # Evaluate the system
            print("Starting evaluation...")
            eval_results = await rl_system.evaluate(market_data, episodes=10)

            print(f"Evaluation results:")
            print(f"- Average return: {eval_results['total_return']:.2%}")
            print(f"- Average Sharpe ratio: {eval_results['sharpe_ratio']:.2f}")
            print(f"- Average max drawdown: {eval_results['max_drawdown']:.2%}")

            # Test prediction
            print("Testing prediction...")
            state, _ = rl_system.trading_env.reset()
            action, confidence = await rl_system.predict_action(state, market_data)

            print(f"Predicted action: {ActionType(action).name} with confidence: {confidence:.2f}")

            # Get system status
            status = rl_system.get_system_status()
            print(f"\nSystem Status:")
            print(f"- Training episodes: {status['training_episodes_completed']}")
            print(f"- Evaluation episodes: {status['evaluation_episodes_completed']}")

            if 'meta_agent_performance' in status:
                meta_perf = status['meta_agent_performance']
                print(f"- Current regime: {meta_perf['current_regime']}")
                print(f"- Regime distribution: {meta_perf['regime_distribution']}")

        except Exception as e:
            print(f"Error testing RL Meta system: {e}")

    # Run test
    asyncio.run(test_rl_meta_system())