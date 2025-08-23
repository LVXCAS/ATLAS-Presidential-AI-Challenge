#!/usr/bin/env python3
"""
HIVE TRADE - Stable RL Training Environment
Improved numerical stability for reinforcement learning
"""

import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
from datetime import datetime
from collections import deque
import random

class StableRLEnvironment:
    """Numerically stable RL trading environment"""
    
    def __init__(self, initial_balance=100000):
        print("=" * 60)
        print("HIVE TRADE - STABLE RL TRAINING")  
        print("Learning Optimal Trading Strategies")
        print("=" * 60)
        
        self.initial_balance = initial_balance
        self.reset_environment()
        
        self.symbols = ['BTC-USD', 'ETH-USD']
        self.episode_length = 50  # Shorter episodes
        self.current_prices = {}
        
        print(f">> Stable RL Environment Initialized")
        print(f"   Virtual Balance: ${self.balance:,.2f}")
        print(f"   Episode Length: {self.episode_length} steps")

    def reset_environment(self):
        """Reset environment state"""
        self.balance = self.initial_balance
        self.positions = {}
        self.step_count = 0
        self.trade_history = []
        self.performance_history = []

    def get_market_data_safe(self, symbol):
        """Safely get market data with error handling"""
        try:
            # Use cached prices to avoid API rate limits
            if symbol in self.current_prices:
                base_price = self.current_prices[symbol]
            else:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if len(hist) > 0:
                    base_price = hist['Close'].iloc[-1]
                    self.current_prices[symbol] = base_price
                else:
                    # Fallback prices
                    base_price = 60000 if 'BTC' in symbol else 3000
            
            # Add realistic price movement
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            current_price = base_price * (1 + price_change)
            
            return {
                'price': current_price,
                'change': price_change,
                'volume': np.random.uniform(1e6, 5e6)
            }
            
        except Exception as e:
            print(f"   Market data error: {e}")
            # Return default data
            base_price = 60000 if 'BTC' in symbol else 3000
            return {
                'price': base_price,
                'change': 0.0,
                'volume': 2e6
            }

    def get_normalized_state(self):
        """Get normalized state vector"""
        state = []
        
        portfolio_value = self.calculate_portfolio_value()
        
        for symbol in self.symbols:
            data = self.get_market_data_safe(symbol)
            
            # Normalized features (all between -1 and 1)
            state.extend([
                np.clip(data['price'] / 50000 - 1, -1, 1),  # Normalized price
                np.clip(data['change'] * 10, -1, 1),  # Price change
                np.clip(data['volume'] / 1e7 - 0.5, -1, 1),  # Volume
            ])
            
            # Position features
            if symbol in self.positions:
                pos = self.positions[symbol]
                unrealized_pnl = (data['price'] - pos['entry_price']) * pos['qty']
                state.extend([
                    np.clip(pos['qty'] * data['price'] / 1000 - 0.5, -1, 1),  # Position size
                    np.clip(unrealized_pnl / 1000, -1, 1),  # Normalized P&L
                ])
            else:
                state.extend([0.0, 0.0])  # No position
        
        # Portfolio features
        state.extend([
            np.clip(self.balance / self.initial_balance - 1, -1, 1),  # Cash ratio
            np.clip(portfolio_value / self.initial_balance - 1, -1, 1),  # Total value ratio
        ])
        
        return np.array(state, dtype=np.float32)

    def calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = self.balance
        
        for symbol in self.positions:
            data = self.get_market_data_safe(symbol)
            pos = self.positions[symbol]
            market_value = data['price'] * pos['qty']
            total_value += market_value
        
        return total_value

    def execute_trade_action(self, symbol, action):
        """Execute trading action (0=HOLD, 1=BUY, 2=SELL)"""
        data = self.get_market_data_safe(symbol)
        current_price = data['price']
        trade_amount = 100  # Fixed trade size
        
        if action == 1:  # BUY
            if self.balance >= trade_amount:
                qty = trade_amount / current_price
                self.balance -= trade_amount
                
                if symbol in self.positions:
                    # Add to position
                    old_pos = self.positions[symbol]
                    total_qty = old_pos['qty'] + qty
                    total_cost = (old_pos['entry_price'] * old_pos['qty']) + trade_amount
                    avg_price = total_cost / total_qty
                    
                    self.positions[symbol] = {
                        'qty': total_qty,
                        'entry_price': avg_price
                    }
                else:
                    self.positions[symbol] = {
                        'qty': qty,
                        'entry_price': current_price
                    }
                
                self.trade_history.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'amount': trade_amount
                })
                
                return 0.01  # Small positive reward for taking action
            else:
                return -0.01  # Penalty for invalid action
        
        elif action == 2:  # SELL
            if symbol in self.positions and self.positions[symbol]['qty'] > 0:
                pos = self.positions[symbol]
                sell_qty = min(pos['qty'], trade_amount / current_price)
                
                if sell_qty > 0:
                    proceeds = sell_qty * current_price
                    self.balance += proceeds
                    
                    # Calculate P&L
                    pnl = (current_price - pos['entry_price']) * sell_qty
                    
                    # Update position
                    remaining_qty = pos['qty'] - sell_qty
                    if remaining_qty < 0.001:
                        del self.positions[symbol]
                    else:
                        self.positions[symbol]['qty'] = remaining_qty
                    
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'amount': proceeds,
                        'pnl': pnl
                    })
                    
                    # Reward based on P&L (clipped)
                    reward = np.clip(pnl / trade_amount, -0.1, 0.1)
                    return reward
                else:
                    return -0.01
            else:
                return -0.01  # No position to sell
        
        else:  # HOLD
            return -0.001  # Tiny penalty for inaction

    def step(self, actions):
        """Execute one environment step"""
        total_reward = 0
        
        # Execute actions for each symbol
        for i, symbol in enumerate(self.symbols):
            if i < len(actions):
                reward = self.execute_trade_action(symbol, actions[i])
                total_reward += reward
        
        self.step_count += 1
        
        # Portfolio performance reward
        portfolio_value = self.calculate_portfolio_value()
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Add small portfolio performance reward
        total_reward += np.clip(portfolio_return * 0.1, -0.05, 0.05)
        
        # Record performance
        self.performance_history.append({
            'step': self.step_count,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'return': portfolio_return
        })
        
        # Check if episode is done
        done = self.step_count >= self.episode_length
        
        return self.get_normalized_state(), total_reward, done, {
            'portfolio_value': portfolio_value,
            'return': portfolio_return,
            'trades': len(self.trade_history)
        }

    def reset(self):
        """Reset for new episode"""
        final_value = self.calculate_portfolio_value()
        final_return = (final_value - self.initial_balance) / self.initial_balance
        
        print(f"\\n>> EPISODE COMPLETE:")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Return: {final_return:+.2%}")
        print(f"   Trades: {len(self.trade_history)}")
        
        self.reset_environment()
        return self.get_normalized_state()

class StableRLAgent:
    """Numerically stable RL agent"""
    
    def __init__(self, state_size, learning_rate=0.01, epsilon=0.2):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        
        # Simple Q-network with clipped weights
        self.q_weights = np.random.randn(state_size, 3) * 0.01  # Small initial weights
        self.episode_rewards = deque(maxlen=100)
        
        print(f">> Stable RL Agent Initialized")
        print(f"   State Size: {state_size}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Initial Epsilon: {epsilon}")

    def get_action(self, state):
        """Get action using epsilon-greedy with numerical stability"""
        if np.random.random() < self.epsilon:
            # Random exploration
            return [np.random.randint(0, 3) for _ in range(2)]
        else:
            # Greedy action with numerical stability
            try:
                q_values = np.dot(state, self.q_weights)
                q_values = np.clip(q_values, -10, 10)  # Clip to prevent overflow
                
                actions = []
                for i in range(2):  # 2 symbols
                    symbol_q = q_values[i*3:(i+1)*3] if i*3+3 <= len(q_values) else q_values[:3]
                    actions.append(np.argmax(symbol_q))
                
                return actions
            except:
                # Fallback to random
                return [np.random.randint(0, 3) for _ in range(2)]

    def update(self, state, actions, reward, next_state, done):
        """Update agent with numerical stability"""
        try:
            # Clip inputs
            state = np.clip(state, -10, 10)
            next_state = np.clip(next_state, -10, 10)
            reward = np.clip(reward, -1, 1)
            
            # Current Q-values
            q_current = np.dot(state, self.q_weights)
            q_current = np.clip(q_current, -10, 10)
            
            # Next Q-values
            q_next = np.dot(next_state, self.q_weights)
            q_next = np.clip(q_next, -10, 10)
            
            # Update for each action
            for i, action in enumerate(actions):
                if i*3+action < len(q_current):
                    target = reward
                    if not done and i*3+2 < len(q_next):
                        target += 0.9 * np.max(q_next[i*3:(i+1)*3])
                    
                    target = np.clip(target, -10, 10)
                    error = target - q_current[i*3 + action]
                    error = np.clip(error, -1, 1)
                    
                    # Gradient update with clipping
                    gradient = self.learning_rate * error * state
                    gradient = np.clip(gradient, -0.1, 0.1)
                    
                    self.q_weights[:, i*3 + action] += gradient
                    
            # Clip weights to prevent explosion
            self.q_weights = np.clip(self.q_weights, -5, 5)
            
            # Decay exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            print(f"   Update error: {e}")

async def run_stable_rl_training():
    """Main stable RL training loop"""
    print("\\n>> STARTING STABLE RL TRAINING")
    
    env = StableRLEnvironment()
    state_size = len(env.get_normalized_state())
    agent = StableRLAgent(state_size)
    
    episode = 0
    
    try:
        while True:
            episode += 1
            state = env.reset()
            total_reward = 0
            
            print(f"\\n>> EPISODE {episode} - Learning optimal strategies...")
            
            for step in range(env.episode_length):
                actions = agent.get_action(state)
                next_state, reward, done, info = env.step(actions)
                
                agent.update(state, actions, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if step % 10 == 0:
                    print(f"   Step {step}: ${info['portfolio_value']:,.0f} | Return: {info['return']:+.1%}")
            
            agent.episode_rewards.append(total_reward)
            
            # Episode summary
            avg_reward = np.mean(agent.episode_rewards)
            print(f">> Episode {episode} Summary:")
            print(f"   Total Reward: {total_reward:+.3f}")
            print(f"   Avg Reward (last 10): {avg_reward:+.3f}")
            print(f"   Exploration: {agent.epsilon:.3f}")
            print(f"   Final Return: {info['return']:+.2%}")
            
            # Wait between episodes
            await asyncio.sleep(45)  # 45 seconds
            
    except KeyboardInterrupt:
        print("\\n>> STABLE RL TRAINING STOPPED")
        print(f">> Episodes: {episode}")
        print(f">> Final avg reward: {np.mean(agent.episode_rewards):+.3f}")

if __name__ == "__main__":
    print("\\nStarting Stable RL Training Environment...")
    asyncio.run(run_stable_rl_training())