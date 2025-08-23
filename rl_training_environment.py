#!/usr/bin/env python3
"""
HIVE TRADE - Reinforcement Learning Training Environment
Parallel training system that learns optimal strategies WITHOUT using real money
"""

import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
from datetime import datetime, timedelta
from collections import deque
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    amount: float
    entry_price: float
    exit_price: float = None
    pnl: float = 0.0
    duration_minutes: int = 0
    
class TradingEnvironment:
    """Simulated trading environment for RL training"""
    
    def __init__(self, initial_balance=100000):
        print("=" * 60)
        print("HIVE TRADE - RL TRAINING ENVIRONMENT")  
        print("PARALLEL TRAINING SYSTEM (NO REAL MONEY)")
        print("=" * 60)
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # symbol -> {qty, entry_price, timestamp}
        self.trade_history = []
        self.performance_history = []
        
        # Crypto pairs for training
        self.symbols = ['BTC-USD', 'ETH-USD']
        self.current_prices = {}
        
        # RL Parameters
        self.episode_length = 100  # 100 trading decisions per episode
        self.step_count = 0
        self.episode_count = 0
        
        print(f">> Training Environment Initialized")
        print(f"   Virtual Balance: ${self.balance:,.2f}")
        print(f"   Training Symbols: {self.symbols}")
        print(f"   Episode Length: {self.episode_length} steps")

    def get_market_data(self, symbol, period="5d"):
        """Get real market data for simulation"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 0:
                return {
                    'price': hist['Close'].iloc[-1],
                    'volume': hist['Volume'].iloc[-1],
                    'high': hist['High'].iloc[-1],
                    'low': hist['Low'].iloc[-1],
                    'change': (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                }
            return None
        except Exception as e:
            print(f"   Market data error for {symbol}: {e}")
            return None

    def get_state_vector(self):
        """Get current market state as vector for RL agent"""
        state = []
        
        for symbol in self.symbols:
            data = self.get_market_data(symbol)
            if data:
                self.current_prices[symbol] = data['price']
                
                # Market features
                state.extend([
                    data['price'] / 50000,  # Normalized price
                    data['change'],  # Price change
                    data['volume'] / 1e6,  # Normalized volume
                ])
                
                # Position features
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    unrealized_pnl = (data['price'] - pos['entry_price']) * pos['qty']
                    state.extend([
                        pos['qty'] / 1000,  # Normalized position size
                        unrealized_pnl / 1000,  # Normalized P&L
                    ])
                else:
                    state.extend([0.0, 0.0])  # No position
        
        # Portfolio features
        total_value = self.calculate_portfolio_value()
        state.extend([
            self.balance / self.initial_balance,  # Cash ratio
            total_value / self.initial_balance,  # Total value ratio
            len(self.positions) / len(self.symbols),  # Position ratio
        ])
        
        return np.array(state)

    def calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        total_value = self.balance
        
        for symbol, pos in self.positions.items():
            if symbol in self.current_prices:
                market_value = self.current_prices[symbol] * pos['qty']
                total_value += market_value
        
        return total_value

    def execute_action(self, action, symbol):
        """Execute trading action in simulation"""
        # Actions: 0=HOLD, 1=BUY, 2=SELL
        
        if symbol not in self.current_prices:
            return -0.1, "No price data"  # Small penalty
        
        current_price = self.current_prices[symbol]
        trade_amount = 75  # Same as live system
        
        if action == 1:  # BUY
            if self.balance >= trade_amount:
                qty = trade_amount / current_price
                
                if symbol in self.positions:
                    # Add to existing position
                    old_pos = self.positions[symbol]
                    total_qty = old_pos['qty'] + qty
                    total_cost = (old_pos['entry_price'] * old_pos['qty']) + trade_amount
                    avg_price = total_cost / total_qty
                    
                    self.positions[symbol] = {
                        'qty': total_qty,
                        'entry_price': avg_price,
                        'timestamp': datetime.now()
                    }
                else:
                    self.positions[symbol] = {
                        'qty': qty,
                        'entry_price': current_price,
                        'timestamp': datetime.now()
                    }
                
                self.balance -= trade_amount
                
                # Record trade
                trade = Trade(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='BUY',
                    amount=trade_amount,
                    entry_price=current_price
                )
                self.trade_history.append(trade)
                
                return 0.05, f"BUY ${trade_amount} {symbol}"  # Small positive reward
            else:
                return -0.1, "Insufficient balance"
        
        elif action == 2:  # SELL
            if symbol in self.positions:
                pos = self.positions[symbol]
                sell_qty = min(pos['qty'], trade_amount / current_price)
                
                if sell_qty > 0:
                    proceeds = sell_qty * current_price
                    self.balance += proceeds
                    
                    # Calculate P&L
                    pnl = (current_price - pos['entry_price']) * sell_qty
                    
                    # Update position
                    remaining_qty = pos['qty'] - sell_qty
                    if remaining_qty < 0.0001:  # Close position
                        del self.positions[symbol]
                    else:
                        self.positions[symbol]['qty'] = remaining_qty
                    
                    # Record trade
                    trade = Trade(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action='SELL',
                        amount=proceeds,
                        entry_price=pos['entry_price'],
                        exit_price=current_price,
                        pnl=pnl
                    )
                    self.trade_history.append(trade)
                    
                    # Reward based on profit/loss
                    reward = pnl / trade_amount  # Normalized P&L reward
                    return reward, f"SELL ${proceeds:.2f} {symbol} (P&L: ${pnl:+.2f})"
                else:
                    return -0.05, "No position to sell"
            else:
                return -0.05, "No position to sell"
        
        else:  # HOLD
            return -0.001, "HOLD"  # Tiny penalty for inaction

    def step(self, actions):
        """Execute one step in the environment"""
        total_reward = 0
        action_results = []
        
        # Execute action for each symbol
        for i, symbol in enumerate(self.symbols):
            if i < len(actions):
                reward, message = self.execute_action(actions[i], symbol)
                total_reward += reward
                action_results.append(f"{symbol}: {message}")
        
        self.step_count += 1
        
        # Calculate portfolio performance
        portfolio_value = self.calculate_portfolio_value()
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Add portfolio performance reward
        total_reward += portfolio_return * 0.1
        
        # Record performance
        self.performance_history.append({
            'step': self.step_count,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'num_positions': len(self.positions),
            'total_trades': len(self.trade_history)
        })
        
        # Check if episode is done
        done = self.step_count >= self.episode_length
        
        # Get new state
        next_state = self.get_state_vector()
        
        return next_state, total_reward, done, {
            'portfolio_value': portfolio_value,
            'portfolio_return': portfolio_return,
            'actions': action_results,
            'trades_count': len(self.trade_history)
        }

    def reset(self):
        """Reset environment for new episode"""
        self.episode_count += 1
        
        # Record episode results
        final_value = self.calculate_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        print(f"\\n>> EPISODE {self.episode_count} COMPLETE:")
        print(f"   Final Portfolio: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2%}")
        print(f"   Total Trades: {len(self.trade_history)}")
        print(f"   Win Rate: {self.calculate_win_rate():.1%}")
        
        # Reset for next episode
        self.balance = self.initial_balance
        self.positions = {}
        self.step_count = 0
        
        # Keep some trade history for learning
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
        
        return self.get_state_vector()

    def calculate_win_rate(self):
        """Calculate win rate from completed trades"""
        completed_trades = [t for t in self.trade_history if t.pnl != 0]
        if not completed_trades:
            return 0.0
        
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        return winning_trades / len(completed_trades)

class SimpleRLAgent:
    """Simple Q-learning agent for trading"""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size  # 3 actions per symbol: HOLD, BUY, SELL
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        
        # Simple Q-table approximation using neural network weights
        self.weights = np.random.randn(state_size, action_size) * 0.1
        self.episode_rewards = []
        
        print(f">> RL Agent Initialized:")
        print(f"   State Size: {state_size}")
        print(f"   Action Size: {action_size}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Exploration Rate: {epsilon}")

    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Random exploration for each symbol
            return [np.random.randint(0, 3) for _ in range(2)]  # 2 symbols
        else:
            # Greedy action
            q_values = np.dot(state, self.weights)
            actions = []
            for i in range(2):  # For each symbol
                symbol_q = q_values[i*3:(i+1)*3]  # 3 actions per symbol
                actions.append(np.argmax(symbol_q))
            return actions

    def update(self, state, actions, reward, next_state, done):
        """Update agent based on experience"""
        # Simple gradient update
        q_current = np.dot(state, self.weights)
        q_next = np.dot(next_state, self.weights)
        
        # Target for each action taken
        for i, action in enumerate(actions):
            target = reward
            if not done:
                target += 0.95 * np.max(q_next[i*3:(i+1)*3])  # Discount factor 0.95
            
            # Update weights
            error = target - q_current[i*3 + action]
            self.weights[:, i*3 + action] += self.learning_rate * error * state
        
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.995)

async def run_rl_training():
    """Main RL training loop"""
    print("\\n>> STARTING RL TRAINING ENVIRONMENT")
    print(">> This runs parallel to live trading")
    print(">> Learning optimal strategies without real money")
    
    # Initialize environment and agent
    env = TradingEnvironment()
    state_size = len(env.get_state_vector())
    action_size = 6  # 3 actions × 2 symbols
    agent = SimpleRLAgent(state_size, action_size)
    
    episode = 0
    
    try:
        while True:
            episode += 1
            state = env.reset()
            episode_reward = 0
            
            print(f"\\n>> TRAINING EPISODE {episode}")
            
            # Run one episode
            for step in range(env.episode_length):
                actions = agent.get_action(state)
                next_state, reward, done, info = env.step(actions)
                
                agent.update(state, actions, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"   Step {step}: Portfolio ${info['portfolio_value']:,.2f} | Reward: {reward:+.3f}")
            
            agent.episode_rewards.append(episode_reward)
            
            # Print episode summary
            avg_reward = np.mean(agent.episode_rewards[-10:])
            print(f">> Episode {episode} Complete:")
            print(f"   Episode Reward: {episode_reward:+.3f}")
            print(f"   Avg Reward (10ep): {avg_reward:+.3f}")
            print(f"   Exploration Rate: {agent.epsilon:.3f}")
            
            # Wait between episodes
            await asyncio.sleep(30)  # 30 seconds between episodes
            
    except KeyboardInterrupt:
        print("\\n>> RL TRAINING STOPPED")
        print(f">> Completed {episode} training episodes")
        print(f">> Final avg reward: {np.mean(agent.episode_rewards[-10:]):+.3f}")

if __name__ == "__main__":
    print("\\nStarting HIVE TRADE RL Training Environment...")
    print("This is the TRAINING system (parallel to live trading)")
    asyncio.run(run_rl_training())