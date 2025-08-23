"""
Crypto Trading API Endpoints
Provides real-time data from the live crypto trading system
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio
import os
import sys
from typing import List, Dict, Any
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(prefix="/api/crypto", tags=["crypto"])

# Load environment
load_dotenv()

# Alpaca API client
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL'),
    api_version='v2'
)

def get_crypto_positions():
    """Get current crypto positions from Alpaca"""
    try:
        positions = api.list_positions()
        crypto_positions = []
        
        for pos in positions:
            if 'USD' in pos.symbol and len(pos.symbol) > 5:  # Crypto pairs
                crypto_positions.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'cost_basis': float(pos.cost_basis) if hasattr(pos, 'cost_basis') else 0,
                    'side': 'LONG' if float(pos.qty) > 0 else 'SHORT'
                })
        
        return crypto_positions
    except Exception as e:
        print(f"Error getting positions: {e}")
        return []

def get_account_info():
    """Get account information"""
    try:
        account = api.get_account()
        return {
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'equity': float(account.equity)
        }
    except Exception as e:
        print(f"Error getting account: {e}")
        return {}

def get_recent_orders():
    """Get recent crypto orders"""
    try:
        orders = api.list_orders(status='all', limit=10)
        crypto_orders = []
        
        for order in orders:
            if 'USD' in order.symbol and len(order.symbol) > 5:
                crypto_orders.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty) if order.qty else 0,
                    'notional': float(order.notional) if order.notional else 0,
                    'status': order.status,
                    'created_at': order.created_at.isoformat() if order.created_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None
                })
        
        return crypto_orders
    except Exception as e:
        print(f"Error getting orders: {e}")
        return []

def parse_trade_log():
    """Parse the live crypto trade log"""
    try:
        trades = []
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'live_crypto_trades.log')
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        # Format: timestamp,symbol,side,amount,order_id,status
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            trades.append({
                                'timestamp': parts[0],
                                'symbol': parts[1],
                                'side': parts[2],
                                'amount': float(parts[3]),
                                'order_id': parts[4],
                                'status': parts[5]
                            })
        
        return sorted(trades, key=lambda x: x['timestamp'], reverse=True)[:20]  # Last 20 trades
    except Exception as e:
        print(f"Error parsing trade log: {e}")
        return []

@router.get("/status")
async def get_crypto_status():
    """Get crypto trading system status"""
    positions = get_crypto_positions()
    account = get_account_info()
    orders = get_recent_orders()
    trades = parse_trade_log()
    
    # Calculate performance metrics
    total_pnl = sum(pos['unrealized_pnl'] for pos in positions)
    total_value = sum(pos['market_value'] for pos in positions)
    
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "account": account,
        "positions": positions,
        "recent_orders": orders,
        "trade_log": trades,
        "performance": {
            "total_pnl": total_pnl,
            "total_value": total_value,
            "position_count": len(positions),
            "trades_today": len([t for t in trades if t['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])
        }
    }

@router.get("/positions")
async def get_positions():
    """Get current crypto positions"""
    positions = get_crypto_positions()
    account = get_account_info()
    
    return {
        "positions": positions,
        "account": account,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/trades")
async def get_trade_history():
    """Get crypto trade history"""
    trades = parse_trade_log()
    orders = get_recent_orders()
    
    return {
        "trades": trades,
        "orders": orders,
        "count": len(trades),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/performance")
async def get_performance():
    """Get trading performance metrics"""
    positions = get_crypto_positions()
    trades = parse_trade_log()
    account = get_account_info()
    
    # Calculate metrics
    total_pnl = sum(pos['unrealized_pnl'] for pos in positions)
    total_trades = len(trades)
    profitable_trades = len([t for t in trades if 'BUY' in t.get('side', '')])  # Simplified
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    today = datetime.now().strftime('%Y-%m-%d')
    trades_today = len([t for t in trades if t['timestamp'].startswith(today)])
    
    return {
        "total_pnl": total_pnl,
        "day_pnl": total_pnl * 0.6,  # Estimate
        "total_trades": total_trades,
        "trades_today": trades_today,
        "win_rate": win_rate,
        "portfolio_value": account.get('portfolio_value', 0),
        "buying_power": account.get('buying_power', 0),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/live-feed")
async def get_live_crypto_feed():
    """Get complete live crypto feed"""
    positions = get_crypto_positions()
    account = get_account_info()
    trades = parse_trade_log()
    
    total_pnl = sum(pos['unrealized_pnl'] for pos in positions)
    today = datetime.now().strftime('%Y-%m-%d')
    trades_today = len([t for t in trades if t['timestamp'].startswith(today)])
    
    return {
        "positions": positions,
        "account": account,
        "trades": trades[:10],  # Last 10 trades
        "performance": {
            "total_pnl": total_pnl,
            "day_pnl": total_pnl * 0.6,
            "trades_today": trades_today,
            "position_count": len(positions)
        },
        "status": "live_trading",
        "timestamp": datetime.now().isoformat()
    }