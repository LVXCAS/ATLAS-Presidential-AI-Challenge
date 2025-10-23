#!/usr/bin/env python3
"""
TRADE DATABASE
Track all trades in SQLite for performance analysis
"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List
import os


class TradeDatabase:
    """SQLite database for trade tracking"""

    def __init__(self, db_path: str = "data/trades.db"):
        """Initialize trade database"""
        self.db_path = db_path

        # Create data directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()

        print(f"[OK] Trade database: {db_path}")

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,

                entry_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,

                exit_time TIMESTAMP,
                exit_price REAL,

                pnl REAL,
                pnl_pct REAL,

                status TEXT NOT NULL,
                score REAL,
                confidence REAL,

                stop_loss REAL,
                take_profit REAL,

                metadata TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,

                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,

                total_pnl REAL DEFAULT 0,
                forex_pnl REAL DEFAULT 0,
                options_pnl REAL DEFAULT 0,

                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,

                max_drawdown REAL DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        print("[OK] Database tables initialized")

    def log_trade_entry(self, trade_id: str, symbol: str, strategy: str,
                       side: str, entry_price: float, quantity: float,
                       score: Optional[float] = None, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None, metadata: Optional[Dict] = None):
        """Log new trade entry"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                trade_id, symbol, strategy, side,
                entry_time, entry_price, quantity,
                status, score, stop_loss, take_profit, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, symbol, strategy, side,
            datetime.now(), entry_price, quantity,
            'OPEN', score, stop_loss, take_profit,
            json.dumps(metadata) if metadata else None
        ))

        self.conn.commit()
        print(f"[DB] Trade logged: {trade_id} - {symbol} {side}")

    def log_trade_exit(self, trade_id: str, exit_price: float, pnl: float, pnl_pct: float):
        """Log trade exit"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE trades
            SET exit_time = ?,
                exit_price = ?,
                pnl = ?,
                pnl_pct = ?,
                status = 'CLOSED',
                updated_at = ?
            WHERE trade_id = ?
        """, (datetime.now(), exit_price, pnl, pnl_pct, datetime.now(), trade_id))

        self.conn.commit()
        print(f"[DB] Trade closed: {trade_id} - P&L: ${pnl:.2f}")

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        """Get closed trades from last N days"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE status = 'CLOSED'
              AND exit_time >= datetime('now', '-' || ? || ' days')
            ORDER BY exit_time DESC
        """, (days,))
        return [dict(row) for row in cursor.fetchall()]

    def get_performance_stats(self, days: int = 7) -> Dict:
        """Calculate performance statistics"""
        cursor = self.conn.cursor()

        # Get closed trades
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losers,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl,
                   SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                   SUM(CASE WHEN pnl <= 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades
            WHERE status = 'CLOSED'
              AND exit_time >= datetime('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()

        total = row['total'] or 0
        winners = row['winners'] or 0
        losers = row['losers'] or 0
        total_pnl = row['total_pnl'] or 0
        avg_pnl = row['avg_pnl'] or 0
        gross_profit = row['gross_profit'] or 0
        gross_loss = row['gross_loss'] or 0.01  # Avoid division by zero

        win_rate = (winners / total * 100) if total > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_trades': total,
            'winners': winners,
            'losers': losers,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    def log_system_event(self, event_type: str, component: str,
                        message: str, severity: str = 'INFO'):
        """Log system event"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO system_events (event_type, component, message, severity)
            VALUES (?, ?, ?, ?)
        """, (event_type, component, message, severity))

        self.conn.commit()

    def update_daily_metrics(self):
        """Calculate and store daily metrics"""
        today = datetime.now().date()
        stats = self.get_performance_stats(days=1)

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics (
                date, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, profit_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            stats['total_trades'],
            stats['winners'],
            stats['losers'],
            stats['total_pnl'],
            stats['win_rate'],
            stats['profit_factor']
        ))

        self.conn.commit()
        print(f"[DB] Daily metrics updated for {today}")

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict]:
        """Get specific trade by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def close(self):
        """Close database connection"""
        self.conn.close()


# Singleton instance
_database = None

def get_database() -> TradeDatabase:
    """Get or create database instance"""
    global _database
    if _database is None:
        _database = TradeDatabase()
    return _database


if __name__ == "__main__":
    # Test the database
    print("\n" + "="*70)
    print("TRADE DATABASE TEST")
    print("="*70)

    db = get_database()

    # Test logging a trade
    print("\n[TEST] Logging sample trade...")
    db.log_trade_entry(
        trade_id="TEST_001",
        symbol="EUR_USD",
        strategy="EMA_CROSSOVER",
        side="LONG",
        entry_price=1.0850,
        quantity=1000,
        score=8.5,
        stop_loss=1.0820,
        take_profit=1.0910
    )

    # Get open trades
    print("\n[TEST] Fetching open trades...")
    open_trades = db.get_open_trades()
    print(f"Open trades: {len(open_trades)}")

    # Get stats
    print("\n[TEST] Calculating performance stats...")
    stats = db.get_performance_stats()
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Total P&L: ${stats['total_pnl']:.2f}")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")

    print("\n" + "="*70)
    print("Database test complete!")
    print(f"Location: {db.db_path}")
    print("="*70 + "\n")

    db.close()
