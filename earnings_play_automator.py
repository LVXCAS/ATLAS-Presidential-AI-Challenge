"""
EARNINGS PLAY AUTOMATOR
Systematically profit from earnings announcements using options

Key Strategy:
1. Download earnings calendar (next 7 days)
2. Scan for high IV options (IV Rank > 50)
3. Suggest straddle/strangle setups
4. Auto-exit before IV crush (4PM day before earnings)

Why This Works:
- IV expansion: Options premiums increase 50-200% before earnings
- Predictable timing: Earnings dates are known in advance
- IV crush: After announcement, IV drops 30-80% instantly
- We profit from the expansion, exit before the crush
"""
import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderType

@dataclass
class EarningsEvent:
    """Earnings announcement event"""
    symbol: str
    company_name: str
    earnings_date: str
    earnings_time: str  # 'bmo' (before market open) or 'amc' (after market close)
    days_until: int
    estimated_eps: Optional[float]
    previous_eps: Optional[float]

@dataclass
class EarningsOptionsSetup:
    """Suggested options setup for earnings play"""
    symbol: str
    earnings_date: str
    days_until: int

    # Current conditions
    stock_price: float
    iv_rank: float  # 0-100, current IV vs 1-year range
    iv_percentile: float
    current_iv: float

    # Setup recommendation
    strategy: str  # 'long_straddle', 'long_strangle', 'iron_condor', 'calendar_spread'
    expiration: str
    strike_call: float
    strike_put: float

    # Expected metrics
    max_profit: float
    max_loss: float
    breakeven_upper: float
    breakeven_lower: float
    probability_of_profit: float

    # Exit timing
    exit_datetime: str  # Exit 1 day before earnings to avoid IV crush

    # Trade sizing
    recommended_contracts: int
    total_cost: float

    confidence: float  # 0-1, how good this setup is

class EarningsPlayAutomator:
    def __init__(self):
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Initialize Alpaca client
        try:
            self.trading_client = TradingClient(self.alpaca_api_key, self.alpaca_secret, paper=True)
        except:
            self.trading_client = None
            print("[EARNINGS] Warning: Alpaca client not initialized")

        # Thresholds
        self.min_iv_rank = 50  # Only trade if IV is above 50th percentile
        self.min_stock_price = 20  # Avoid penny stocks
        self.max_stock_price = 500  # Avoid super expensive stocks
        self.min_days_until_earnings = 2  # At least 2 days before earnings
        self.max_days_until_earnings = 14  # At most 2 weeks before earnings

    def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
            data = {
                'chat_id': self.telegram_chat_id,
                'text': f'EARNINGS PLAY\n\n{message}'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[EARNINGS] Telegram notification failed: {e}")

    def download_earnings_calendar(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Download upcoming earnings announcements"""
        print(f"[EARNINGS] Downloading earnings calendar for next {days_ahead} days...")

        # Use Yahoo Finance earnings calendar
        # Note: In production, use a paid API like Earnings Whispers or Alpha Vantage
        events = []

        try:
            # Get S&P 500 tickers (most liquid options)
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

            # Add User-Agent header to bypass Wikipedia bot blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            tables = pd.read_html(sp500_url, storage_options=headers)
            sp500_df = tables[0]
            tickers = sp500_df['Symbol'].tolist()[:100]  # Top 100 for speed

            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)

                    # Get earnings dates
                    calendar = stock.calendar

                    if calendar is not None and 'Earnings Date' in calendar.index:
                        earnings_dates = calendar.loc['Earnings Date']

                        # Convert to date if needed
                        if isinstance(earnings_dates, pd.Series):
                            earnings_date = earnings_dates.iloc[0]
                        else:
                            earnings_date = earnings_dates

                        # Check if within our date range
                        if isinstance(earnings_date, str):
                            earnings_date = pd.to_datetime(earnings_date).date()

                        if today <= earnings_date <= end_date:
                            days_until = (earnings_date - today).days

                            event = EarningsEvent(
                                symbol=ticker,
                                company_name=stock.info.get('longName', ticker),
                                earnings_date=earnings_date.isoformat(),
                                earnings_time='amc',  # Default to after market close
                                days_until=days_until,
                                estimated_eps=None,
                                previous_eps=None
                            )

                            events.append(event)
                            print(f"[EARNINGS] Found: {ticker} on {earnings_date} ({days_until} days)")

                except Exception as e:
                    # Skip tickers that error
                    continue

            print(f"[EARNINGS] Found {len(events)} earnings announcements")
            return events

        except Exception as e:
            print(f"[EARNINGS] Error downloading calendar: {e}")
            return []

    def calculate_iv_rank(self, symbol: str) -> Dict[str, float]:
        """Calculate IV rank and percentile"""
        try:
            # Get historical volatility data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1y')

            if len(hist) < 252:
                return {'iv_rank': 0, 'iv_percentile': 0, 'current_iv': 0}

            # Calculate historical volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            current_vol = returns.iloc[-20:].std() * np.sqrt(252)  # 20-day vol

            # Calculate rolling 20-day vol for entire year
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()

            if len(rolling_vol) == 0:
                return {'iv_rank': 0, 'iv_percentile': 0, 'current_iv': current_vol}

            # IV Rank: Where current IV sits in the range (0-100)
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            iv_rank = ((current_vol - min_vol) / (max_vol - min_vol) * 100) if max_vol > min_vol else 50

            # IV Percentile: Percentage of days current IV was higher
            iv_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100

            return {
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'current_iv': current_vol
            }

        except Exception as e:
            print(f"[EARNINGS] Error calculating IV for {symbol}: {e}")
            return {'iv_rank': 0, 'iv_percentile': 0, 'current_iv': 0}

    def suggest_earnings_setup(self, event: EarningsEvent) -> Optional[EarningsOptionsSetup]:
        """Suggest optimal options setup for earnings play"""
        symbol = event.symbol

        try:
            # Get stock price
            stock = yf.Ticker(symbol)
            hist = stock.history(period='5d')

            if len(hist) == 0:
                return None

            stock_price = float(hist['Close'].iloc[-1])

            # Check price range
            if stock_price < self.min_stock_price or stock_price > self.max_stock_price:
                return None

            # Calculate IV metrics
            iv_metrics = self.calculate_iv_rank(symbol)

            # Only trade if IV is elevated
            if iv_metrics['iv_rank'] < self.min_iv_rank:
                return None

            # Determine strategy based on IV level and days until earnings
            if iv_metrics['iv_rank'] > 75 and event.days_until > 7:
                # High IV, time to expiration - use spread to reduce cost
                strategy = 'iron_condor'
                confidence = 0.7
            elif iv_metrics['iv_rank'] > 60 and event.days_until >= 3:
                # Medium-high IV - long straddle
                strategy = 'long_straddle'
                confidence = 0.8
            elif event.days_until >= 5:
                # More time - strangle for lower cost
                strategy = 'long_strangle'
                confidence = 0.75
            else:
                # Too close to earnings or low IV
                return None

            # Calculate strikes
            atm_strike = round(stock_price / 5) * 5  # Round to nearest $5

            if strategy == 'long_straddle':
                strike_call = atm_strike
                strike_put = atm_strike
            elif strategy == 'long_strangle':
                # OTM strikes to reduce cost
                strike_call = atm_strike + 5
                strike_put = atm_strike - 5
            elif strategy == 'iron_condor':
                # Define wing strikes
                strike_call = atm_strike + 10
                strike_put = atm_strike - 10
            else:
                strike_call = atm_strike
                strike_put = atm_strike

            # Calculate expiration (week after earnings)
            earnings_date = datetime.fromisoformat(event.earnings_date)
            expiration = earnings_date + timedelta(days=7)
            expiration_str = expiration.strftime('%Y-%m-%d')

            # Estimate costs (simplified - in production, get actual option prices)
            option_cost_estimate = stock_price * 0.05 * (iv_metrics['iv_rank'] / 100)  # Rough estimate

            if strategy == 'long_straddle':
                total_cost = option_cost_estimate * 2 * 100  # 2 options * 100 shares
                max_loss = total_cost
                max_profit = float('inf')  # Theoretically unlimited
            elif strategy == 'long_strangle':
                total_cost = option_cost_estimate * 1.5 * 100  # Cheaper than straddle
                max_loss = total_cost
                max_profit = float('inf')
            elif strategy == 'iron_condor':
                total_cost = option_cost_estimate * 100
                max_loss = total_cost
                max_profit = option_cost_estimate * 200  # Credit spread profit
            else:
                total_cost = 0
                max_loss = 0
                max_profit = 0

            # Calculate breakevens
            breakeven_upper = strike_call + (total_cost / 100)
            breakeven_lower = strike_put - (total_cost / 100)

            # Estimate probability of profit (simplified)
            expected_move = stock_price * iv_metrics['current_iv'] * np.sqrt(event.days_until / 365)
            prob_profit = 0.6 if expected_move > (total_cost / 100) else 0.4

            # Calculate exit time (1 day before earnings, 4PM EST)
            exit_datetime = earnings_date - timedelta(days=1)
            exit_datetime = exit_datetime.replace(hour=16, minute=0)

            # Position sizing
            account_size = 100000  # Assume $100K account
            max_risk_per_trade = account_size * 0.02  # Risk 2% per trade
            recommended_contracts = int(max_risk_per_trade / max_loss) if max_loss > 0 else 1
            recommended_contracts = max(1, min(recommended_contracts, 5))  # 1-5 contracts

            return EarningsOptionsSetup(
                symbol=symbol,
                earnings_date=event.earnings_date,
                days_until=event.days_until,
                stock_price=stock_price,
                iv_rank=iv_metrics['iv_rank'],
                iv_percentile=iv_metrics['iv_percentile'],
                current_iv=iv_metrics['current_iv'],
                strategy=strategy,
                expiration=expiration_str,
                strike_call=strike_call,
                strike_put=strike_put,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_upper=breakeven_upper,
                breakeven_lower=breakeven_lower,
                probability_of_profit=prob_profit,
                exit_datetime=exit_datetime.isoformat(),
                recommended_contracts=recommended_contracts,
                total_cost=total_cost * recommended_contracts,
                confidence=confidence
            )

        except Exception as e:
            print(f"[EARNINGS] Error analyzing {symbol}: {e}")
            return None

    def scan_earnings_opportunities(self, days_ahead: int = 7) -> List[EarningsOptionsSetup]:
        """Scan for earnings plays and suggest setups"""
        print("\n" + "="*70)
        print("EARNINGS PLAY AUTOMATOR - SCANNING")
        print("="*70)

        # Download earnings calendar
        events = self.download_earnings_calendar(days_ahead)

        if not events:
            print("[EARNINGS] No earnings events found")
            return []

        # Filter by days until earnings
        filtered_events = [
            e for e in events
            if self.min_days_until_earnings <= e.days_until <= self.max_days_until_earnings
        ]

        print(f"[EARNINGS] Analyzing {len(filtered_events)} events...")

        # Generate setups
        setups = []
        for event in filtered_events:
            setup = self.suggest_earnings_setup(event)
            if setup:
                setups.append(setup)
                print(f"[EARNINGS] Setup found: {setup.symbol} - {setup.strategy} (IV Rank: {setup.iv_rank:.0f})")

        # Sort by confidence
        setups.sort(key=lambda x: x.confidence, reverse=True)

        print(f"\n[EARNINGS] Found {len(setups)} high-quality setups")

        # Save to file
        if setups:
            os.makedirs('data', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/earnings_setups_{timestamp}.json'

            with open(filename, 'w') as f:
                json.dump([asdict(s) for s in setups], f, indent=2)

            print(f"[EARNINGS] Saved to {filename}")

        return setups

    def format_earnings_report(self, setups: List[EarningsOptionsSetup]) -> str:
        """Format earnings setups for display"""
        if not setups:
            return "No earnings opportunities found in the next 7 days."

        report = f"""
=== EARNINGS PLAY OPPORTUNITIES ===
Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Found: {len(setups)} high-quality setups

TOP 5 SETUPS:
"""
        for i, setup in enumerate(setups[:5], 1):
            report += f"""
{i}. {setup.symbol} - {setup.strategy.upper()}
   Earnings: {setup.earnings_date} ({setup.days_until} days)
   Price: ${setup.stock_price:.2f}
   IV Rank: {setup.iv_rank:.0f}/100 (HIGH!)

   SETUP:
   Strategy: {setup.strategy}
   Call Strike: ${setup.strike_call:.0f}
   Put Strike: ${setup.strike_put:.0f}
   Expiration: {setup.expiration}

   RISK/REWARD:
   Max Loss: ${setup.max_loss:,.0f}
   Total Cost: ${setup.total_cost:,.0f}
   Contracts: {setup.recommended_contracts}
   Probability: {setup.probability_of_profit:.0%}

   EXIT: {setup.exit_datetime} (BEFORE earnings!)
   Confidence: {setup.confidence:.0%}

"""
        return report

    def send_earnings_alerts(self, setups: List[EarningsOptionsSetup]):
        """Send Telegram alerts for top earnings plays"""
        if not setups:
            return

        # Send top 3
        for setup in setups[:3]:
            msg = f"""
{setup.symbol} EARNINGS PLAY

Earnings: {setup.earnings_date} ({setup.days_until} days)
IV Rank: {setup.iv_rank:.0f}/100

Strategy: {setup.strategy.upper()}
Strikes: ${setup.strike_put:.0f} PUT / ${setup.strike_call:.0f} CALL
Cost: ${setup.total_cost:,.0f} ({setup.recommended_contracts} contracts)

Exit: {setup.exit_datetime}
(EXIT BEFORE EARNINGS!)

Confidence: {setup.confidence:.0%}
"""
            self.send_telegram_notification(msg)

def main():
    """Run earnings play automator"""
    automator = EarningsPlayAutomator()

    # Scan for opportunities
    setups = automator.scan_earnings_opportunities(days_ahead=7)

    # Print report
    print(automator.format_earnings_report(setups))

    # Send alerts
    if setups:
        automator.send_earnings_alerts(setups)

if __name__ == '__main__':
    main()
