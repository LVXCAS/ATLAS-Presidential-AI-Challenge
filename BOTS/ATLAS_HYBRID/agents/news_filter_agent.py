"""
News Filter Agent

Blocks trades and auto-closes positions before major economic events.

THIS IS THE AGENT THAT WOULD HAVE SAVED YOUR $8K PROFIT.

Specialization: News event detection and risk protection.
VETO power: Can block all trades regardless of other agent votes.
"""

from typing import Dict, Tuple, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent

# Import news filter from existing implementation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from news_filter import NewsFilter
    NEWS_FILTER_AVAILABLE = True
except ImportError:
    NEWS_FILTER_AVAILABLE = False
    print("[WARNING] NewsFilter not available - will use basic time-based protection")


class NewsFilterAgent(BaseAgent):
    """
    News event protection agent.

    Responsibilities:
    1. Block new trades before major news events (60 min buffer)
    2. Auto-close existing positions before news (30 min buffer)
    3. VETO power - overrides all other agents

    High-impact events monitored:
    - NFP (Non-Farm Payroll) - First Friday, 8:30 AM EST
    - FOMC (Federal Reserve) - 8x/year, 2:00 PM EST
    - CPI (Consumer Price Index) - Monthly, 8:30 AM EST
    - GDP, Retail Sales, ECB/BOE/BOJ decisions
    """

    def __init__(self, initial_weight: float = 2.0):
        """
        Initialize with VETO power (weight 2.0).

        When this agent votes BLOCK, trade is blocked regardless
        of other agent scores.
        """
        super().__init__(name="NewsFilterAgent", initial_weight=initial_weight)

        # Initialize news filter
        if NEWS_FILTER_AVAILABLE:
            self.news_filter = NewsFilter()
        else:
            self.news_filter = None

        # Time buffers
        self.block_new_trades_buffer = 60  # minutes before news
        self.auto_close_buffer = 30  # minutes before news

        # High-impact event times (if API unavailable, use known schedule)
        self.known_events = self._load_known_events()

    def _load_known_events(self) -> List[Dict]:
        """
        Load known high-impact event schedule.

        Backup if news API unavailable.
        """
        # NFP: First Friday of every month, 8:30 AM EST
        # FOMC: 8 times/year, 2:00 PM EST
        # CPI: Around 13th of every month, 8:30 AM EST

        return [
            # December 2025
            {"date": "2025-12-05 08:30", "event": "NFP", "currency": "USD"},
            {"date": "2025-12-11 08:30", "event": "CPI", "currency": "USD"},
            {"date": "2025-12-18 14:00", "event": "FOMC", "currency": "USD"},

            # January 2026
            {"date": "2026-01-09 08:30", "event": "NFP", "currency": "USD"},
            {"date": "2026-01-14 08:30", "event": "CPI", "currency": "USD"},

            # Add more as needed...
        ]

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        Check if news event is approaching.

        Returns:
            - "ALLOW" if safe to trade
            - "BLOCK" if news event approaching (VETO)
        """
        pair = market_data.get("pair", "EUR_USD")
        current_time = market_data.get("time", datetime.now())

        # Check for upcoming news events
        if self.news_filter:
            # Use real news API
            is_safe, news_msg = self.news_filter.check_news_safety(pair)

            if not is_safe:
                return ("BLOCK", 1.0, {
                    "reason": "High-impact news event approaching",
                    "message": news_msg,
                    "veto": True
                })

        else:
            # Use known event schedule as backup
            upcoming = self._check_known_events(current_time)

            if upcoming:
                event = upcoming[0]
                return ("BLOCK", 1.0, {
                    "reason": f"{event['event']} approaching",
                    "time_until": event['minutes_until'],
                    "event_time": event['date'],
                    "veto": True
                })

        # No news events detected - safe to trade
        return ("ALLOW", 1.0, {
            "reason": "No high-impact news in next 60 minutes",
            "safe": True
        })

    def _check_known_events(self, current_time: datetime) -> List[Dict]:
        """
        Check known event schedule for upcoming events.

        Args:
            current_time: Current datetime

        Returns:
            List of upcoming events within block buffer
        """
        upcoming = []

        for event in self.known_events:
            event_time = datetime.fromisoformat(event['date'])
            time_diff = (event_time - current_time).total_seconds() / 60  # minutes

            if 0 < time_diff <= self.block_new_trades_buffer:
                upcoming.append({
                    **event,
                    'minutes_until': int(time_diff)
                })

        return upcoming

    def should_close_positions(self, market_data: Dict) -> Tuple[bool, List[Dict]]:
        """
        Check if existing positions should be closed due to approaching news.

        This is called separately by the coordinator to auto-close positions.

        Args:
            market_data: Current market state

        Returns:
            (should_close, events_list)
        """
        current_time = market_data.get("time", datetime.now())

        if self.news_filter:
            # Use real news API
            should_close, events = self.news_filter.should_close_positions_before_news(
                minutes_ahead=self.auto_close_buffer
            )
            return (should_close, events)

        else:
            # Check known events with auto-close buffer
            upcoming = []

            for event in self.known_events:
                event_time = datetime.fromisoformat(event['date'])
                time_diff = (event_time - current_time).total_seconds() / 60

                if 0 < time_diff <= self.auto_close_buffer:
                    upcoming.append(event)

            return (len(upcoming) > 0, upcoming)

    def get_affected_pairs(self, event_currency: str, all_pairs: List[str]) -> List[str]:
        """
        Get list of pairs affected by a currency's news event.

        Args:
            event_currency: Currency with news (e.g., "USD")
            all_pairs: List of all trading pairs

        Returns:
            List of affected pairs
        """
        if self.news_filter:
            return self.news_filter.get_affected_pairs(event_currency, all_pairs)

        # Simple implementation
        affected = []
        for pair in all_pairs:
            pair_clean = pair.replace('_', '').replace('/', '')
            if event_currency in pair_clean:
                affected.append(pair)

        return affected

    def record_news_avoidance(self, event: Dict, positions_closed: int, pnl_saved: float):
        """
        Record when news protection saved us from potential loss.

        This helps track how valuable the news filter is.

        Args:
            event: News event that was avoided
            positions_closed: Number of positions auto-closed
            pnl_saved: Estimated P/L saved (unrealized P/L at close time)
        """
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "news_avoidance",
            "event": event.get("event", "Unknown"),
            "positions_closed": positions_closed,
            "pnl_saved": pnl_saved,
            "message": f"Auto-closed {positions_closed} positions before {event.get('event')}, saved ${pnl_saved:,.2f}"
        })

    def get_performance_metrics(self) -> Dict:
        """
        Override base method to include news-specific metrics.

        Returns:
            Performance metrics including news avoidances
        """
        base_metrics = super().get_performance_metrics()

        # Count news avoidances
        avoidances = [h for h in self.performance_history if h.get("type") == "news_avoidance"]

        total_saved = sum(h.get("pnl_saved", 0) for h in avoidances)
        total_positions_saved = sum(h.get("positions_closed", 0) for h in avoidances)

        base_metrics.update({
            "news_avoidances": len(avoidances),
            "positions_protected": total_positions_saved,
            "estimated_pnl_saved": total_saved,
            "agent_type": "VETO_POWER"
        })

        return base_metrics

    def adjust_weight(self, learning_rate: float = 0.0):
        """
        Override weight adjustment - NewsFilterAgent maintains VETO weight.

        This agent's weight should NOT change based on performance.
        It's a safety mechanism, not a profit optimizer.
        """
        # Lock weight at 2.0 (VETO power)
        self.weight = 2.0
