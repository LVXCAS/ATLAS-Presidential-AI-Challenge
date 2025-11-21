"""
E8 MARKET MICROSTRUCTURE SETUP DETECTOR
Phase 1: Detection Only - No Trading

Detects institutional behavior patterns:
1. London Fake-out (stop hunts)
2. NY Absorption (order flow imbalances)
3. Tokyo Gap Fill (mechanical reversion)

Logs every setup for statistical validation.
"""

import os
import json
from datetime import datetime, time as dt_time
from E8_TRADELOCKER_ADAPTER import E8TradeLockerAdapter
import time

class SetupDetector:
    """
    Detects market microstructure setups without executing trades.
    Builds statistical evidence through observation.
    """

    def __init__(self):
        self.adapter = E8TradeLockerAdapter()
        self.log_file = "setup_detection_log.json"
        self.setups_found = []

        print("=" * 70)
        print("E8 SETUP DETECTOR - PHASE 1: OBSERVATION ONLY")
        print("=" * 70)
        print("Mode: Detection & Logging (NO TRADING)")
        print("Goal: Validate edge through data collection")
        print("Duration: 2 weeks minimum")
        print("=" * 70)

    def detect_london_fakeout(self, pair):
        """
        Detect: Low volume breakout of Asian range = likely fake
        Theory: Retail gets stopped, institutions reverse

        Returns: setup dict or None
        """
        try:
            # Get current price
            candles = self.adapter.get_candles(pair, count=20, granularity='M15')
            if not candles or len(candles) < 10:
                return None

            current_price = candles[-1]['mid']['c']

            # Calculate Asian range (last 8 hours = 32 x 15min candles)
            asian_candles = candles[-32:] if len(candles) >= 32 else candles
            asian_high = max([c['mid']['h'] for c in asian_candles])
            asian_low = min([c['mid']['l'] for c in asian_candles])
            asian_range = asian_high - asian_low

            # Calculate average volume
            avg_volume = sum([c.get('volume', 0) for c in candles[-10:]]) / 10
            recent_volume = candles[-1].get('volume', 0)

            # Detection logic
            breakout_buffer = asian_range * 0.02  # 2% beyond range

            if current_price > asian_high + breakout_buffer:
                # Broke above Asian high
                if recent_volume < avg_volume * 0.7:  # Low volume = fake
                    return {
                        'setup': 'london_fakeout',
                        'pair': pair,
                        'direction': 'SHORT',  # Fade the breakout
                        'entry': current_price,
                        'stop': asian_high + (asian_range * 0.05),
                        'target': asian_low,
                        'confidence': 0.82,
                        'reason': 'Low volume break above Asian range',
                        'timestamp': datetime.utcnow().isoformat()
                    }

            elif current_price < asian_low - breakout_buffer:
                # Broke below Asian low
                if recent_volume < avg_volume * 0.7:  # Low volume = fake
                    return {
                        'setup': 'london_fakeout',
                        'pair': pair,
                        'direction': 'LONG',  # Fade the breakout
                        'entry': current_price,
                        'stop': asian_low - (asian_range * 0.05),
                        'target': asian_high,
                        'confidence': 0.82,
                        'reason': 'Low volume break below Asian range',
                        'timestamp': datetime.utcnow().isoformat()
                    }

            return None

        except Exception as e:
            print(f"[ERROR] London fakeout detection: {e}")
            return None

    def detect_ny_absorption(self, pair):
        """
        Detect: Price rejection at previous day high/low
        Theory: Institutional orders absorb retail flow

        Returns: setup dict or None
        """
        try:
            # Get daily and hourly candles
            daily_candles = self.adapter.get_candles(pair, count=5, granularity='D')
            hourly_candles = self.adapter.get_candles(pair, count=24, granularity='H1')

            if not daily_candles or not hourly_candles:
                return None

            # Previous day high/low
            prev_day = daily_candles[-2] if len(daily_candles) >= 2 else daily_candles[-1]
            prev_high = prev_day['mid']['h']
            prev_low = prev_day['mid']['l']

            # Current price
            current = hourly_candles[-1]['mid']['c']
            current_high = hourly_candles[-1]['mid']['h']
            current_low = hourly_candles[-1]['mid']['l']

            # Check for rejection (wick at least 2x body size)
            body_size = abs(hourly_candles[-1]['mid']['c'] - hourly_candles[-1]['mid']['o'])

            # Rejection at previous day high?
            if current_high >= prev_high * 0.9998:  # Within 2 pips
                upper_wick = current_high - max(hourly_candles[-1]['mid']['o'], current)
                if upper_wick > body_size * 2:
                    return {
                        'setup': 'ny_absorption',
                        'pair': pair,
                        'direction': 'SHORT',
                        'entry': current,
                        'stop': prev_high + (body_size * 1.5),
                        'target': prev_low,
                        'confidence': 0.76,
                        'reason': 'Rejection at previous day high',
                        'timestamp': datetime.utcnow().isoformat()
                    }

            # Rejection at previous day low?
            if current_low <= prev_low * 1.0002:  # Within 2 pips
                lower_wick = min(hourly_candles[-1]['mid']['o'], current) - current_low
                if lower_wick > body_size * 2:
                    return {
                        'setup': 'ny_absorption',
                        'pair': pair,
                        'direction': 'LONG',
                        'entry': current,
                        'stop': prev_low - (body_size * 1.5),
                        'target': prev_high,
                        'confidence': 0.76,
                        'reason': 'Rejection at previous day low',
                        'timestamp': datetime.utcnow().isoformat()
                    }

            return None

        except Exception as e:
            print(f"[ERROR] NY absorption detection: {e}")
            return None

    def detect_tokyo_gap_fill(self, pair):
        """
        Detect: Unfilled gaps from weekend/session open
        Theory: Gaps fill 70%+ of the time within 24h

        Returns: setup dict or None
        """
        try:
            # Get hourly candles
            candles = self.adapter.get_candles(pair, count=48, granularity='H1')
            if not candles or len(candles) < 10:
                return None

            current_price = candles[-1]['mid']['c']

            # Look for gaps (close to next open difference > 0.1%)
            for i in range(len(candles) - 10, len(candles) - 1):
                current_close = candles[i]['mid']['c']
                next_open = candles[i + 1]['mid']['o']

                gap = abs(next_open - current_close)
                gap_pct = gap / current_close

                if gap_pct > 0.001:  # Gap larger than 0.1%
                    gap_high = max(current_close, next_open)
                    gap_low = min(current_close, next_open)

                    # Is gap still unfilled?
                    filled = False
                    for j in range(i + 1, len(candles)):
                        if candles[j]['mid']['h'] >= gap_high and candles[j]['mid']['l'] <= gap_low:
                            filled = True
                            break

                    if not filled:
                        # Gap still open - trade toward fill
                        if current_price < gap_low:
                            return {
                                'setup': 'tokyo_gap_fill',
                                'pair': pair,
                                'direction': 'LONG',
                                'entry': current_price,
                                'stop': current_price - (gap * 0.5),
                                'target': gap_high,
                                'confidence': 0.71,
                                'reason': f'Unfilled gap from {i} candles ago',
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        elif current_price > gap_high:
                            return {
                                'setup': 'tokyo_gap_fill',
                                'pair': pair,
                                'direction': 'SHORT',
                                'entry': current_price,
                                'stop': current_price + (gap * 0.5),
                                'target': gap_low,
                                'confidence': 0.71,
                                'reason': f'Unfilled gap from {i} candles ago',
                                'timestamp': datetime.utcnow().isoformat()
                            }

            return None

        except Exception as e:
            print(f"[ERROR] Tokyo gap fill detection: {e}")
            return None

    def is_trading_session(self, session_name):
        """Check if we're in the specified trading session (EST)"""
        now = datetime.utcnow()
        current_time = now.time()

        sessions = {
            'london': (dt_time(3, 0), dt_time(5, 0)),    # 3-5 AM EST
            'ny': (dt_time(8, 0), dt_time(12, 0)),       # 8 AM - 12 PM EST
            'tokyo': (dt_time(19, 0), dt_time(2, 0))     # 7 PM - 2 AM EST (next day)
        }

        start, end = sessions.get(session_name, (dt_time(0, 0), dt_time(0, 0)))

        if start < end:
            return start <= current_time <= end
        else:  # Session crosses midnight
            return current_time >= start or current_time <= end

    def scan_for_setups(self):
        """Scan all pairs for all setup types"""
        pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        found_setups = []

        print(f"\n[SCAN] {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        for pair in pairs:
            try:
                # London Fake-out (3-5 AM EST)
                if self.is_trading_session('london'):
                    setup = self.detect_london_fakeout(pair)
                    if setup:
                        print(f"  [FOUND] London Fake-out: {pair} {setup['direction']}")
                        found_setups.append(setup)

                # NY Absorption (8 AM - 12 PM EST)
                if self.is_trading_session('ny'):
                    setup = self.detect_ny_absorption(pair)
                    if setup:
                        print(f"  [FOUND] NY Absorption: {pair} {setup['direction']}")
                        found_setups.append(setup)

                # Tokyo Gap Fill (7 PM - 2 AM EST)
                if self.is_trading_session('tokyo'):
                    setup = self.detect_tokyo_gap_fill(pair)
                    if setup:
                        print(f"  [FOUND] Tokyo Gap Fill: {pair} {setup['direction']}")
                        found_setups.append(setup)

            except Exception as e:
                print(f"  [ERROR] {pair}: {e}")

        if not found_setups:
            print("  No setups detected this scan")

        return found_setups

    def log_setup(self, setup):
        """Log detected setup to JSON file"""
        try:
            # Load existing log
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'setups': [], 'statistics': {}}

            # Add setup
            log_data['setups'].append(setup)

            # Update statistics
            setup_type = setup['setup']
            if setup_type not in log_data['statistics']:
                log_data['statistics'][setup_type] = {'count': 0, 'pending': 0}

            log_data['statistics'][setup_type]['count'] += 1
            log_data['statistics'][setup_type]['pending'] += 1

            # Save
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            print(f"[ERROR] Failed to log setup: {e}")

    def run(self, scan_interval=900):  # 15 minutes
        """Run continuous detection loop"""
        print(f"\n[START] Setup detector running")
        print(f"Scan interval: {scan_interval} seconds")
        print(f"Log file: {self.log_file}")
        print(f"\nWaiting for setups...\n")

        while True:
            try:
                setups = self.scan_for_setups()

                for setup in setups:
                    self.log_setup(setup)

                # Wait for next scan
                time.sleep(scan_interval)

            except KeyboardInterrupt:
                print("\n[STOP] Detector stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] Scan failed: {e}")
                time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    detector = SetupDetector()
    detector.run(scan_interval=900)  # Scan every 15 minutes
