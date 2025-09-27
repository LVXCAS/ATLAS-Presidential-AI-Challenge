#!/usr/bin/env python3
"""
OPTIONS GREEKS MONITOR - FRIDAY EXPIRATION MANAGEMENT
Real-time monitoring of options Greeks for positions expiring this Friday
"""

import asyncio
import alpaca_trade_api as tradeapi
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import math

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - GREEKS MONITOR - %(message)s')

class OptionsGreeksMonitor:
    """Monitor options Greeks and manage Friday expiration risk"""

    def __init__(self):
        self.alpaca = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )

        # Current options positions (from previous analysis)
        self.options_positions = [
            'INTC250926C00032000',  # INTC $32 calls expiring Friday
            'INTC250926P00029000',  # INTC $29 puts expiring Friday
            'LYFT250926P00021000',  # LYFT $21 puts expiring Friday
            'RIVN250926P00014000',  # RIVN $14 puts expiring Friday
            'SNAP250926C00009000',  # SNAP $9 calls expiring Friday
            'SNAP250926P00008000'   # SNAP $8 puts expiring Friday
        ]

        self.risk_thresholds = {
            'high_theta_decay': -0.05,      # High theta decay warning
            'low_delta': 0.15,              # Low delta = out of money
            'high_gamma_risk': 0.20,        # High gamma = volatile
            'days_to_expiry_alert': 1,      # Alert when <1 day to expiry
            'max_loss_threshold': -50.0     # Close if >50% loss
        }

        logging.info("OPTIONS GREEKS MONITOR INITIALIZED")
        logging.info(f"Monitoring {len(self.options_positions)} options positions")
        logging.info("FRIDAY EXPIRATION MANAGEMENT ACTIVE")

    def parse_option_symbol(self, symbol):
        """Parse option symbol to extract details"""
        try:
            # Format: INTC250926C00032000 = INTC Sep26 2025 Call $32.00
            underlying = symbol[:4]
            date_part = symbol[4:10]  # 250926
            option_type = symbol[10]  # C or P
            strike_part = symbol[11:]  # 00032000

            # Parse date
            year = 2000 + int(date_part[:2])
            month = int(date_part[2:4])
            day = int(date_part[4:6])
            expiry_date = datetime(year, month, day)

            # Parse strike
            strike_price = float(strike_part) / 1000

            return {
                'underlying': underlying,
                'expiry_date': expiry_date,
                'option_type': 'CALL' if option_type == 'C' else 'PUT',
                'strike_price': strike_price,
                'days_to_expiry': (expiry_date - datetime.now()).days
            }

        except Exception as e:
            logging.error(f"Error parsing option symbol {symbol}: {e}")
            return None

    def calculate_black_scholes_greeks(self, S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes Greeks (simplified)"""
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

            d1 = (math.log(S/K) + (r + sigma**2/2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            from scipy.stats import norm
            N = norm.cdf
            n = norm.pdf

            if option_type == 'CALL':
                delta = N(d1)
                theta = -(S * n(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2)
            else:  # PUT
                delta = N(d1) - 1
                theta = -(S * n(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * N(-d2)

            gamma = n(d1) / (S * sigma * math.sqrt(T))
            vega = S * n(d1) * math.sqrt(T) / 100

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Per day
                'vega': vega
            }

        except Exception as e:
            logging.error(f"Greeks calculation error: {e}")
            # Fallback to estimated Greeks
            return {
                'delta': 0.5 if option_type == 'CALL' else -0.5,
                'gamma': 0.1,
                'theta': -0.02,
                'vega': 0.1
            }

    async def get_options_analysis(self):
        """Analyze all options positions"""

        options_analysis = []
        total_theta_decay = 0

        print(f"\n=== OPTIONS GREEKS ANALYSIS - {datetime.now().strftime('%H:%M:%S PST')} ===")
        print("FRIDAY EXPIRATION RISK MONITORING")
        print("-" * 60)

        for option_symbol in self.options_positions:
            try:
                # Parse option details
                option_info = self.parse_option_symbol(option_symbol)
                if not option_info:
                    continue

                # Get current position
                positions = self.alpaca.list_positions()
                position = next((p for p in positions if p.symbol == option_symbol), None)

                if not position:
                    print(f"CLOSED: {option_symbol} - Position no longer held")
                    continue

                # Get underlying stock price
                underlying_quote = self.alpaca.get_latest_quote(option_info['underlying'])
                underlying_price = float(underlying_quote.bid_price)

                # Get option current price
                option_quote = self.alpaca.get_latest_quote(option_symbol)
                option_price = float(option_quote.bid_price) if option_quote.bid_price else 0.01

                # Calculate Greeks (simplified)
                time_to_expiry = max(0.001, option_info['days_to_expiry'] / 365)
                greeks = self.calculate_black_scholes_greeks(
                    S=underlying_price,
                    K=option_info['strike_price'],
                    T=time_to_expiry,
                    r=0.05,  # Risk-free rate
                    sigma=0.30,  # Implied volatility estimate
                    option_type=option_info['option_type']
                )

                # Calculate P&L
                position_qty = float(position.qty)
                position_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                pnl_pct = (unrealized_pl / abs(position_value - unrealized_pl)) * 100 if position_value != unrealized_pl else 0

                # Risk analysis
                risk_level = self.assess_option_risk(option_info, greeks, pnl_pct)

                analysis = {
                    'symbol': option_symbol,
                    'underlying': option_info['underlying'],
                    'underlying_price': underlying_price,
                    'strike_price': option_info['strike_price'],
                    'option_type': option_info['option_type'],
                    'days_to_expiry': option_info['days_to_expiry'],
                    'position_qty': position_qty,
                    'current_price': option_price,
                    'market_value': position_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': pnl_pct,
                    'greeks': greeks,
                    'risk_level': risk_level,
                    'moneyness': self.calculate_moneyness(underlying_price, option_info['strike_price'], option_info['option_type'])
                }

                options_analysis.append(analysis)
                total_theta_decay += greeks['theta'] * abs(position_qty) * 100  # Per contract

                # Display analysis
                risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "💀"}
                moneyness_status = analysis['moneyness']

                print(f"{risk_emoji.get(risk_level, '❓')} {option_symbol}")
                print(f"   {option_info['underlying']} @ ${underlying_price:.2f} | Strike ${option_info['strike_price']:.2f} | {moneyness_status}")
                print(f"   P&L: {pnl_pct:+.1f}% (${unrealized_pl:,.0f}) | Days: {option_info['days_to_expiry']}")
                print(f"   Delta: {greeks['delta']:.3f} | Theta: ${greeks['theta']*100:.2f}/day | Gamma: {greeks['gamma']:.3f}")
                print(f"   Risk Level: {risk_level}")
                print()

            except Exception as e:
                logging.error(f"Analysis error for {option_symbol}: {e}")

        print(f"TOTAL DAILY THETA DECAY: ${total_theta_decay:.2f}")
        print("=" * 60)

        return options_analysis

    def calculate_moneyness(self, underlying_price, strike_price, option_type):
        """Calculate option moneyness"""
        if option_type == 'CALL':
            if underlying_price > strike_price * 1.05:
                return "ITM (In The Money)"
            elif underlying_price > strike_price * 0.95:
                return "ATM (At The Money)"
            else:
                return "OTM (Out The Money)"
        else:  # PUT
            if underlying_price < strike_price * 0.95:
                return "ITM (In The Money)"
            elif underlying_price < strike_price * 1.05:
                return "ATM (At The Money)"
            else:
                return "OTM (Out The Money)"

    def assess_option_risk(self, option_info, greeks, pnl_pct):
        """Assess option risk level"""
        risk_factors = []

        # Days to expiry risk
        if option_info['days_to_expiry'] <= 0:
            return "CRITICAL"
        elif option_info['days_to_expiry'] <= 1:
            risk_factors.append("EXPIRES_FRIDAY")

        # P&L risk
        if pnl_pct <= -75:
            risk_factors.append("MAJOR_LOSS")
        elif pnl_pct <= -50:
            risk_factors.append("HIGH_LOSS")

        # Greeks risk
        if greeks['theta'] < self.risk_thresholds['high_theta_decay']:
            risk_factors.append("HIGH_THETA")

        if abs(greeks['delta']) < self.risk_thresholds['low_delta']:
            risk_factors.append("LOW_DELTA")

        # Determine overall risk
        if "MAJOR_LOSS" in risk_factors or len(risk_factors) >= 3:
            return "CRITICAL"
        elif "HIGH_LOSS" in risk_factors or len(risk_factors) >= 2:
            return "HIGH"
        elif len(risk_factors) == 1:
            return "MEDIUM"
        else:
            return "LOW"

    async def generate_expiration_alerts(self, analysis):
        """Generate alerts for Friday expiration"""
        alerts = []

        for option in analysis:
            if option['days_to_expiry'] <= 1:
                if option['risk_level'] in ['HIGH', 'CRITICAL']:
                    alerts.append({
                        'symbol': option['symbol'],
                        'alert_type': 'CLOSE_BEFORE_EXPIRY',
                        'reason': f"High risk {option['pnl_pct']:.1f}% loss, expires Friday",
                        'recommendation': 'CLOSE POSITION',
                        'urgency': 'HIGH'
                    })

                elif option['pnl_pct'] > 20:
                    alerts.append({
                        'symbol': option['symbol'],
                        'alert_type': 'TAKE_PROFITS',
                        'reason': f"Profitable {option['pnl_pct']:.1f}% gain, expires Friday",
                        'recommendation': 'CONSIDER CLOSING',
                        'urgency': 'MEDIUM'
                    })

        return alerts

    async def run_greeks_monitoring(self):
        """Run comprehensive options Greeks monitoring"""

        print("OPTIONS GREEKS MONITOR - FRIDAY EXPIRATION MANAGEMENT")
        print("=" * 60)

        # Analyze all options
        analysis = await self.get_options_analysis()

        if not analysis:
            print("No options positions found")
            return

        # Generate alerts
        alerts = await self.generate_expiration_alerts(analysis)

        if alerts:
            print("🚨 EXPIRATION ALERTS:")
            print("-" * 30)
            for alert in alerts:
                print(f"⚠️  {alert['symbol']}: {alert['recommendation']}")
                print(f"   Reason: {alert['reason']}")
                print(f"   Urgency: {alert['urgency']}")
                print()

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'options_analysis': analysis,
            'alerts': alerts
        }

        with open(f'greeks_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    async def continuous_greeks_monitoring(self):
        """Run continuous Greeks monitoring"""

        logging.info("Starting continuous options Greeks monitoring")

        while True:
            try:
                await self.run_greeks_monitoring()
                await asyncio.sleep(180)  # Update every 3 minutes

            except Exception as e:
                logging.error(f"Greeks monitoring error: {e}")
                await asyncio.sleep(60)

async def main():
    """Run options Greeks monitor"""
    monitor = OptionsGreeksMonitor()
    await monitor.continuous_greeks_monitoring()

if __name__ == "__main__":
    asyncio.run(main())