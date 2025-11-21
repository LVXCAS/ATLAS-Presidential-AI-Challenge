"""
KELLY CRITERION & BLACK-SCHOLES IMPLEMENTATION
Professional position sizing and options pricing
For your $100K accelerated path
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing

    Used by prop traders and hedge funds
    Maximizes long-term growth while managing risk
    """

    @staticmethod
    def calculate_kelly(win_rate, avg_win, avg_loss):
        """
        Calculate optimal position size using Kelly Criterion

        Formula: f* = (p*b - q) / b
        where:
            f* = fraction of capital to risk
            p = probability of win
            q = probability of loss (1-p)
            b = win/loss ratio

        Args:
            win_rate: Win rate (0-1), e.g. 0.65 for 65%
            avg_win: Average win size (e.g. $100)
            avg_loss: Average loss size (e.g. $50)

        Returns:
            Optimal fraction of capital to risk per trade
        """
        if avg_loss == 0:
            return 0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Win/loss ratio

        kelly_fraction = (p * b - q) / b

        # Never risk more than 50% (safety check)
        return max(0, min(kelly_fraction, 0.5))

    @staticmethod
    def half_kelly(win_rate, avg_win, avg_loss):
        """
        Half-Kelly (more conservative)
        Used by most professional traders
        Reduces variance while maintaining good growth
        """
        full_kelly = KellyCriterion.calculate_kelly(win_rate, avg_win, avg_loss)
        return full_kelly / 2

    @staticmethod
    def quarter_kelly(win_rate, avg_win, avg_loss):
        """
        Quarter-Kelly (very conservative)
        Used for prop firm challenges (avoid drawdowns)
        """
        full_kelly = KellyCriterion.calculate_kelly(win_rate, avg_win, avg_loss)
        return full_kelly / 4

    @staticmethod
    def from_trade_history(trades):
        """
        Calculate Kelly from actual trade history

        Args:
            trades: List of trade dicts with 'pnl' and 'win' keys

        Returns:
            dict with full_kelly, half_kelly, quarter_kelly
        """
        wins = [t['pnl'] for t in trades if t.get('win', False)]
        losses = [abs(t['pnl']) for t in trades if not t.get('win', False)]

        if not wins or not losses:
            return {'full_kelly': 0, 'half_kelly': 0, 'quarter_kelly': 0}

        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        full_kelly = KellyCriterion.calculate_kelly(win_rate, avg_win, avg_loss)

        return {
            'full_kelly': full_kelly,
            'half_kelly': full_kelly / 2,
            'quarter_kelly': full_kelly / 4,
            'recommended': full_kelly / 2,  # Half-Kelly is standard
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss
        }


class BlackScholes:
    """
    Black-Scholes Model for options pricing

    The Nobel Prize-winning formula used by:
    - Goldman Sachs
    - Citadel
    - Jane Street
    - Every professional options desk
    """

    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """
        Calculate call option price using Black-Scholes

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate (e.g. 0.05 for 5%)
            sigma: Volatility (annualized, e.g. 0.30 for 30%)

        Returns:
            Theoretical call option price
        """
        if T <= 0:
            return max(S - K, 0)  # Intrinsic value at expiration

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """
        Calculate put option price using Black-Scholes

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate (e.g. 0.05 for 5%)
            sigma: Volatility (annualized, e.g. 0.30 for 30%)

        Returns:
            Theoretical put option price
        """
        if T <= 0:
            return max(K - S, 0)  # Intrinsic value at expiration

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put

    @staticmethod
    def greeks(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks

        Returns:
            dict with delta, gamma, vega, theta, rho
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change

        # Theta
        if option_type == 'call':
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365  # Per day
        else:
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365  # Per day

        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # For 1% change
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call', max_iter=100):
        """
        Calculate implied volatility from market price
        Uses Newton-Raphson method

        Args:
            market_price: Observed option price in market
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Implied volatility (annualized)
        """
        sigma = 0.5  # Initial guess (50% volatility)

        for i in range(max_iter):
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)

            diff = market_price - price

            if abs(diff) < 0.01:  # Convergence threshold
                return sigma

            # Vega (derivative of price with respect to volatility)
            vega = S * norm.pdf(BlackScholes.d1(S, K, T, r, sigma)) * np.sqrt(T)

            if vega == 0:
                break

            sigma = sigma + diff / vega

            # Bounds check
            if sigma <= 0:
                sigma = 0.01
            elif sigma > 5:
                sigma = 5

        return sigma


def example_kelly():
    """Example: Calculate position sizing for your trading"""
    print("="*70)
    print("KELLY CRITERION - POSITION SIZING")
    print("="*70)

    # Your target stats from 100K_ACCELERATED_PATH.md
    win_rate = 0.65  # 65% win rate
    avg_win = 100  # $100 average win
    avg_loss = 50  # $50 average loss

    kelly = KellyCriterion()

    full = kelly.calculate_kelly(win_rate, avg_win, avg_loss)
    half = kelly.half_kelly(win_rate, avg_win, avg_loss)
    quarter = kelly.quarter_kelly(win_rate, avg_win, avg_loss)

    print(f"\nWin Rate: {win_rate*100}%")
    print(f"Avg Win: ${avg_win}")
    print(f"Avg Loss: ${avg_loss}")
    print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")
    print("\nKELLY FRACTIONS:")
    print(f"  Full Kelly: {full*100:.2f}% of capital per trade")
    print(f"  Half Kelly: {half*100:.2f}% (RECOMMENDED)")
    print(f"  Quarter Kelly: {quarter*100:.2f}% (for prop firms)")

    # Example with $100k account
    capital = 100000
    print(f"\nWITH $100K ACCOUNT:")
    print(f"  Full Kelly: ${capital * full:,.2f} per trade")
    print(f"  Half Kelly: ${capital * half:,.2f} per trade (RECOMMENDED)")
    print(f"  Quarter Kelly: ${capital * quarter:,.2f} per trade (prop firm)")

    print("="*70)


def example_black_scholes():
    """Example: Price options and calculate Greeks"""
    print("\n"+"="*70)
    print("BLACK-SCHOLES - OPTIONS PRICING")
    print("="*70)

    # Example: TSLA call option
    S = 250  # TSLA at $250
    K = 260  # $260 strike
    T = 30/365  # 30 days to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.50  # 50% implied volatility (TSLA is volatile)

    bs = BlackScholes()

    call = bs.call_price(S, K, T, r, sigma)
    put = bs.put_price(S, K, T, r, sigma)

    print(f"\nTSLA at ${S}")
    print(f"Strike: ${K}")
    print(f"Days to Expiration: {int(T*365)}")
    print(f"Implied Volatility: {sigma*100}%")
    print(f"\nTHEORETICAL PRICES:")
    print(f"  Call: ${call:.2f}")
    print(f"  Put: ${put:.2f}")

    # Calculate Greeks
    greeks_call = bs.greeks(S, K, T, r, sigma, 'call')
    greeks_put = bs.greeks(S, K, T, r, sigma, 'put')

    print(f"\nCALL GREEKS:")
    print(f"  Delta: {greeks_call['delta']:.4f} (stock moves $1 -> option moves ${greeks_call['delta']:.2f})")
    print(f"  Gamma: {greeks_call['gamma']:.4f}")
    print(f"  Vega: {greeks_call['vega']:.4f} (IV +1% -> price +${greeks_call['vega']:.2f})")
    print(f"  Theta: {greeks_call['theta']:.4f} (1 day -> price {greeks_call['theta']:.2f})")

    print(f"\nPUT GREEKS:")
    print(f"  Delta: {greeks_put['delta']:.4f}")
    print(f"  Gamma: {greeks_put['gamma']:.4f}")
    print(f"  Vega: {greeks_put['vega']:.4f}")
    print(f"  Theta: {greeks_put['theta']:.4f}")

    # Implied volatility example
    market_price = 8.50  # Option trading at $8.50
    iv = bs.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"\nIMPLIED VOLATILITY:")
    print(f"  Market Price: ${market_price}")
    print(f"  Implied Vol: {iv*100:.1f}%")

    print("="*70)


if __name__ == "__main__":
    example_kelly()
    example_black_scholes()
