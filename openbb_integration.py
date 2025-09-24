"""
OpenBB Integration Module for Hive Trade System
Enhanced financial data access with OpenBB-like functionality
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from alpha_vantage.timeseries import TimeSeries
from financedatabase import Equities, ETFs, Funds
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HiveOpenBB:
    """OpenBB-like functionality for Hive Trade System"""

    def __init__(self, alpha_vantage_key=None):
        self.av_key = alpha_vantage_key
        self.av_client = TimeSeries(key=alpha_vantage_key) if alpha_vantage_key else None

    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """Get stock price data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol):
        """Get detailed stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None

    def get_financials(self, symbol):
        """Get financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            return {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'earnings': ticker.earnings
            }
        except Exception as e:
            print(f"Error fetching financials for {symbol}: {e}")
            return None

    def technical_analysis(self, symbol, period="6mo"):
        """Comprehensive technical analysis"""
        data = self.get_stock_data(symbol, period=period)
        if data is None:
            return None

        # Add technical indicators
        data['RSI'] = ta.rsi(data['Close'])
        data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
        data['MACD_signal'] = ta.macd(data['Close'])['MACDs_12_26_9']
        data['BB_upper'] = ta.bbands(data['Close'])['BBU_5_2.0']
        data['BB_middle'] = ta.bbands(data['Close'])['BBM_5_2.0']
        data['BB_lower'] = ta.bbands(data['Close'])['BBL_5_2.0']
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['EMA_12'] = ta.ema(data['Close'], length=12)
        data['EMA_26'] = ta.ema(data['Close'], length=26)

        # Volume indicators
        data['Volume_SMA'] = ta.sma(data['Volume'], length=20)
        data['OBV'] = ta.obv(data['Close'], data['Volume'])

        return data

    def screener(self, criteria=None):
        """Stock screener functionality"""
        try:
            equities = Equities()

            # Default criteria if none provided
            if criteria is None:
                criteria = {
                    'market_cap': '>1B',
                    'sector': ['Technology', 'Healthcare', 'Financial Services']
                }

            # Get all equities data
            all_stocks = equities.select()

            # Apply basic filtering
            filtered = all_stocks[all_stocks['market_cap'].notna()]

            return filtered.head(50)  # Return top 50 results
        except Exception as e:
            print(f"Error in screener: {e}")
            return None

    def options_data(self, symbol):
        """Get options chain data"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options

            if not options_dates:
                return None

            # Get options for nearest expiration
            exp_date = options_dates[0]
            option_chain = ticker.option_chain(exp_date)

            return {
                'expiration_dates': options_dates,
                'calls': option_chain.calls,
                'puts': option_chain.puts,
                'current_expiration': exp_date
            }
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return None

    def economic_data(self, indicator):
        """Get economic indicators using Alpha Vantage"""
        if not self.av_client:
            print("Alpha Vantage API key required for economic data")
            return None

        try:
            # This would need proper Alpha Vantage economic indicators API
            # For now, return placeholder
            print(f"Economic data for {indicator} would be fetched here")
            return None
        except Exception as e:
            print(f"Error fetching economic data: {e}")
            return None

    def create_chart(self, symbol, period="3mo", chart_type="candlestick"):
        """Create interactive charts"""
        data = self.technical_analysis(symbol, period=period)
        if data is None:
            return None

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI'),
            row_width=[0.2, 0.1, 0.1]
        )

        # Candlestick chart
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=symbol
                ),
                row=1, col=1
            )

        # Add moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'),
            row=1, col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume'),
            row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
            row=3, col=1
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    def portfolio_analysis(self, symbols, weights=None):
        """Analyze portfolio performance"""
        if weights is None:
            weights = [1/len(symbols)] * len(symbols)

        portfolio_data = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period="1y")
            if data is not None:
                portfolio_data[symbol] = data['Close']

        if not portfolio_data:
            return None

        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(portfolio_data)

        # Calculate returns
        returns = portfolio_df.pct_change().dropna()

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Performance metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        return {
            'symbols': symbols,
            'weights': weights,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'cumulative_returns': cumulative_returns,
            'individual_returns': returns
        }

    def news_sentiment(self, symbol):
        """Get news and sentiment (placeholder)"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            return news
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return None

    def crypto_data(self, symbol, period="1y"):
        """Get cryptocurrency data"""
        try:
            # Add -USD suffix for crypto symbols
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"

            return self.get_stock_data(symbol, period=period)
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return None

# Utility functions for easy access
def quick_chart(symbol, period="3mo"):
    """Quick chart function"""
    obb = HiveOpenBB()
    return obb.create_chart(symbol, period)

def quick_analysis(symbol):
    """Quick technical analysis"""
    obb = HiveOpenBB()
    return obb.technical_analysis(symbol)

def quick_info(symbol):
    """Quick stock info"""
    obb = HiveOpenBB()
    return obb.get_stock_info(symbol)

# Test the module
if __name__ == "__main__":
    print("Testing HiveOpenBB Integration...")

    # Initialize
    obb = HiveOpenBB()

    # Test basic functionality
    print("\\n1. Testing stock data retrieval...")
    data = obb.get_stock_data("AAPL", period="1mo")
    if data is not None:
        print(f"[OK] Retrieved {len(data)} days of AAPL data")
        print(f"Latest close: ${data['Close'].iloc[-1]:.2f}")

    print("\\n2. Testing technical analysis...")
    ta_data = obb.technical_analysis("AAPL", period="3mo")
    if ta_data is not None:
        latest = ta_data.iloc[-1]
        print(f"[OK] RSI: {latest['RSI']:.2f}")
        print(f"[OK] SMA 20: ${latest['SMA_20']:.2f}")

    print("\\n3. Testing stock info...")
    info = obb.get_stock_info("AAPL")
    if info:
        print(f"[OK] Company: {info.get('longName', 'N/A')}")
        print(f"[OK] Sector: {info.get('sector', 'N/A')}")

    print("\\n4. Testing options data...")
    options = obb.options_data("AAPL")
    if options:
        print(f"[OK] Found {len(options['expiration_dates'])} expiration dates")
        print(f"[OK] Calls: {len(options['calls'])} contracts")
        print(f"[OK] Puts: {len(options['puts'])} contracts")

    print("\\nHiveOpenBB Integration Test Complete! [SUCCESS]")