"""
Download and analyze last week's market data for OPTIONS_BOT stock universe
Identifies best trading opportunities, volatility patterns, and performance metrics
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Stock universe from OPTIONS_BOT.py
STOCK_UNIVERSE = [
    # Mega cap (best liquidity)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Major ETFs (excellent options liquidity)
    'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLV', 'XLE', 'GLD', 'TLT', 'EEM', 'XOP',
    # Large cap leaders
    'JPM', 'BAC', 'WFC', 'GS', 'JNJ', 'UNH', 'PFE', 'MRK', 'CVX', 'XOM',
    # High-volume options stocks
    'NFLX', 'CRM', 'AMD', 'INTC', 'DIS', 'V', 'MA', 'COIN', 'UBER',
    # Tech & Growth
    'ORCL', 'ADBE', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'ABNB', 'SQ', 'PYPL',
    # Consumer & Retail
    'HD', 'WMT', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST',
    # Healthcare & Biotech
    'ABBV', 'LLY', 'TMO', 'DHR', 'GILD', 'AMGN',
    # Industrial & Aerospace
    'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX',
    # Energy
    'SLB', 'COP', 'OXY', 'HAL', 'DVN',
    # Semiconductors
    'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL'
]

def download_stock_data(symbol, start_date, end_date):
    """Download historical data for a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if df.empty:
            print(f"⚠️  No data for {symbol}")
            return None

        return df
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {str(e)}")
        return None

def calculate_metrics(symbol, df):
    """Calculate trading metrics for a stock"""
    if df is None or len(df) < 2:
        return None

    try:
        # Basic price metrics
        first_close = df['Close'].iloc[0]
        last_close = df['Close'].iloc[-1]
        week_return = ((last_close - first_close) / first_close) * 100

        high = df['High'].max()
        low = df['Low'].min()
        price_range = ((high - low) / first_close) * 100

        # Volatility (standard deviation of daily returns)
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100

        # Average daily range
        df['daily_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
        avg_daily_range = df['daily_range'].mean()

        # Volume analysis
        avg_volume = df['Volume'].mean()
        max_volume = df['Volume'].max()
        volume_surge = (max_volume / avg_volume) if avg_volume > 0 else 0

        # Trend strength (simple momentum)
        sma_5 = df['Close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else last_close
        trend_strength = ((last_close - sma_5) / sma_5) * 100 if not pd.isna(sma_5) else 0

        # Identify best single day move
        best_day_return = daily_returns.max() * 100
        worst_day_return = daily_returns.min() * 100

        # Intraday volatility (useful for options)
        intraday_moves = []
        for idx, row in df.iterrows():
            if row['Open'] > 0:
                intraday_move = ((row['High'] - row['Low']) / row['Open']) * 100
                intraday_moves.append(intraday_move)

        avg_intraday_move = np.mean(intraday_moves) if intraday_moves else 0
        max_intraday_move = np.max(intraday_moves) if intraday_moves else 0

        return {
            'symbol': symbol,
            'week_return_pct': round(week_return, 2),
            'volatility_pct': round(volatility, 2),
            'avg_daily_range_pct': round(avg_daily_range, 2),
            'price_range_pct': round(price_range, 2),
            'avg_volume': int(avg_volume),
            'volume_surge_ratio': round(volume_surge, 2),
            'trend_strength_pct': round(trend_strength, 2),
            'best_day_return_pct': round(best_day_return, 2),
            'worst_day_return_pct': round(worst_day_return, 2),
            'avg_intraday_move_pct': round(avg_intraday_move, 2),
            'max_intraday_move_pct': round(max_intraday_move, 2),
            'first_close': round(first_close, 2),
            'last_close': round(last_close, 2),
            'days_analyzed': len(df)
        }
    except Exception as e:
        print(f"❌ Error calculating metrics for {symbol}: {str(e)}")
        return None

def rank_trading_opportunities(results):
    """Rank stocks by trading opportunity potential"""
    df = pd.DataFrame(results)

    # Options trading opportunity score
    # High volatility + high volume + good price movement = good for options
    df['opportunity_score'] = (
        df['volatility_pct'] * 2.0 +  # Volatility is key for options
        df['avg_daily_range_pct'] * 1.5 +  # Daily movement important
        df['volume_surge_ratio'] * 0.5 +  # Volume surge indicates interest
        abs(df['week_return_pct']) * 0.5  # Trending stocks are better
    )

    # Call opportunity score (bullish)
    df['call_score'] = (
        df['week_return_pct'].apply(lambda x: max(0, x)) * 2.0 +  # Positive return
        df['trend_strength_pct'].apply(lambda x: max(0, x)) * 1.5 +  # Uptrend
        df['volatility_pct'] * 1.0 +
        df['avg_daily_range_pct'] * 1.0
    )

    # Put opportunity score (bearish)
    df['put_score'] = (
        df['week_return_pct'].apply(lambda x: max(0, -x)) * 2.0 +  # Negative return
        df['trend_strength_pct'].apply(lambda x: max(0, -x)) * 1.5 +  # Downtrend
        df['volatility_pct'] * 1.0 +
        df['avg_daily_range_pct'] * 1.0
    )

    return df

def generate_report(df, start_date, end_date):
    """Generate comprehensive analysis report"""

    print("\n" + "="*80)
    print(f"📊 MARKET ANALYSIS REPORT: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("="*80)

    print(f"\n📈 Total Stocks Analyzed: {len(df)}")
    print(f"   Average Weekly Return: {df['week_return_pct'].mean():.2f}%")
    print(f"   Average Volatility: {df['volatility_pct'].mean():.2f}%")
    print(f"   Average Daily Range: {df['avg_daily_range_pct'].mean():.2f}%")

    # Market sentiment
    bullish_count = len(df[df['week_return_pct'] > 0])
    bearish_count = len(df[df['week_return_pct'] < 0])
    print(f"\n📊 Market Sentiment:")
    print(f"   Bullish (up): {bullish_count} stocks ({bullish_count/len(df)*100:.1f}%)")
    print(f"   Bearish (down): {bearish_count} stocks ({bearish_count/len(df)*100:.1f}%)")

    # Top performers
    print("\n🚀 TOP 10 PERFORMERS (Best Weekly Return):")
    print("-" * 80)
    top_performers = df.nlargest(10, 'week_return_pct')
    for idx, row in top_performers.iterrows():
        print(f"   {row['symbol']:6s} +{row['week_return_pct']:6.2f}% | Vol: {row['volatility_pct']:5.2f}% | Range: {row['avg_daily_range_pct']:5.2f}%")

    # Worst performers
    print("\n📉 WORST 10 PERFORMERS (Biggest Declines):")
    print("-" * 80)
    worst_performers = df.nsmallest(10, 'week_return_pct')
    for idx, row in worst_performers.iterrows():
        print(f"   {row['symbol']:6s} {row['week_return_pct']:6.2f}% | Vol: {row['volatility_pct']:5.2f}% | Range: {row['avg_daily_range_pct']:5.2f}%")

    # Most volatile (best for options)
    print("\n🔥 TOP 10 MOST VOLATILE (Best for Options Trading):")
    print("-" * 80)
    most_volatile = df.nlargest(10, 'volatility_pct')
    for idx, row in most_volatile.iterrows():
        print(f"   {row['symbol']:6s} Vol: {row['volatility_pct']:5.2f}% | Weekly: {row['week_return_pct']:+6.2f}% | Avg Range: {row['avg_daily_range_pct']:5.2f}%")

    # Best overall trading opportunities
    print("\n⭐ TOP 15 TRADING OPPORTUNITIES (Opportunity Score):")
    print("-" * 80)
    print(f"   {'Symbol':<6s} {'Score':<8s} {'Return':<9s} {'Vol':<8s} {'Range':<8s} {'Trend':<8s}")
    print("-" * 80)
    top_opportunities = df.nlargest(15, 'opportunity_score')
    for idx, row in top_opportunities.iterrows():
        print(f"   {row['symbol']:<6s} {row['opportunity_score']:6.1f}   {row['week_return_pct']:+6.2f}%  {row['volatility_pct']:5.2f}%  {row['avg_daily_range_pct']:5.2f}%  {row['trend_strength_pct']:+6.2f}%")

    # Best call opportunities
    print("\n📞 TOP 10 CALL OPPORTUNITIES (Bullish Plays):")
    print("-" * 80)
    top_calls = df.nlargest(10, 'call_score')
    for idx, row in top_calls.iterrows():
        print(f"   {row['symbol']:6s} Score: {row['call_score']:6.1f} | Return: +{row['week_return_pct']:5.2f}% | Trend: +{row['trend_strength_pct']:5.2f}%")

    # Best put opportunities
    print("\n📞 TOP 10 PUT OPPORTUNITIES (Bearish Plays):")
    print("-" * 80)
    top_puts = df.nlargest(10, 'put_score')
    for idx, row in top_puts.iterrows():
        print(f"   {row['symbol']:6s} Score: {row['put_score']:6.1f} | Return: {row['week_return_pct']:6.2f}% | Trend: {row['trend_strength_pct']:6.2f}%")

    # Volume leaders
    print("\n📊 TOP 10 VOLUME LEADERS:")
    print("-" * 80)
    volume_leaders = df.nlargest(10, 'avg_volume')
    for idx, row in volume_leaders.iterrows():
        print(f"   {row['symbol']:6s} Avg Vol: {row['avg_volume']:>15,} | Surge: {row['volume_surge_ratio']:.2f}x")

    # Biggest single-day movers
    print("\n🎯 TOP 10 BIGGEST SINGLE-DAY GAINS:")
    print("-" * 80)
    best_days = df.nlargest(10, 'best_day_return_pct')
    for idx, row in best_days.iterrows():
        print(f"   {row['symbol']:6s} Best Day: +{row['best_day_return_pct']:5.2f}% | Max Intraday Move: {row['max_intraday_move_pct']:5.2f}%")

    # Key insights
    print("\n💡 KEY INSIGHTS FOR OPTIONS BOT:")
    print("-" * 80)

    high_vol_stocks = df[df['volatility_pct'] > df['volatility_pct'].quantile(0.75)]
    print(f"   • {len(high_vol_stocks)} stocks in high volatility regime (top 25%)")

    strong_trends = df[abs(df['trend_strength_pct']) > 2.0]
    print(f"   • {len(strong_trends)} stocks with strong trends (±2%+ from 5-day MA)")

    high_volume = df[df['volume_surge_ratio'] > 1.5]
    print(f"   • {len(high_volume)} stocks with volume surges (>1.5x average)")

    avg_best_day = df['best_day_return_pct'].mean()
    avg_worst_day = df['worst_day_return_pct'].mean()
    print(f"   • Average best single day: +{avg_best_day:.2f}%")
    print(f"   • Average worst single day: {avg_worst_day:.2f}%")
    print(f"   • Average intraday volatility: {df['avg_intraday_move_pct'].mean():.2f}%")

    # Strategy recommendations
    print("\n🎲 STRATEGY RECOMMENDATIONS:")
    print("-" * 80)

    if df['week_return_pct'].mean() > 1.0:
        print("   ✓ BULLISH MARKET - Favor call options and bullish spreads")
    elif df['week_return_pct'].mean() < -1.0:
        print("   ✓ BEARISH MARKET - Favor put options and bearish spreads")
    else:
        print("   ✓ NEUTRAL MARKET - Consider straddles/strangles on high IV stocks")

    if df['volatility_pct'].mean() > 3.0:
        print("   ✓ HIGH VOLATILITY - Options premiums elevated, consider selling strategies")
    else:
        print("   ✓ NORMAL VOLATILITY - Good environment for directional plays")

    if df['avg_daily_range_pct'].mean() > 2.5:
        print("   ✓ WIDE DAILY RANGES - Use wider stops, higher profit targets")

    print("\n" + "="*80)

def main():
    """Main execution function"""
    print("🚀 Starting market data download and analysis...")
    print(f"   Analyzing {len(STOCK_UNIVERSE)} stocks from OPTIONS_BOT universe\n")

    # Calculate date range (last week = last 7 calendar days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Total stocks to analyze: {len(STOCK_UNIVERSE)}\n")

    results = []
    failed_symbols = []

    # Download and analyze each stock
    for i, symbol in enumerate(STOCK_UNIVERSE, 1):
        print(f"   [{i}/{len(STOCK_UNIVERSE)}] Processing {symbol}...", end=' ')

        df = download_stock_data(symbol, start_date, end_date)

        if df is not None and len(df) > 0:
            metrics = calculate_metrics(symbol, df)
            if metrics:
                results.append(metrics)
                print(f"✓ ({len(df)} days)")
            else:
                print(f"❌ Failed to calculate metrics")
                failed_symbols.append(symbol)
        else:
            print(f"❌ No data")
            failed_symbols.append(symbol)

    print(f"\n✅ Downloaded data for {len(results)}/{len(STOCK_UNIVERSE)} stocks")

    if failed_symbols:
        print(f"⚠️  Failed symbols: {', '.join(failed_symbols)}")

    # Rank and analyze
    if results:
        df = rank_trading_opportunities(results)

        # Generate comprehensive report
        generate_report(df, start_date, end_date)

        # Save to files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'market_analysis_{timestamp}.csv'
        json_file = f'market_analysis_{timestamp}.json'

        df.to_csv(csv_file, index=False)
        print(f"\n💾 Detailed data saved to: {csv_file}")

        results_dict = {
            'analysis_date': datetime.now().isoformat(),
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'total_stocks': len(results),
            'failed_symbols': failed_symbols,
            'market_summary': {
                'avg_weekly_return': float(df['week_return_pct'].mean()),
                'avg_volatility': float(df['volatility_pct'].mean()),
                'avg_daily_range': float(df['avg_daily_range_pct'].mean()),
                'bullish_count': int(len(df[df['week_return_pct'] > 0])),
                'bearish_count': int(len(df[df['week_return_pct'] < 0]))
            },
            'top_opportunities': df.nlargest(15, 'opportunity_score').to_dict('records'),
            'top_calls': df.nlargest(10, 'call_score').to_dict('records'),
            'top_puts': df.nlargest(10, 'put_score').to_dict('records'),
            'all_stocks': df.to_dict('records')
        }

        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"💾 JSON report saved to: {json_file}")

        print("\n✅ Analysis complete!")

    else:
        print("\n❌ No data to analyze!")

if __name__ == "__main__":
    main()
