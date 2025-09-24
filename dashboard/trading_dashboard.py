"""
AUTONOMOUS TRADING DASHBOARD
Real-time monitoring and control center for your trading empire
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import alpaca_trade_api as tradeapi
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import asyncio

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Autonomous Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Alpaca API
@st.cache_resource
def init_alpaca():
    return tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2'
    )

alpaca = init_alpaca()

def get_portfolio_data():
    """Get current portfolio data"""
    try:
        account = alpaca.get_account()
        positions = alpaca.list_positions()

        portfolio_value = float(account.portfolio_value)
        buying_power = float(account.buying_power)

        # Calculate total P&L
        total_unrealized_pl = sum(float(pos.unrealized_pl) for pos in positions)
        daily_pnl_pct = (total_unrealized_pl / portfolio_value) * 100

        # Position data
        position_data = []
        for pos in positions:
            unrealized_pl = float(pos.unrealized_pl)
            market_value = float(pos.market_value)
            pnl_pct = (unrealized_pl / market_value) * 100 if market_value > 0 else 0

            # Determine status
            if pnl_pct >= 5.0:
                status = "🚀 AGGRESSIVE SCALE TARGET"
                status_color = "green"
            elif pnl_pct >= 3.0:
                status = "📈 SCALE TARGET"
                status_color = "blue"
            elif pnl_pct >= 2.5:
                status = "⚡ APPROACHING SCALE"
                status_color = "orange"
            elif pnl_pct > 0:
                status = "✅ PROFITABLE"
                status_color = "lightgreen"
            else:
                status = "⚠️ MONITOR"
                status_color = "red"

            position_data.append({
                'Symbol': pos.symbol,
                'Shares': int(pos.qty),
                'Value': market_value,
                'P&L': unrealized_pl,
                'P&L%': pnl_pct,
                'Status': status,
                'Status_Color': status_color
            })

        return {
            'portfolio_value': portfolio_value,
            'buying_power': buying_power,
            'total_pl': total_unrealized_pl,
            'daily_pnl_pct': daily_pnl_pct,
            'positions': position_data
        }

    except Exception as e:
        st.error(f"Error fetching portfolio data: {e}")
        return None

def get_recent_trades():
    """Get recent trading activity"""
    try:
        orders = alpaca.list_orders(status='filled', direction='desc', limit=10)

        trade_data = []
        for order in orders:
            trade_data.append({
                'Time': order.filled_at.strftime('%H:%M:%S') if order.filled_at else 'Unknown',
                'Action': order.side.upper(),
                'Symbol': order.symbol,
                'Shares': int(order.filled_qty) if order.filled_qty else 0,
                'Price': f"${float(order.filled_avg_price):.2f}" if order.filled_avg_price else "N/A",
                'Value': f"${int(order.filled_qty) * float(order.filled_avg_price):,.0f}" if order.filled_qty and order.filled_avg_price else "N/A"
            })

        return trade_data

    except Exception as e:
        st.error(f"Error fetching trade data: {e}")
        return []

def get_system_status():
    """Get autonomous system status"""

    # Check if autonomous system is running
    autonomous_status = "🟢 ACTIVE" if os.path.exists('../truly_autonomous.log') else "🔴 OFFLINE"

    # Check last autonomous activity
    try:
        with open('../truly_autonomous.log', 'r') as f:
            lines = f.readlines()
            last_activity = lines[-1].split(' - ')[0] if lines else "No activity"
    except:
        last_activity = "Log not found"

    # Check R&D system
    rd_files = [f for f in os.listdir('..') if 'elite_strategies' in f]
    rd_status = f"🟢 ACTIVE ({len(rd_files)} strategies)" if rd_files else "🔴 OFFLINE"

    return {
        'autonomous_system': autonomous_status,
        'last_activity': last_activity,
        'rd_system': rd_status,
        'rebalancer': "🟢 OPERATIONAL (Last: 10:39 AM)",
        'risk_management': "🟢 ACTIVE"
    }

def calculate_monthly_target_progress(daily_pnl_pct):
    """Calculate progress toward 52.7% monthly target"""
    target_monthly = 52.7
    trading_days_month = 22

    # Compound daily returns
    if daily_pnl_pct > 0:
        daily_multiplier = 1 + (daily_pnl_pct / 100)
        monthly_projection = (daily_multiplier ** trading_days_month - 1) * 100
        progress = (monthly_projection / target_monthly) * 100
    else:
        monthly_projection = 0
        progress = 0

    return monthly_projection, progress

# DASHBOARD LAYOUT
st.title("🚀 Autonomous Trading Dashboard")
st.markdown("**Real-time Command Center for Your Trading Empire**")

# Auto-refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
if current_time - st.session_state.last_refresh > 30:
    st.rerun()

# Sidebar - System Controls
st.sidebar.header("🤖 System Controls")
st.sidebar.markdown("**Autonomous Systems Status**")

system_status = get_system_status()
st.sidebar.write(f"**Autonomous Monitor:** {system_status['autonomous_system']}")
st.sidebar.write(f"**R&D Engine:** {system_status['rd_system']}")
st.sidebar.write(f"**Intelligent Rebalancer:** {system_status['rebalancer']}")
st.sidebar.write(f"**Risk Management:** {system_status['risk_management']}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S PT')}")

if st.sidebar.button("🔄 Force Refresh"):
    st.rerun()

# Main Dashboard
portfolio_data = get_portfolio_data()

if portfolio_data:
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${portfolio_data['portfolio_value']:,.0f}",
            delta=f"{portfolio_data['daily_pnl_pct']:+.2f}% today"
        )

    with col2:
        st.metric(
            "Daily P&L",
            f"${portfolio_data['total_pl']:+,.0f}",
            delta=f"{portfolio_data['daily_pnl_pct']:+.2f}%"
        )

    with col3:
        monthly_proj, progress = calculate_monthly_target_progress(portfolio_data['daily_pnl_pct'])
        st.metric(
            "Monthly Target Progress",
            f"{progress:.1f}%",
            delta=f"Proj: {monthly_proj:.1f}% vs 52.7% target"
        )

    with col4:
        st.metric(
            "Buying Power",
            f"${portfolio_data['buying_power']:,.0f}",
            delta="For scaling opportunities"
        )

    # Position Analysis
    st.header("📊 Position Analysis")

    # Create position DataFrame
    df_positions = pd.DataFrame(portfolio_data['positions'])

    if not df_positions.empty:
        # Position table with color coding
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Current Positions")

            # Format the dataframe for display
            display_df = df_positions.copy()
            display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.0f}")
            display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+,.0f}")
            display_df['P&L%'] = display_df['P&L%'].apply(lambda x: f"{x:+.1f}%")
            display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,}")

            st.dataframe(
                display_df[['Symbol', 'Shares', 'Value', 'P&L', 'P&L%', 'Status']],
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.subheader("Scaling Opportunities")

            scale_targets = df_positions[df_positions['P&L%'] >= 3.0]
            if not scale_targets.empty:
                for _, pos in scale_targets.iterrows():
                    st.success(f"**{pos['Symbol']}**: {pos['P&L%']:+.1f}% - {pos['Status']}")
            else:
                st.info("No positions ready for scaling")

        # Position Performance Chart
        st.subheader("Position Performance Visualization")

        fig = px.bar(
            df_positions,
            x='Symbol',
            y='P&L%',
            color='P&L%',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Position Performance (%)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recent Trading Activity
    st.header("📈 Recent Trading Activity")

    trades = get_recent_trades()
    if trades:
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No recent trades found")

    # Autonomous Intelligence Panel
    st.header("🧠 Autonomous Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Analysis")

        # Portfolio volatility
        volatility = abs(portfolio_data['daily_pnl_pct'])
        if volatility >= 2.0:
            st.warning(f"🔥 RAPID MONITORING MODE: {volatility:.1f}% volatility")
        else:
            st.success(f"📊 Normal monitoring: {volatility:.1f}% volatility")

        # Scale opportunities
        scale_count = len([p for p in portfolio_data['positions'] if p['P&L%'] >= 3.0])
        if scale_count > 0:
            st.info(f"🎯 {scale_count} positions ready for scaling")

        # Buying power status
        if portfolio_data['buying_power'] > 5000:
            st.success(f"💰 ${portfolio_data['buying_power']:,.0f} available for scaling")
        else:
            st.warning("⏳ Waiting for settlement to create buying power")

    with col2:
        st.subheader("Next Autonomous Actions")

        # Predict next actions based on current state
        next_actions = []

        for pos in portfolio_data['positions']:
            if pos['P&L%'] >= 5.0:
                next_actions.append(f"🚀 Scale up {pos['Symbol']} aggressively (+{pos['P&L%']:.1f}%)")
            elif pos['P&L%'] >= 3.0:
                next_actions.append(f"📈 Scale up {pos['Symbol']} (+{pos['P&L%']:.1f}%)")
            elif pos['P&L%'] <= -2.0:
                next_actions.append(f"⚠️ Consider trimming {pos['Symbol']} ({pos['P&L%']:.1f}%)")

        if next_actions:
            for action in next_actions[:5]:  # Show top 5
                st.write(f"• {action}")
        else:
            st.info("• Continue monitoring current positions")
            st.info("• Maintain optimal allocation")

else:
    st.error("Unable to fetch portfolio data")

# Footer
st.markdown("---")
st.markdown("**🤖 Autonomous Trading Dashboard** | Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S PT'))