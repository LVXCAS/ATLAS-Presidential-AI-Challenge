@echo off
echo LAUNCHING AUTONOMOUS TRADING DASHBOARD
echo =======================================
echo Opening real-time command center for your trading empire
echo Dashboard will open in your browser automatically
echo =======================================

cd dashboard
streamlit run trading_dashboard.py --server.port 8501

pause