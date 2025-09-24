#!/usr/bin/env python3
"""
R&D SYSTEM STATUS CHECKER
========================
Simple status checker for the R&D system
"""

import json
import os
from datetime import datetime

def check_rd_status():
    """Check R&D system status"""
    
    print("🌌 HIVE TRADING R&D SYSTEM STATUS CHECK")
    print("=" * 50)
    print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check core files
    core_files = [
        'after_hours_rd_engine.py',
        'hive_rd_orchestrator.py', 
        'rd_strategy_integrator.py',
        'rd_advanced_config.py'
    ]
    
    files_present = sum([1 for f in core_files if os.path.exists(f)])
    print(f"\n📂 Core Files: {files_present}/{len(core_files)} present")
    
    # Check strategy data
    strategies_generated = 0
    deployable = 0
    if os.path.exists('validated_strategies.json'):
        try:
            with open('validated_strategies.json', 'r') as f:
                strategies = json.load(f)
                strategies_generated = len(strategies)
                deployable = len([s for s in strategies if s.get('deployment_ready', False)])
        except:
            pass
    
    print(f"🧠 R&D Strategies Generated: {strategies_generated}")
    print(f"✅ Ready for Deployment: {deployable}")
    
    # Check Hive integration
    hive_strategies = 0
    active_strategies = 0
    total_allocation = 0.0
    
    if os.path.exists('hive_trading_strategies.json'):
        try:
            with open('hive_trading_strategies.json', 'r') as f:
                hive_data = json.load(f)
                strategies = hive_data.get('active_strategies', [])
                hive_strategies = len(strategies)
                active_strategies = len([s for s in strategies if s.get('active', False)])
                total_allocation = sum([s.get('allocation', 0) for s in strategies])
        except:
            pass
    
    print(f"🏠 Hive Strategies: {hive_strategies}")
    print(f"⚡ Active Strategies: {active_strategies}")
    print(f"💰 Total Allocation: {total_allocation:.1%}")
    
    # Check sessions
    sessions = 0
    successful = 0
    if os.path.exists('rd_session_history.json'):
        try:
            with open('rd_session_history.json', 'r') as f:
                session_data = json.load(f)
                sessions = len(session_data)
                successful = len([s for s in session_data if s.get('status') == 'completed'])
        except:
            pass
    
    print(f"🔄 R&D Sessions: {sessions}")
    print(f"✅ Successful: {successful}")
    
    # Overall status
    print(f"\n🎯 SYSTEM STATUS:")
    if files_present == len(core_files):
        if strategies_generated > 0:
            print("🟢 OPERATIONAL - Generating strategies")
        else:
            print("🟡 READY - Awaiting first R&D session")
    else:
        print("🔴 INCOMPLETE - Missing core files")
    
    # Market session
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    is_market_hours = 9.5 <= now.hour + now.minute/60 <= 16
    
    if is_weekend:
        session = "Weekend R&D Mode"
    elif is_market_hours:
        session = "Market Hours (System in standby)"
    else:
        session = "After Hours R&D Mode"
    
    print(f"📊 Current Session: {session}")
    
    print(f"\n💡 NEXT STEPS:")
    if strategies_generated == 0:
        print("- Wait for first R&D session to complete")
        print("- R&D sessions run every 4 hours when market closed")
    elif active_strategies == 0 and deployable > 0:
        print("- Consider deploying validated strategies")
        print("- Check deployment recommendations")
    else:
        print("- System operating normally")
        print("- Monitor performance and adjust as needed")

if __name__ == "__main__":
    check_rd_status()