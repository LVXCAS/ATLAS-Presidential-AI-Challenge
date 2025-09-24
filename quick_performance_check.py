#!/usr/bin/env python3
"""
QUICK PERFORMANCE CHECK - Simplified Analytics
==============================================
Simplified version of performance analytics that avoids JSON serialization issues
"""

import json
import os
from datetime import datetime

def quick_system_check():
    """Quick system performance check"""
    
    print("HIVE TRADING R&D SYSTEM - QUICK PERFORMANCE CHECK")
    print("=" * 55)
    print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check R&D strategies
    if os.path.exists('validated_strategies.json'):
        with open('validated_strategies.json', 'r') as f:
            rd_strategies = json.load(f)
        print(f"R&D Strategies Generated: {len(rd_strategies)}")
        
        # Count deployable
        deployable = len([s for s in rd_strategies if s.get('deployment_ready', False)])
        print(f"Deployment Ready: {deployable}")
    else:
        print("R&D Strategies Generated: 0")
        print("Deployment Ready: 0")
    
    # Check Hive integration
    if os.path.exists('hive_trading_strategies.json'):
        with open('hive_trading_strategies.json', 'r') as f:
            hive_data = json.load(f)
        
        hive_strategies = hive_data.get('active_strategies', [])
        active_count = len([s for s in hive_strategies if s.get('active', False)])
        total_allocation = sum([s.get('allocation', 0) for s in hive_strategies])
        
        print(f"Hive Strategies: {len(hive_strategies)}")
        print(f"Active Strategies: {active_count}")
        print(f"Total Allocation: {total_allocation:.1%}")
    else:
        print("Hive Strategies: 0")
        print("Active Strategies: 0")
        print("Total Allocation: 0.0%")
    
    # Check session history
    if os.path.exists('rd_session_history.json'):
        with open('rd_session_history.json', 'r') as f:
            sessions = json.load(f)
        
        successful_sessions = len([s for s in sessions if s.get('status') == 'completed'])
        total_generated = sum([s.get('strategies_generated', 0) for s in sessions])
        
        print(f"R&D Sessions Completed: {len(sessions)}")
        print(f"Successful Sessions: {successful_sessions}")
        print(f"Total Strategies Generated: {total_generated}")
    else:
        print("R&D Sessions Completed: 0")
        print("Successful Sessions: 0")
        print("Total Strategies Generated: 0")
    
    print()
    print("SYSTEM STATUS:")
    
    # System health indicators
    files_present = [
        'after_hours_rd_engine.py',
        'hive_rd_orchestrator.py', 
        'rd_strategy_integrator.py'
    ]
    
    files_ready = sum([1 for f in files_present if os.path.exists(f)])
    print(f"Core Files Ready: {files_ready}/{len(files_present)}")
    
    if files_ready == len(files_present):
        print("System Status: [OPERATIONAL]")
    else:
        print("System Status: [SETUP REQUIRED]")
    
    print()
    print("To check detailed status: python hive_rd_orchestrator.py --mode status")
    print("To run single R&D session: python hive_rd_orchestrator.py --mode single")

if __name__ == "__main__":
    quick_system_check()