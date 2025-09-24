#!/usr/bin/env python3
"""
R&D SYSTEM MONITOR - Real-time System Monitoring
===============================================
Comprehensive monitoring dashboard for the running R&D system
"""

import json
import os
import time
from datetime import datetime, timedelta
import subprocess

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_system_metrics():
    """Get current system metrics"""
    
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rd_strategies': 0,
        'deployable_strategies': 0,
        'hive_strategies': 0,
        'active_strategies': 0,
        'total_allocation': 0.0,
        'rd_sessions': 0,
        'successful_sessions': 0,
        'last_session': 'None'
    }
    
    # R&D strategies
    if os.path.exists('validated_strategies.json'):
        try:
            with open('validated_strategies.json', 'r') as f:
                rd_data = json.load(f)
                metrics['rd_strategies'] = len(rd_data)
                metrics['deployable_strategies'] = len([s for s in rd_data if s.get('deployment_ready', False)])
        except:
            pass
    
    # Hive strategies
    if os.path.exists('hive_trading_strategies.json'):
        try:
            with open('hive_trading_strategies.json', 'r') as f:
                hive_data = json.load(f)
                strategies = hive_data.get('active_strategies', [])
                metrics['hive_strategies'] = len(strategies)
                metrics['active_strategies'] = len([s for s in strategies if s.get('active', False)])
                metrics['total_allocation'] = sum([s.get('allocation', 0) for s in strategies])
        except:
            pass
    
    # Session history
    if os.path.exists('rd_session_history.json'):
        try:
            with open('rd_session_history.json', 'r') as f:
                sessions = json.load(f)
                metrics['rd_sessions'] = len(sessions)
                metrics['successful_sessions'] = len([s for s in sessions if s.get('status') == 'completed'])
                
                if sessions:
                    last_session = max(sessions, key=lambda x: x.get('start_time', ''))
                    metrics['last_session'] = last_session.get('start_time', 'Unknown')
        except:
            pass
    
    return metrics

def check_process_running():
    """Check if R&D orchestrator is running"""
    try:
        # On Windows, use tasklist
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        
        # Count python processes (rough estimate)
        python_processes = len([line for line in result.stdout.split('\n') if 'python.exe' in line])
        return python_processes > 1  # Assume more than 1 means our script is running
        
    except:
        return False

def display_monitoring_dashboard():
    """Display real-time monitoring dashboard"""
    
    while True:
        clear_screen()
        
        print("=" * 70)
        print("🌌 HIVE TRADING R&D SYSTEM - LIVE MONITORING DASHBOARD")
        print("=" * 70)
        
        metrics = get_system_metrics()
        is_running = check_process_running()
        
        print(f"⏰ Current Time: {metrics['timestamp']}")
        print(f"🔄 System Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")
        
        # Detect market hours
        now = datetime.now()
        market_hours = 9.5 <= now.hour + now.minute/60 <= 16 and now.weekday() < 5
        session_type = "🏦 TRADING HOURS" if market_hours else "🌙 AFTER HOURS R&D"
        print(f"📊 Market Session: {session_type}")
        
        print("\n" + "=" * 70)
        print("📈 STRATEGY METRICS")
        print("=" * 70)
        
        print(f"🧠 R&D Strategies Generated: {metrics['rd_strategies']}")
        print(f"✅ Deployment Ready: {metrics['deployable_strategies']}")
        print(f"🏠 Hive Strategies: {metrics['hive_strategies']}")
        print(f"⚡ Active Strategies: {metrics['active_strategies']}")
        print(f"💰 Total Allocation: {metrics['total_allocation']:.1%}")
        
        print("\n" + "=" * 70)
        print("🔄 SESSION METRICS")
        print("=" * 70)
        
        print(f"📊 R&D Sessions Completed: {metrics['rd_sessions']}")
        print(f"✅ Successful Sessions: {metrics['successful_sessions']}")
        success_rate = (metrics['successful_sessions'] / metrics['rd_sessions'] * 100 
                       if metrics['rd_sessions'] > 0 else 0)
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️ Last Session: {metrics['last_session']}")
        
        print("\n" + "=" * 70)
        print("💡 SYSTEM HEALTH")
        print("=" * 70)
        
        # Health indicators
        health_score = 0
        total_checks = 4
        
        if os.path.exists('after_hours_rd_engine.py'):
            print("✅ R&D Engine: Ready")
            health_score += 1
        else:
            print("❌ R&D Engine: Missing")
        
        if os.path.exists('hive_rd_orchestrator.py'):
            print("✅ Orchestrator: Ready")
            health_score += 1
        else:
            print("❌ Orchestrator: Missing")
            
        if metrics['rd_strategies'] > 0:
            print("✅ Strategy Generation: Active")
            health_score += 1
        else:
            print("⚠️ Strategy Generation: No strategies yet")
        
        if is_running:
            print("✅ Process Status: Running")
            health_score += 1
        else:
            print("⚠️ Process Status: Not detected")
        
        overall_health = (health_score / total_checks) * 100
        health_status = ("🟢 EXCELLENT" if overall_health >= 75 else 
                        "🟡 GOOD" if overall_health >= 50 else 
                        "🔴 NEEDS ATTENTION")
        
        print(f"\n🎯 Overall Health: {health_status} ({overall_health:.0f}%)")
        
        print("\n" + "=" * 70)
        print("🎮 MONITORING COMMANDS")
        print("=" * 70)
        print("Press Ctrl+C to exit monitor")
        print("In another terminal, run:")
        print("  python hive_rd_orchestrator.py --mode status")
        print("  python quick_performance_check.py")
        
        # Update every 30 seconds
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            clear_screen()
            print("🛑 Monitoring stopped by user")
            print("✅ R&D System continues running in background")
            break

def show_log_tail():
    """Show recent log entries"""
    
    log_files = ['hive_rd_orchestrator.log', 'quantum_trading.log']
    
    print("\n📄 RECENT LOG ENTRIES:")
    print("=" * 50)
    
    for log_file in log_files:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            print(f"\n--- {log_file} ---")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Show last 5 lines
                    for line in lines[-5:]:
                        print(line.strip())
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        else:
            print(f"\n--- {log_file} ---")
            print("(No recent entries or file not found)")

def main():
    """Main monitoring interface"""
    
    print("HIVE TRADING R&D SYSTEM MONITOR")
    print("=" * 40)
    print("1. Live Dashboard (updates every 30s)")
    print("2. Quick Status Check")
    print("3. Show Recent Logs")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                display_monitoring_dashboard()
                break
            elif choice == '2':
                metrics = get_system_metrics()
                print(f"\n🎯 QUICK STATUS:")
                print(f"R&D Strategies: {metrics['rd_strategies']}")
                print(f"Active Strategies: {metrics['active_strategies']}")
                print(f"Sessions Completed: {metrics['rd_sessions']}")
                print(f"System Health: {'Good' if metrics['rd_strategies'] > 0 else 'Starting up'}")
            elif choice == '3':
                show_log_tail()
            elif choice == '4':
                print("Monitor exited. R&D system continues running.")
                break
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nMonitor exited.")
            break

if __name__ == "__main__":
    main()