#!/usr/bin/env python3
"""
Visualization Startup Script

This script provides a simple interface to start different visualization components
of the trading system.
"""

import sys
import os
import argparse
import subprocess
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def start_tkinter_dashboard():
    """Start the Tkinter unified dashboard"""
    print("Starting Tkinter Unified Dashboard...")
    try:
        from agents.unified_dashboard import UnifiedTradingDashboard
        dashboard = UnifiedTradingDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Error starting Tkinter dashboard: {e}")

def start_performance_dashboard():
    """Start the performance monitoring dashboard"""
    print("Starting Performance Monitoring Dashboard...")
    try:
        from agents.performance_dashboard import PerformanceDashboard
        dashboard = PerformanceDashboard()
        dashboard.start_dashboard()
    except Exception as e:
        print(f"Error starting performance dashboard: {e}")

def start_web_dashboard():
    """Start the web-based dashboard"""
    print("Starting Web Dashboard...")
    print("Open your browser to http://localhost:5000")
    try:
        # Start Flask app
        subprocess.Popen([sys.executable, "agents/web_dashboard.py"])
        print("Web dashboard started successfully!")
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping web dashboard...")
    except Exception as e:
        print(f"Error starting web dashboard: {e}")

def run_visualization_demo():
    """Run the visualization demo"""
    print("Running Visualization Demo...")
    try:
        subprocess.run([sys.executable, "examples/visualization_demo.py"])
    except Exception as e:
        print(f"Error running visualization demo: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Trading System Visualization Components")
    parser.add_argument("component", nargs="?", default="help",
                       choices=["tkinter", "performance", "web", "demo", "help"],
                       help="Component to start: tkinter, performance, web, demo, or help")
    
    args = parser.parse_args()
    
    if args.component == "help" or args.component is None:
        print("""
Trading System Visualization Components

Available components:
  tkinter     - Start the Tkinter unified dashboard
  performance - Start the performance monitoring dashboard
  web        - Start the web-based dashboard
  demo       - Run the visualization demo
  help       - Show this help message

Usage:
  python start_visualization.py <component>
  
Examples:
  python start_visualization.py tkinter
  python start_visualization.py web
  python start_visualization.py demo
        """)
        return
    
    if args.component == "tkinter":
        start_tkinter_dashboard()
    elif args.component == "performance":
        start_performance_dashboard()
    elif args.component == "web":
        start_web_dashboard()
    elif args.component == "demo":
        run_visualization_demo()

if __name__ == "__main__":
    main()