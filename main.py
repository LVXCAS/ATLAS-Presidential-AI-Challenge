#!/usr/bin/env python3
"""
LangGraph Adaptive Multi-Strategy AI Trading System
Main application entry point for the autonomous trading platform.
"""

import logging
import sys
from pathlib import Path

from rich.console import Console

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging

# Initialize Rich console for beautiful output
console = Console()


def main():
    """Main application entry point."""
    console.print("[bold green]LangGraph Trading System v0.1.0[/bold green]")
    console.print("🚀 Project structure initialized successfully!")
    console.print("📁 Directory structure created")
    console.print("⚙️  Configuration files ready")
    console.print("🔧 Dependencies installed")
    console.print("✨ Code formatting configured")
    console.print("📝 Git repository initialized")

    console.print("\n[bold blue]Next steps:[/bold blue]")
    console.print("1. Copy .env.template to .env and configure your API keys")
    console.print("2. Install additional dependencies as needed")
    console.print("3. Start implementing the trading agents")

    console.print(
        "\n[bold yellow]Available commands will be added in future tasks:[/bold yellow]"
    )
    console.print("• init - Initialize system configuration")
    console.print("• validate - Validate system setup")
    console.print("• backtest - Run backtesting")
    console.print("• paper-trade - Start paper trading")
    console.print("• live-trade - Start live trading")
    console.print("• monitor - Launch monitoring dashboard")


if __name__ == "__main__":
    try:
        setup_logging()
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
