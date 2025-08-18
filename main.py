#!/usr/bin/env python3
"""
LangGraph Adaptive Multi-Strategy AI Trading System
Main application entry point for the autonomous trading platform.
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix for Windows Unicode encoding issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to detect if we should use mock mode
import socket

def check_port_open(host, port):
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

# Check if PostgreSQL is available
MOCK_MODE = not check_port_open('localhost', 5432)

if MOCK_MODE:
    from config.mock_database import init_database, check_database_health, close_database
else:
    from config.database import init_database, check_database_health, close_database
from config.logging_config import setup_logging, get_logger
from config.settings import settings
from config.secure_config import get_env_manager
from config.security import get_security_manager

# Initialize Rich console for beautiful output
console = Console()
logger = get_logger(__name__)


async def initialize_system():
    """Initialize the trading system."""
    logger.info("Initializing LangGraph Trading System...")
    
    try:
        # Initialize security system
        logger.info("Initializing security system...")
        security_manager = get_security_manager()
        security_manager.initialize_api_keys_from_env()
        
        # Initialize database connections
        await init_database()
        
        # Check system health
        health_status = await check_database_health()
        
        # Validate security configuration
        env_manager = get_env_manager()
        security_validation = env_manager.validate_configuration()
        
        if health_status["overall"]:
            logger.info("System initialization completed successfully")
            logger.info(f"Security validation: {sum(security_validation.values())}/{len(security_validation)} components valid")
            return True
        else:
            logger.error(f"System health check failed: {health_status}")
            return False
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


async def main():
    """Main application entry point."""
    console.print("[bold green]LangGraph Trading System v0.1.0[/bold green]")
    console.print("[ROBOT] Autonomous Multi-Strategy AI Trading Platform")
    
    # Load environment configuration
    env_file = ".env.development" if settings.debug else ".env.production"
    if Path(env_file).exists():
        console.print(f"[DOCUMENT] Loaded configuration from {env_file}")
    else:
        console.print(f"[WARNING] Configuration file {env_file} not found")
    
    # Initialize system
    console.print("[WRENCH] Initializing system...")
    
    if await initialize_system():
        console.print("[CHECKMARK] System initialized successfully!")
        
        # Display system status
        health_status = await check_database_health()
        env_manager = get_env_manager()
        security_validation = env_manager.validate_configuration()
        
        console.print(f"\n[bold blue]System Status:[/bold blue]")
        console.print(f"• Environment: {settings.app_name} ({env_manager.environment})")
        console.print(f"• Database: {'Mock PostgreSQL' if MOCK_MODE else 'PostgreSQL'} ({'Connected' if health_status['postgres'] else 'Disconnected'})")
        console.print(f"• Cache: {'Mock Redis' if MOCK_MODE else 'Redis'} ({'Connected' if health_status['redis'] else 'Disconnected'})")
        console.print(f"• Security: {sum(security_validation.values())}/{len(security_validation)} components configured")
        console.print(f"• Trading Mode: {'Paper Trading' if settings.trading.paper_trading else 'Live Trading'}")
        console.print(f"• Initial Capital: ${settings.trading.initial_capital:,.2f}")
        
        if MOCK_MODE:
            console.print(f"• [yellow]Running in MOCK MODE (no external dependencies)[/yellow]")
        
        console.print("\n[bold blue]Available Operations:[/bold blue]")
        console.print("• System is ready for agent implementation")
        console.print("• Database schema initialized")
        console.print("• Configuration management active")
        console.print("• Logging system operational")
        
        console.print("\n[bold yellow]Next Development Steps:[/bold yellow]")
        console.print("1. Implement Market Data Ingestor Agent")
        console.print("2. Implement News and Sentiment Analysis Agent")
        console.print("3. Implement Trading Strategy Agents")
        console.print("4. Implement Portfolio Allocator Agent")
        console.print("5. Implement Risk Manager Agent")
        
    else:
        console.print("[CROSS] System initialization failed!")
        console.print("Please check the logs and configuration")
        return False
    
    return True


async def cleanup():
    """Cleanup system resources."""
    logger.info("Shutting down system...")
    await close_database()
    logger.info("System shutdown completed")


if __name__ == "__main__":
    try:
        # Set up logging
        setup_logging(
            level=settings.log_level,
            log_file=settings.log_file,
            json_logs=not settings.debug
        )
        
        # Run main application
        success = asyncio.run(main())
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
        asyncio.run(cleanup())
        sys.exit(1)
    except Exception as e:
        console.print("[bold red]Application error occurred[/bold red]")
        logger.error(f"Application error: {e}", exc_info=True)
        asyncio.run(cleanup())
        sys.exit(1)
