#!/usr/bin/env python3
"""
Development environment setup script.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.database import init_database, check_database_health
from config.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    logger.info(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result


async def setup_database():
    """Set up database connections and run migrations."""
    logger.info("Setting up database...")
    
    try:
        # Initialize database connections
        await init_database()
        
        # Check health
        health = await check_database_health()
        if not health["overall"]:
            logger.error(f"Database health check failed: {health}")
            return False
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def setup_docker():
    """Set up Docker environment."""
    logger.info("Setting up Docker environment...")
    
    # Check if Docker is running
    result = run_command("docker --version", check=False)
    if result.returncode != 0:
        logger.error("Docker is not installed or not running")
        return False
    
    # Check if Docker Compose is available
    result = run_command("docker compose version", check=False)
    if result.returncode != 0:
        logger.error("Docker Compose is not available")
        return False
    
    # Build and start services
    logger.info("Building Docker images...")
    run_command("docker compose build")
    
    logger.info("Starting Docker services...")
    run_command("docker compose up -d postgres redis")
    
    # Wait for services to be ready
    logger.info("Waiting for services to be ready...")
    import time
    time.sleep(10)
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    # Check if Poetry is installed
    result = run_command("poetry --version", check=False)
    if result.returncode != 0:
        logger.error("Poetry is not installed. Please install Poetry first.")
        return False
    
    # Install dependencies
    run_command("poetry install")
    
    return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    logger.info("Setting up pre-commit hooks...")
    
    run_command("poetry run pre-commit install")
    
    return True


async def main():
    """Run the complete development setup."""
    setup_logging(level="INFO")
    
    logger.info("Starting development environment setup...")
    
    try:
        # Install dependencies
        if not install_dependencies():
            sys.exit(1)
        
        # Set up Docker
        if not setup_docker():
            sys.exit(1)
        
        # Set up database
        if not await setup_database():
            sys.exit(1)
        
        # Set up pre-commit
        if not setup_pre_commit():
            sys.exit(1)
        
        logger.info("Development environment setup completed successfully!")
        logger.info("You can now run: docker compose up -d")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())