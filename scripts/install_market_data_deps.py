#!/usr/bin/env python3
"""
Install Market Data Ingestor Dependencies

This script installs the required dependencies for the Market Data Ingestor Agent.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required packages for Market Data Ingestor
REQUIRED_PACKAGES = [
    "langgraph>=0.0.40",
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "alpaca-trade-api>=3.1.1",
    "polygon-api-client>=1.12.0",
    "asyncpg>=0.29.0",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
]

def install_package(package):
    """Install a single package using pip"""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages"""
    logger.info("🚀 Installing Market Data Ingestor dependencies...")
    
    failed_packages = []
    
    for package in REQUIRED_PACKAGES:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        logger.error(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            logger.error(f"   - {package}")
        sys.exit(1)
    else:
        logger.info("✅ All dependencies installed successfully!")
        logger.info("🎉 Market Data Ingestor Agent is ready to use!")

if __name__ == "__main__":
    main()