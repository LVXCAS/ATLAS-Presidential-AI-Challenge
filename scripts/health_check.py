#!/usr/bin/env python3
"""
Health check script for Docker containers.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.database import check_database_health
from config.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


async def main():
    """Run health checks."""
    setup_logging(level="INFO")
    
    try:
        logger.info("Starting health checks...")
        
        # Check database health
        health_status = await check_database_health()
        
        if health_status["overall"]:
            logger.info("All health checks passed")
            sys.exit(0)
        else:
            logger.error(f"Health checks failed: {health_status}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())