#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Hive Trade Backend...${NC}"

# Function to check if service is available
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo -e "${GREEN}$service is ready!${NC}"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}Failed to connect to $service after $max_attempts attempts${NC}"
    exit 1
}

# Wait for dependencies
if [ "$ENVIRONMENT" = "production" ]; then
    wait_for_service "timescaledb" "5432" "TimescaleDB"
    wait_for_service "redis" "6379" "Redis"
fi

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
python -c "
import asyncio
from core.database import init_database

async def main():
    try:
        await init_database()
        print('Database migrations completed successfully')
    except Exception as e:
        print(f'Database migration failed: {e}')
        exit(1)

if __name__ == '__main__':
    asyncio.run(main())
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Database migration failed${NC}"
    exit 1
fi

# Initialize Redis cache
echo -e "${YELLOW}Initializing Redis cache...${NC}"
python -c "
import asyncio
from core.redis_manager import get_redis_manager

async def main():
    try:
        redis_manager = get_redis_manager()
        await redis_manager.initialize()
        health = await redis_manager.health_check()
        if health:
            print('Redis cache initialized successfully')
        else:
            raise Exception('Redis health check failed')
        await redis_manager.close()
    except Exception as e:
        print(f'Redis initialization failed: {e}')
        exit(1)

if __name__ == '__main__':
    asyncio.run(main())
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Redis initialization failed${NC}"
    exit 1
fi

# Load initial data if needed
if [ "$LOAD_SAMPLE_DATA" = "true" ]; then
    echo -e "${YELLOW}Loading sample data...${NC}"
    python scripts/load_sample_data.py
fi

# Set up log directory
mkdir -p /app/logs
touch /app/logs/app.log
touch /app/logs/error.log

echo -e "${GREEN}Backend initialization completed successfully!${NC}"
echo -e "${YELLOW}Starting application server...${NC}"

# Execute the main command
exec "$@"