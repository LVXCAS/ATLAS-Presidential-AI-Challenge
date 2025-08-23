#!/bin/bash
set -e

# Backup service health check
LOG_FILE="/backups/logs/health.log"

# Check if backup directories exist
if [ ! -d "/backups/db" ] || [ ! -d "/backups/data" ]; then
    echo "Backup directories are missing"
    exit 1
fi

# Check if cron is running
if ! pgrep crond > /dev/null; then
    echo "Cron daemon is not running"
    exit 1
fi

# Check if backup scripts are executable
if [ ! -x "/app/scripts/database-backup.sh" ] || [ ! -x "/app/scripts/data-backup.sh" ]; then
    echo "Backup scripts are not executable"
    exit 1
fi

# Check if recent backup exists (within last 25 hours)
RECENT_DB_BACKUP=$(find /backups/db -name "hive_trading_*.sql.gz" -mtime -1 | head -1)
if [ -z "$RECENT_DB_BACKUP" ]; then
    echo "No recent database backup found"
    exit 1
fi

# Log health check
echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup service health check passed" >> "$LOG_FILE"

echo "Backup service is healthy"
exit 0