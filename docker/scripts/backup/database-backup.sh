#!/bin/bash
set -e

# Database backup script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/backups/db"
LOG_FILE="/backups/logs/backup_${TIMESTAMP}.log"

# Create necessary directories
mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting database backup..."

# Database backup
PGPASSWORD="$DB_PASSWORD" pg_dump \
    -h "$DB_HOST" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --verbose \
    --no-owner \
    --no-privileges \
    --create \
    --format=custom \
    > "$BACKUP_DIR/hive_trading_${TIMESTAMP}.sql.gz" 2>> "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "Database backup completed successfully: hive_trading_${TIMESTAMP}.sql.gz"
    
    # Get backup file size
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/hive_trading_${TIMESTAMP}.sql.gz" | cut -f1)
    log "Backup file size: $BACKUP_SIZE"
    
    # Upload to S3 if configured
    if [ -n "$S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
        log "Uploading backup to S3..."
        aws s3 cp "$BACKUP_DIR/hive_trading_${TIMESTAMP}.sql.gz" \
            "s3://$S3_BUCKET/database/hive_trading_${TIMESTAMP}.sql.gz" \
            2>> "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            log "Backup uploaded to S3 successfully"
        else
            log "Failed to upload backup to S3"
        fi
    fi
    
    # Clean up old backups
    find "$BACKUP_DIR" -name "hive_trading_*.sql.gz" -mtime +${RETENTION_DAYS:-30} -delete
    log "Cleaned up backups older than ${RETENTION_DAYS:-30} days"
    
else
    log "Database backup failed!"
    exit 1
fi

log "Database backup process completed"