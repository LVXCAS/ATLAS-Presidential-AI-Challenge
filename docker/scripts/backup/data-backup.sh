#!/bin/bash
set -e

# Application data backup script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/backups/data"
LOG_FILE="/backups/logs/data_backup_${TIMESTAMP}.log"

# Create necessary directories
mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting application data backup..."

# Create archive of important application data
tar -czf "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" \
    -C /app \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='__pycache__' \
    --exclude='.git' \
    data/ \
    config/ \
    2>> "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "Application data backup completed successfully: app_data_${TIMESTAMP}.tar.gz"
    
    # Get backup file size
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" | cut -f1)
    log "Backup file size: $BACKUP_SIZE"
    
    # Upload to S3 if configured
    if [ -n "$S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
        log "Uploading data backup to S3..."
        aws s3 cp "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" \
            "s3://$S3_BUCKET/data/app_data_${TIMESTAMP}.tar.gz" \
            2>> "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            log "Data backup uploaded to S3 successfully"
        else
            log "Failed to upload data backup to S3"
        fi
    fi
    
    # Clean up old backups
    find "$BACKUP_DIR" -name "app_data_*.tar.gz" -mtime +${RETENTION_DAYS:-30} -delete
    log "Cleaned up data backups older than ${RETENTION_DAYS:-30} days"
    
else
    log "Application data backup failed!"
    exit 1
fi

log "Application data backup process completed"