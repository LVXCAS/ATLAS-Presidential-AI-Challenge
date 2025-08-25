#!/bin/bash

# Hive Trade Database Backup Script
# Performs automated backups of PostgreSQL database

set -euo pipefail

# Configuration
BACKUP_DIR="/app/backups"
LOG_FILE="/app/logs/backup.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Database configuration
DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-trading_db}
DB_USER=${DB_USER:-trader}
DB_PASSWORD=${DB_PASSWORD:-secure_password_123}

# Backup file names
DB_BACKUP_FILE="$BACKUP_DIR/db_backup_$TIMESTAMP.sql"
COMPRESSED_BACKUP="$DB_BACKUP_FILE.gz"
LATEST_BACKUP="$BACKUP_DIR/latest.sql"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Cleanup function
cleanup() {
    if [ -f "$DB_BACKUP_FILE" ]; then
        rm -f "$DB_BACKUP_FILE"
    fi
}

trap cleanup EXIT

# Main backup function
perform_backup() {
    log "Starting database backup..."
    
    # Test database connectivity
    if ! PGPASSWORD="$DB_PASSWORD" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"; then
        error_exit "Database is not accessible"
    fi
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Perform database backup
    log "Creating database dump..."
    if PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        > "$DB_BACKUP_FILE"; then
        
        log "Database dump completed successfully"
    else
        error_exit "Database dump failed"
    fi
    
    # Compress backup
    log "Compressing backup..."
    if gzip -9 "$DB_BACKUP_FILE"; then
        log "Backup compressed: $COMPRESSED_BACKUP"
    else
        error_exit "Backup compression failed"
    fi
    
    # Create symlink to latest backup
    ln -sf "$(basename "$COMPRESSED_BACKUP")" "$LATEST_BACKUP.gz"
    
    # Get backup size
    BACKUP_SIZE=$(du -h "$COMPRESSED_BACKUP" | cut -f1)
    log "Backup completed: $COMPRESSED_BACKUP ($BACKUP_SIZE)"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Verify backup integrity
    verify_backup
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    
    local removed_count
    removed_count=$(find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -mtime +$RETENTION_DAYS | wc -l)
    
    if [ "$removed_count" -gt 0 ]; then
        log "Removed $removed_count old backup(s)"
    else
        log "No old backups to remove"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    if gzip -t "$COMPRESSED_BACKUP"; then
        log "Backup integrity check passed"
    else
        error_exit "Backup integrity check failed"
    fi
    
    # Check if backup contains expected content
    if zcat "$COMPRESSED_BACKUP" | head -10 | grep -q "PostgreSQL database dump"; then
        log "Backup content verification passed"
    else
        error_exit "Backup content verification failed"
    fi
}

# Send backup notification (if configured)
send_notification() {
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Hive Trade backup completed successfully: $COMPRESSED_BACKUP ($BACKUP_SIZE)\"}" \
            "$SLACK_WEBHOOK_URL" || log "Failed to send Slack notification"
    fi
    
    if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-Type: application/json' \
            --data "{\"content\":\"Hive Trade backup completed: $COMPRESSED_BACKUP ($BACKUP_SIZE)\"}" \
            "$DISCORD_WEBHOOK_URL" || log "Failed to send Discord notification"
    fi
}

# Health check function
health_check() {
    local latest_backup_age
    
    if [ -f "$LATEST_BACKUP.gz" ]; then
        latest_backup_age=$(find "$LATEST_BACKUP.gz" -mtime +1 | wc -l)
        if [ "$latest_backup_age" -gt 0 ]; then
            error_exit "Latest backup is older than 24 hours"
        fi
        log "Health check passed: Recent backup found"
        return 0
    else
        error_exit "No backup found"
    fi
}

# Main execution
case "${1:-backup}" in
    "backup")
        perform_backup
        send_notification
        ;;
    "health")
        health_check
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 {backup|health|cleanup}"
        exit 1
        ;;
esac

log "Backup script completed successfully"