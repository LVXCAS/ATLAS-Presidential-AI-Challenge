#!/bin/bash

# Hive Trade Database Restore Script
# Restores PostgreSQL database from backup

set -euo pipefail

# Configuration
BACKUP_DIR="/app/backups"
LOG_FILE="/app/logs/restore.log"

# Database configuration
DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-trading_db}
DB_USER=${DB_USER:-trader}
DB_PASSWORD=${DB_PASSWORD:-secure_password_123}

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# List available backups
list_backups() {
    log "Available backups:"
    find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f | sort -r | head -20 | while read -r backup; do
        local size
        local date
        size=$(du -h "$backup" | cut -f1)
        date=$(stat -c %y "$backup" | cut -d' ' -f1,2 | cut -d'.' -f1)
        log "  $(basename "$backup") - $size - $date"
    done
}

# Validate backup file
validate_backup() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        error_exit "Backup file not found: $backup_file"
    fi
    
    log "Validating backup file: $backup_file"
    
    # Check if file is gzipped
    if ! gzip -t "$backup_file" 2>/dev/null; then
        error_exit "Backup file is not a valid gzip file"
    fi
    
    # Check if it contains SQL dump content
    if ! zcat "$backup_file" | head -10 | grep -q "PostgreSQL database dump"; then
        error_exit "Backup file does not appear to be a PostgreSQL dump"
    fi
    
    log "Backup file validation passed"
}

# Test database connectivity
test_db_connection() {
    log "Testing database connectivity..."
    
    if ! PGPASSWORD="$DB_PASSWORD" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; then
        error_exit "Cannot connect to database server"
    fi
    
    log "Database connection test passed"
}

# Create database if it doesn't exist
create_database() {
    log "Checking if database exists..."
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log "Database $DB_NAME already exists"
    else
        log "Creating database $DB_NAME..."
        PGPASSWORD="$DB_PASSWORD" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
        log "Database created successfully"
    fi
}

# Perform restore
perform_restore() {
    local backup_file="$1"
    local confirm_restore="${2:-false}"
    
    validate_backup "$backup_file"
    test_db_connection
    
    if [ "$confirm_restore" != "true" ]; then
        log "WARNING: This will overwrite the existing database!"
        log "To confirm restore, run: $0 restore $backup_file true"
        return 1
    fi
    
    log "Starting database restore from: $backup_file"
    
    # Create database if needed
    create_database
    
    # Drop existing connections to the database
    log "Terminating existing database connections..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" || true
    
    # Restore database
    log "Restoring database..."
    if zcat "$backup_file" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1; then
        log "Database restore completed successfully"
    else
        error_exit "Database restore failed"
    fi
    
    # Verify restore
    verify_restore
}

# Verify restore
verify_restore() {
    log "Verifying database restore..."
    
    # Check if we can connect to the restored database
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null; then
        error_exit "Cannot connect to restored database"
    fi
    
    # Check if tables exist
    local table_count
    table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
    
    if [ "${table_count:-0}" -eq 0 ]; then
        error_exit "No tables found in restored database"
    fi
    
    log "Database restore verification passed ($table_count tables found)"
}

# Emergency restore from latest backup
emergency_restore() {
    local latest_backup="$BACKUP_DIR/latest.sql.gz"
    
    if [ ! -f "$latest_backup" ]; then
        error_exit "No latest backup found at: $latest_backup"
    fi
    
    log "Performing emergency restore from latest backup..."
    perform_restore "$latest_backup" true
}

# Point-in-time recovery simulation
point_in_time_restore() {
    local target_time="$1"
    
    if [ -z "$target_time" ]; then
        error_exit "Target time required for point-in-time restore (format: YYYY-MM-DD HH:MM:SS)"
    fi
    
    log "Finding backup closest to: $target_time"
    
    # Find the backup closest to the target time
    local closest_backup
    closest_backup=$(find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f | while read -r backup; do
        local backup_time
        backup_time=$(echo "$backup" | sed 's/.*db_backup_\([0-9]*_[0-9]*\)\.sql\.gz/\1/' | sed 's/_/ /')
        backup_time=$(date -d "${backup_time:0:8} ${backup_time:9:6}" +%s 2>/dev/null || echo 0)
        local target_timestamp
        target_timestamp=$(date -d "$target_time" +%s 2>/dev/null || echo 0)
        
        if [ "$backup_time" -le "$target_timestamp" ]; then
            echo "$backup_time $backup"
        fi
    done | sort -nr | head -1 | cut -d' ' -f2)
    
    if [ -z "$closest_backup" ]; then
        error_exit "No suitable backup found for point-in-time restore"
    fi
    
    log "Using backup: $(basename "$closest_backup")"
    perform_restore "$closest_backup" true
}

# Usage information
show_usage() {
    cat << EOF
Usage: $0 {command} [options]

Commands:
    list                        List available backups
    restore {backup_file} [true] Restore from specific backup (confirmation required)
    emergency                   Restore from latest backup immediately
    pit {target_time}          Point-in-time restore (YYYY-MM-DD HH:MM:SS)
    validate {backup_file}     Validate backup file integrity

Examples:
    $0 list
    $0 restore /app/backups/db_backup_20240123_120000.sql.gz true
    $0 emergency
    $0 pit "2024-01-23 12:00:00"
    $0 validate /app/backups/db_backup_20240123_120000.sql.gz

EOF
}

# Main execution
case "${1:-help}" in
    "list")
        list_backups
        ;;
    "restore")
        if [ $# -lt 2 ]; then
            error_exit "Backup file required for restore"
        fi
        perform_restore "$2" "${3:-false}"
        ;;
    "emergency")
        emergency_restore
        ;;
    "pit"|"point-in-time")
        if [ $# -lt 2 ]; then
            error_exit "Target time required for point-in-time restore"
        fi
        point_in_time_restore "$2"
        ;;
    "validate")
        if [ $# -lt 2 ]; then
            error_exit "Backup file required for validation"
        fi
        validate_backup "$2"
        log "Backup file is valid"
        ;;
    "help"|*)
        show_usage
        exit 0
        ;;
esac

log "Restore script completed"