#!/bin/bash
set -e

# Hive Trade Deployment Script
# Production deployment for Bloomberg Terminal Trading System

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"
BACKUP_ENV=".env.backup"

# Functions
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "    HIVE TRADE - BLOOMBERG TERMINAL DEPLOYMENT"
    echo "    Professional Trading System"
    echo "=================================================="
    echo -e "${NC}"
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found. Creating template..."
        create_env_template
        error "Please configure $ENV_FILE with your settings and run again."
        exit 1
    fi
    
    # Check disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=10485760  # 10GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        error "Insufficient disk space. Need at least 10GB available."
        exit 1
    fi
    
    # Check memory (need at least 8GB)
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    REQUIRED_MEM=8192  # 8GB in MB
    
    if [ "$TOTAL_MEM" -lt "$REQUIRED_MEM" ]; then
        warn "Recommended minimum 8GB RAM for optimal performance."
    fi
    
    log "Prerequisites check passed"
}

create_env_template() {
    cat > "$ENV_FILE" << 'EOF'
# Hive Trade Production Environment Configuration

# Database Configuration
DB_PASSWORD=your_secure_database_password_here
DB_MEMORY=4GB
DB_CPUS=2

# Redis Configuration  
REDIS_PASSWORD=your_secure_redis_password_here

# API Keys (REQUIRED)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Security
JWT_SECRET=your_very_long_jwt_secret_key_here_at_least_32_characters

# Monitoring
GRAFANA_PASSWORD=your_grafana_admin_password_here

# Backup Configuration (Optional)
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
S3_BACKUP_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Error Tracking (Optional)
SENTRY_DSN=

# Domain Configuration
DOMAIN=localhost
EOF
    
    log "Environment template created at $ENV_FILE"
}

setup_directories() {
    log "Setting up directories..."
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data/{models,backups}
    mkdir -p database/{init,backups}
    mkdir -p monitoring/{grafana/dashboards,loki,promtail}
    mkdir -p docker/nginx/ssl
    
    # Set proper permissions
    chmod 755 logs data database monitoring
    chmod -R 744 docker/scripts/
    
    log "Directory structure created"
}

build_images() {
    log "Building Docker images..."
    
    # Build images in parallel for faster deployment
    docker-compose -f "$COMPOSE_FILE" build --parallel
    
    if [ $? -eq 0 ]; then
        log "Docker images built successfully"
    else
        error "Failed to build Docker images"
        exit 1
    fi
}

deploy_services() {
    log "Deploying services..."
    
    # Load environment variables
    source "$ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    
    # Deploy infrastructure services first
    log "Starting infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d timescaledb redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 30
    
    # Deploy application services
    log "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" up -d backend frontend nginx
    
    # Wait for application to be ready
    log "Waiting for application services..."
    sleep 20
    
    # Deploy monitoring services
    log "Starting monitoring services..."
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana alertmanager jaeger
    
    # Deploy supporting services
    log "Starting supporting services..."
    docker-compose -f "$COMPOSE_FILE" up -d node_exporter redis_exporter cadvisor loki promtail backup
    
    log "All services deployed"
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check service health
    local failed_services=()
    
    # Wait for services to stabilize
    sleep 30
    
    # Check each critical service
    services=("nginx" "backend" "frontend" "timescaledb" "redis" "prometheus" "grafana")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep "$service" | grep -q "Up"; then
            log "$service is running"
        else
            error "$service is not running properly"
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log "All critical services are running"
        return 0
    else
        error "Failed services: ${failed_services[*]}"
        return 1
    fi
}

show_access_info() {
    log "Deployment completed successfully!"
    echo
    echo -e "${BLUE}Access Information:${NC}"
    echo "================================="
    echo "• Trading Terminal: https://localhost"
    echo "• API Documentation: https://localhost/api/docs"
    echo "• Grafana Dashboards: https://localhost/grafana"
    echo "• System Metrics: http://localhost:8080/prometheus"
    echo "• Distributed Tracing: http://localhost:8080/jaeger"
    echo
    echo -e "${YELLOW}Default Credentials:${NC}"
    echo "• Grafana: admin / (check .env.production)"
    echo
    echo -e "${YELLOW}Important Notes:${NC}"
    echo "• Configure your Alpaca API keys in $ENV_FILE"
    echo "• Set up SSL certificates for production use"
    echo "• Review monitoring alerts and thresholds"
    echo "• Configure backup destinations (S3)"
    echo "• Monitor logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo
}

cleanup_on_failure() {
    error "Deployment failed. Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down
    exit 1
}

# Main deployment flow
main() {
    print_banner
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    check_prerequisites
    setup_directories
    build_images
    deploy_services
    
    if verify_deployment; then
        show_access_info
    else
        error "Deployment verification failed. Check logs for details."
        echo "View logs: docker-compose -f $COMPOSE_FILE logs"
        exit 1
    fi
}

# Handle command line arguments
case "$1" in
    "stop")
        log "Stopping all services..."
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    "restart")
        log "Restarting all services..."
        docker-compose -f "$COMPOSE_FILE" restart
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "backup")
        log "Running manual backup..."
        docker-compose -f "$COMPOSE_FILE" exec backup /app/scripts/database-backup.sh
        docker-compose -f "$COMPOSE_FILE" exec backup /app/scripts/data-backup.sh
        ;;
    "update")
        log "Updating deployment..."
        docker-compose -f "$COMPOSE_FILE" pull
        docker-compose -f "$COMPOSE_FILE" up -d
        ;;
    *)
        main
        ;;
esac