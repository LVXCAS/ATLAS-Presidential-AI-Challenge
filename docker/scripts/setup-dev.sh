#!/bin/bash
set -e

# Development environment setup script
echo "Setting up Hive Trade development environment..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Create development environment file
create_dev_env() {
    if [ ! -f ".env.development" ]; then
        log "Creating development environment file..."
        cat > .env.development << 'EOF'
# Hive Trade Development Environment Configuration

# Development Mode
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DB_PASSWORD=dev_password_123
DB_MEMORY=2GB
DB_CPUS=1

# Redis Configuration  
REDIS_PASSWORD=dev_redis_123

# API Keys (Use Alpaca Paper Trading)
ALPACA_API_KEY=your_paper_trading_api_key
ALPACA_SECRET_KEY=your_paper_trading_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Security (Development only - use proper secrets in production)
JWT_SECRET=development_jwt_secret_key_not_for_production_use_only

# Monitoring
GRAFANA_PASSWORD=admin123

# Backup Configuration (Disabled in development)
BACKUP_SCHEDULE=0 4 * * *
BACKUP_RETENTION_DAYS=7

# Development Settings
LOAD_SAMPLE_DATA=true
EOF
        log "Development environment file created: .env.development"
        warn "Please configure your Alpaca paper trading API keys in .env.development"
    else
        log "Development environment file already exists"
    fi
}

# Setup development directories
setup_dev_directories() {
    log "Setting up development directories..."
    
    mkdir -p {logs,data/{models,samples},database/{init,backups},monitoring/grafana/dashboards}
    mkdir -p frontend/public frontend/src
    mkdir -p backend/{api,core,services,strategies,agents}
    
    # Create sample data directory
    mkdir -p data/samples
    
    log "Development directory structure created"
}

# Install development dependencies
install_dev_dependencies() {
    log "Installing development dependencies..."
    
    # Backend dependencies
    if [ -f "backend/pyproject.toml" ]; then
        cd backend
        if command -v poetry &> /dev/null; then
            poetry install --with dev
        else
            warn "Poetry not found. Please install Poetry or use pip with requirements.txt"
        fi
        cd ..
    fi
    
    # Frontend dependencies
    if [ -f "frontend/package.json" ]; then
        cd frontend
        if command -v npm &> /dev/null; then
            npm install
        elif command -v yarn &> /dev/null; then
            yarn install
        else
            warn "Neither npm nor yarn found. Please install Node.js and npm"
        fi
        cd ..
    fi
    
    log "Development dependencies installed"
}

# Create development docker-compose
create_dev_compose() {
    if [ ! -f "docker-compose.development.yml" ]; then
        log "Creating development docker-compose file..."
        cat > docker-compose.development.yml << 'EOF'
version: '3.8'

services:
  # Development database
  timescaledb-dev:
    image: timescale/timescaledb:latest-pg14
    container_name: hive-timescaledb-dev
    environment:
      - POSTGRES_DB=hive_trading_dev
      - POSTGRES_USER=hive_dev
      - POSTGRES_PASSWORD=dev_password_123
    ports:
      - "5433:5432"
    volumes:
      - timescale_dev_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    networks:
      - hive_dev_network

  # Development Redis
  redis-dev:
    image: redis:7-alpine
    container_name: hive-redis-dev
    ports:
      - "6380:6379"
    volumes:
      - redis_dev_data:/data
    networks:
      - hive_dev_network
    command: redis-server --requirepass dev_redis_123

  # Development Grafana
  grafana-dev:
    image: grafana/grafana:latest
    container_name: hive-grafana-dev
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3001:3000"
    volumes:
      - grafana_dev_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - hive_dev_network

  # Development Prometheus
  prometheus-dev:
    image: prom/prometheus:latest
    container_name: hive-prometheus-dev
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_dev_data:/prometheus
    networks:
      - hive_dev_network

volumes:
  timescale_dev_data:
  redis_dev_data:
  grafana_dev_data:
  prometheus_dev_data:

networks:
  hive_dev_network:
    driver: bridge
EOF
        log "Development docker-compose file created"
    else
        log "Development docker-compose file already exists"
    fi
}

# Setup pre-commit hooks
setup_pre_commit() {
    if command -v pre-commit &> /dev/null; then
        log "Setting up pre-commit hooks..."
        pre-commit install
        log "Pre-commit hooks installed"
    else
        warn "pre-commit not found. Install with: pip install pre-commit"
    fi
}

# Create development scripts
create_dev_scripts() {
    log "Creating development scripts..."
    
    # Start development services
    cat > start-dev.sh << 'EOF'
#!/bin/bash
echo "Starting Hive Trade development environment..."

# Start infrastructure services
docker-compose -f docker-compose.development.yml up -d

echo "Development services started:"
echo "• Database: localhost:5433"
echo "• Redis: localhost:6380" 
echo "• Grafana: http://localhost:3001 (admin/admin123)"
echo "• Prometheus: http://localhost:9091"
echo ""
echo "To start the application:"
echo "• Backend: cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001"
echo "• Frontend: cd frontend && npm start"
EOF
    
    chmod +x start-dev.sh
    
    # Stop development services
    cat > stop-dev.sh << 'EOF'
#!/bin/bash
echo "Stopping Hive Trade development environment..."
docker-compose -f docker-compose.development.yml down
echo "Development services stopped"
EOF
    
    chmod +x stop-dev.sh
    
    log "Development scripts created: start-dev.sh, stop-dev.sh"
}

# Main setup function
main() {
    log "Setting up Hive Trade development environment..."
    
    create_dev_env
    setup_dev_directories
    create_dev_compose
    install_dev_dependencies
    setup_pre_commit
    create_dev_scripts
    
    log "Development environment setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Configure your Alpaca paper trading API keys in .env.development"
    echo "2. Run './start-dev.sh' to start development services"
    echo "3. Start the backend: cd backend && python -m uvicorn api.main:app --reload --port 8001"
    echo "4. Start the frontend: cd frontend && npm start"
    echo ""
    echo "Happy coding! 🚀"
}

# Run main function
main