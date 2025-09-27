# Bloomberg Terminal Trading System - Deployment Guide

## 🚀 Quick Start (Recommended)

The fastest way to get the Bloomberg Terminal running:

```bash
# One-command deployment
chmod +x scripts/quick-start.sh
./scripts/quick-start.sh
```

This will:
- ✅ Check system requirements
- ⚙️ Set up configuration
- 🏗️ Build and deploy all services
- 🔍 Perform health checks
- 🌐 Open the terminal in your browser

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended  
- **Storage**: 50GB free space
- **Network**: Stable internet connection

### Required Software
- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 2.0+ ([Install Compose](https://docs.docker.com/compose/install/))
- **Git** for cloning the repository

## 🏗️ Manual Setup

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd bloomberg-terminal-trading-system
```

### Step 2: Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (IMPORTANT: Add your API keys)
nano .env
```

**Required API Keys:**
- **Alpaca Trading API**: Get from [Alpaca Markets](https://app.alpaca.markets/)
- **Polygon Market Data**: Get from [Polygon.io](https://polygon.io/) (optional)

### Step 3: Deploy System
```bash
# Using the professional Makefile
make setup      # First time setup
make deploy     # Deploy production system

# OR using Docker Compose directly
docker-compose -f docker-compose.bloomberg.yml up -d --build
```

### Step 4: Verify Deployment
```bash
# Check system health
make health-check

# View system status
make status

# Monitor logs
make logs
```

## 🌐 Service Access Points

Once deployed, access these services:

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Bloomberg Terminal** | http://localhost:3000 | None | Main trading interface |
| **API Documentation** | http://localhost:8000/docs | None | Backend API reference |
| **Grafana Monitoring** | http://localhost:3001 | admin/bloomberg123 | System monitoring |
| **Kibana Logs** | http://localhost:5601 | None | Log analysis |
| **Health Check** | http://localhost:8000/health | None | System health status |

## ⚙️ Configuration Guide

### Trading Configuration

```bash
# Paper Trading (Safe for testing)
PAPER_TRADING=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Live Trading (Real money - Use with extreme caution)
PAPER_TRADING=false
ALPACA_BASE_URL=https://api.alpaca.markets
```

### Risk Management Settings
```bash
# Position and risk limits
MAX_POSITION_SIZE=10000        # Maximum position size
MAX_PORTFOLIO_VALUE=1000000    # Maximum portfolio value
RISK_LIMIT=0.02               # Max 2% daily loss
MAX_DRAWDOWN=0.10             # Max 10% drawdown
INITIAL_CAPITAL=100000        # Starting capital
```

### Performance Tuning
```bash
# Worker processes
MAX_WORKERS=4
WORKER_TIMEOUT=30

# Database connections
MAX_DB_CONNECTIONS=50
DB_POOL_SIZE=20

# WebSocket settings
WS_MAX_CONNECTIONS=1000
WS_PING_INTERVAL=30000
```

## 🛠️ Development Mode

For development with hot reloading:

```bash
# Start development environment
make dev

# OR manually
docker-compose -f docker-compose.bloomberg.yml up -d timescaledb redis
cd backend && python -m uvicorn api.main:app --reload &
cd frontend && npm start &
```

## 📊 Monitoring and Observability

### System Metrics
Access Grafana at http://localhost:3001 (admin/bloomberg123) for:
- Real-time system performance
- Trading metrics and P&L
- Risk monitoring dashboards
- Agent performance tracking

### Log Analysis
Access Kibana at http://localhost:5601 for:
- Structured log search
- Error tracking and debugging
- Performance analysis
- Audit trail review

### Health Monitoring
```bash
# Automated health checks
make health-check

# Individual service status
curl http://localhost:8000/health
docker-compose -f docker-compose.bloomberg.yml ps
```

## 🚨 Emergency Procedures

### Emergency Stop
```bash
# Immediate halt of all trading operations
make emergency-stop

# OR direct API call
curl -X POST http://localhost:8000/emergency/stop
```

### Emergency Restart
```bash
# Complete system restart with health checks
make emergency-restart
```

### System Recovery
```bash
# If system becomes unresponsive
docker-compose -f docker-compose.bloomberg.yml down
docker system prune -f
make deploy
```

## 🔒 Security Best Practices

### API Key Security
- ✅ Never commit API keys to version control
- ✅ Use environment variables for all secrets
- ✅ Rotate API keys regularly
- ✅ Enable 2FA on all broker accounts

### Access Control
- ✅ Restrict network access to trading system
- ✅ Use VPN for remote access
- ✅ Monitor for unauthorized access attempts
- ✅ Enable audit logging

### Data Protection
- ✅ Enable database encryption at rest
- ✅ Use HTTPS for all external communications
- ✅ Regular security scans
- ✅ Backup encryption

## 📦 Backup and Recovery

### Automated Backups
```bash
# Enable automated backups
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30

# Manual backup
make backup
```

### System Restore
```bash
# Restore from backup
make restore

# Database restore only
docker-compose -f docker-compose.bloomberg.yml exec -T timescaledb \
  psql -U trading_user -d bloomberg_trading < backup.sql
```

## 🐛 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check port conflicts
netstat -tuln | grep -E ':(3000|8000|5432|6379)'

# View detailed logs
docker-compose -f docker-compose.bloomberg.yml logs --tail=100
```

**Database connection issues:**
```bash
# Check TimescaleDB status
docker-compose -f docker-compose.bloomberg.yml exec timescaledb pg_isready

# Reset database (DESTRUCTIVE)
make db-reset
```

**WebSocket connection problems:**
```bash
# Check WebSocket service
curl -f http://localhost:8001/health

# Restart WebSocket service
docker-compose -f docker-compose.bloomberg.yml restart backend
```

**High memory usage:**
```bash
# Check resource usage
docker stats

# Optimize Docker resources
docker system prune -f
docker volume prune -f
```

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG
DEBUG=true

# Restart services
make restart
```

### Getting Help

1. **Check logs**: `make logs`
2. **Health status**: `make health-check`  
3. **System metrics**: `make metrics`
4. **Debug info**: `make info`

## 📈 Performance Optimization

### Database Optimization
```bash
# Database tuning for TimescaleDB
shared_buffers = 256MB
effective_cache_size = 1GB
max_connections = 200
```

### Application Tuning
```bash
# Backend optimization
MAX_WORKERS=4
WORKER_TIMEOUT=30
DB_POOL_SIZE=20

# Frontend optimization
NODE_ENV=production
BUILD_PATH=build
```

### Infrastructure Scaling
```bash
# Horizontal scaling
docker-compose -f docker-compose.bloomberg.yml up -d --scale backend=3

# Resource limits
docker-compose -f docker-compose.bloomberg.yml config
```

## 🚀 Production Deployment

### Production Checklist
- ✅ All API keys configured
- ✅ SSL certificates installed
- ✅ Firewall rules configured
- ✅ Monitoring alerts set up
- ✅ Backup strategy implemented
- ✅ Emergency procedures tested
- ✅ Performance baselines established

### Production Environment Variables
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
JSON_LOGS=true
PAPER_TRADING=false  # Only if live trading approved
```

### Load Balancing
```nginx
# Example nginx configuration
upstream bloomberg_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 443 ssl;
    server_name bloomberg-terminal.your-domain.com;
    
    location /api {
        proxy_pass http://bloomberg_backend;
    }
    
    location / {
        proxy_pass http://frontend:3000;
    }
}
```

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **System Architecture**: See `/docs/architecture.md`
- **Trading Strategies**: See `/docs/strategies.md`
- **Risk Management**: See `/docs/risk-management.md`
- **Agent Framework**: See `/docs/agents.md`

---

## ⚠️ Important Disclaimers

1. **Trading Risk**: This system involves substantial risk of loss. Past performance does not guarantee future results.

2. **Use at Your Own Risk**: The authors and contributors are not responsible for any financial losses.

3. **Paper Trading First**: Always test with paper trading before using real money.

4. **Regulatory Compliance**: Ensure compliance with all applicable financial regulations.

5. **Monitoring Required**: Never leave the system unattended during live trading.

---

**Happy Trading! 📈💰**

For support and updates, check the project repository and documentation.