# Hive Trade - Bloomberg Terminal Style Trading System

A professional-grade autonomous trading system with a Bloomberg Terminal-inspired interface, comprehensive monitoring, and advanced risk management capabilities.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (recommended)
- 20GB+ free disk space
- Alpaca Trading API account

### Production Deployment

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd hive-trade
   chmod +x deploy.sh
   ```

2. **Deploy:**
   ```bash
   ./deploy.sh
   ```
   This will create `.env.production` template - configure it with your settings and run again.

3. **Access the system:**
   - Trading Terminal: https://localhost
   - Grafana Dashboards: https://localhost/grafana
   - API Documentation: https://localhost/api/docs

## 📋 System Architecture

### Core Services

- **Frontend**: React-based Bloomberg Terminal interface
- **Backend**: FastAPI-based trading engine with WebSocket support  
- **Database**: TimescaleDB for time-series financial data
- **Cache**: Redis for real-time data and session management
- **Monitoring**: Prometheus, Grafana, Jaeger distributed tracing
- **Proxy**: Nginx reverse proxy with SSL termination

### Trading Components

- **Market Data**: Real-time streaming from Alpaca API
- **Order Management**: Advanced order types and execution
- **Risk Management**: Real-time position and portfolio risk monitoring
- **Analytics**: Performance tracking and strategy analysis
- **Agents**: AI-powered trading strategies and automation

## 🎛️ Bloomberg Terminal Interface

### Available Panels

1. **Dashboard** - Executive overview with key metrics
2. **System Status** - System health and connectivity
3. **Market Watch** - Real-time quotes and watchlists
4. **Technical Chart** - Advanced charting with indicators
5. **Order Book** - Level 2 market data
6. **Positions** - Portfolio positions tracking
7. **Orders** - Active orders management
8. **Risk Monitor** - Real-time risk metrics
9. **News Feed** - Financial news with filtering
10. **Analytics** - Trading performance analysis
11. **Alerts** - System and trading notifications
12. **Monitoring** - System metrics and health
13. **Settings** - Terminal configuration

### Features

- **6x6 Grid Layout** - Flexible panel arrangement
- **Real-time Updates** - WebSocket-based live data
- **Panel Management** - Minimize, maximize, close controls
- **Quick Symbol Navigation** - Cross-panel symbol synchronization
- **Keyboard Shortcuts** - Professional hotkeys
- **Dark Theme** - Bloomberg-inspired professional styling

## ⚙️ Configuration

### Environment Variables

Create `.env.production` with the following:

```bash
# Database Configuration
DB_PASSWORD=your_secure_database_password
DB_MEMORY=4GB
DB_CPUS=2

# Redis Configuration  
REDIS_PASSWORD=your_secure_redis_password

# API Keys (REQUIRED)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Security
JWT_SECRET=your_very_long_jwt_secret_key_32_chars_min

# Monitoring
GRAFANA_PASSWORD=your_grafana_admin_password
```

## 📊 Monitoring & Observability

### Metrics Collection

- **Business Metrics**: Trading performance, P&L, win rates
- **System Metrics**: CPU, memory, disk, network usage
- **Application Metrics**: API latency, error rates, throughput
- **Custom Metrics**: Agent performance, risk calculations

### Dashboards

Access Grafana at `https://localhost/grafana`:

- **Trading Performance Dashboard**: P&L, positions, orders
- **System Health Dashboard**: Infrastructure metrics
- **Risk Management Dashboard**: Risk metrics and alerts
- **Agent Performance Dashboard**: AI strategy tracking

## 💾 Data Management

### Backup Strategy

Automated backups:
- **Database**: Daily PostgreSQL dumps
- **Application Data**: Configuration and models
- **Retention**: Configurable (default 30 days)
- **Cloud Storage**: Optional S3 integration

## 🤖 Trading Agents

### Available Strategies

1. **Mean Reversion Agent**: Statistical arbitrage
2. **Momentum Agent**: Trend following
3. **Arbitrage Agent**: Cross-market opportunities
4. **News Sentiment Agent**: NLP-based trading
5. **Adaptive Optimizer**: Dynamic strategy allocation

## 🏗️ Development

### Development Setup

1. **Setup development environment:**
   ```bash
   chmod +x docker/scripts/setup-dev.sh
   ./docker/scripts/setup-dev.sh
   ```

2. **Start development services:**
   ```bash
   ./start-dev.sh
   ```

## 📚 API Documentation

Interactive API documentation available at:
- **Swagger UI**: `https://localhost/api/docs`
- **ReDoc**: `https://localhost/api/redoc`

## 🔧 Deployment Commands

```bash
# Deploy production system
./deploy.sh

# Stop all services
./deploy.sh stop

# View service status
./deploy.sh status

# View logs
./deploy.sh logs [service]

# Manual backup
./deploy.sh backup

# Update deployment
./deploy.sh update
```

## 📈 Performance Features

- **Async Operations**: Non-blocking I/O throughout
- **Connection Pooling**: Optimized database connections
- **Multi-layer Caching**: Redis and application-level caching
- **Load Balancing**: Nginx upstream configuration
- **Code Splitting**: Dynamic frontend loading
- **Time-series Optimization**: TimescaleDB hypertables

## 🔐 Security Features

- **JWT Authentication**: Secure API access
- **SSL/TLS Encryption**: HTTPS by default
- **Rate Limiting**: API protection
- **Input Validation**: Comprehensive data validation
- **Audit Logging**: Complete activity tracking

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting:**
   ```bash
   docker-compose -f docker-compose.production.yml logs
   ```

2. **Database connection issues:**
   ```bash
   docker-compose -f docker-compose.production.yml exec timescaledb psql -U hive_user -d hive_trading
   ```

3. **API key errors:**
   - Verify Alpaca credentials
   - Check API key permissions
   - Confirm paper vs live environment

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

---

**Built with ❤️ for professional traders and developers**
