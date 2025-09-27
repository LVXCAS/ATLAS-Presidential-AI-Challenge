# Hive Trade v0.2 - Project Structure

This document outlines the clean, organized structure of the Hive Trade algorithmic trading system.

## 🏗️ Core Directories

### `/agents/` - AI Trading Agents
- **Core agents**: Market making, momentum, statistical arbitrage
- **Support agents**: Risk management, portfolio allocation, execution
- **Workflow**: LangGraph orchestration and communication protocols

### `/backend/` - FastAPI Backend Services
- **API routes**: REST endpoints for all system operations
- **Core services**: Database, Redis, WebSocket management
- **ML pipeline**: Feature engineering and model inference
- **Monitoring**: Metrics collection and health checks

### `/frontend/` - React Trading Dashboard
- **Components**: Bloomberg-style terminal interface
- **Real-time data**: WebSocket integration for live updates
- **Panels**: Orders, positions, risk, analytics, charts

### `/config/` - System Configuration
- **Database**: Optimized PostgreSQL and Redis settings
- **Security**: Encryption, authentication, SSL certificates
- **Settings**: Environment-specific configurations

### `/docker/` - Containerization
- **Main compose**: `docker-compose.yml` for full system
- **Services**: Individual Dockerfiles for each component
- **Monitoring**: Prometheus, Grafana, ELK stack integration

## 📊 Data & Analytics

### `/database/` - Database Management
- **Schema**: Optimized tables for high-frequency trading
- **Migrations**: Version-controlled schema changes
- **Performance**: Indexes, partitioning, query optimization

### `/models/` - ML Model Artifacts
- **Trading models**: Trained RL and ensemble models
- **Risk models**: Volatility and correlation predictions
- **Metadata**: Model versioning and performance metrics

### `/strategies/` - Trading Strategies
- **Backtesting**: Historical performance testing
- **Technical indicators**: TA-Lib integration
- **Parameter optimization**: Grid search and genetic algorithms

## 🛠️ Development & Operations

### `/scripts/` - Automation Scripts
- **Setup**: Development environment initialization
- **Validation**: System health and performance checks
- **Deployment**: Production deployment automation

### `/deploy/` - Production Deployment
- **Monitoring**: Complete observability stack setup
- **Configuration**: Production-ready settings
- **Scripts**: Automated deployment and scaling

### `/tests/` - Test Suite
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end system validation
- **Performance tests**: Load and stress testing

## 📁 Support Directories

### `/logs/` - System Logs
- **Trading logs**: Order execution and performance
- **System logs**: Application events and errors
- **Audit logs**: Compliance and security tracking

### `/data/` - Data Storage
- **Market data**: Historical and real-time price data
- **Backtest results**: Strategy performance analysis
- **Training data**: ML model training datasets

### `/archive/` - Historical Artifacts
- **Old dashboards**: Previous UI iterations
- **Task summaries**: Development milestone documentation
- **Docker configs**: Legacy container configurations

## 🚀 Key Entry Points

### Production Deployment
```bash
# Full system deployment
python deploy/setup-production.py

# Docker-based deployment
cd docker && docker-compose up -d
```

### Development Setup
```bash
# Initialize development environment
python scripts/setup_dev.py

# Run individual components
python backend/main.py        # Backend API
npm start                     # Frontend dashboard (in /frontend)
python agents/workflow.py     # AI agents
```

### System Validation
```bash
# Health check
python scripts/health_check.py

# Performance validation
python scripts/validate_performance.py

# Security audit
python scripts/validate_security.py
```

## 📋 File Organization Rules

### What Goes Where
- **Root level**: Only essential files (README, requirements, config)
- **Timestamped outputs**: Automatically cleaned up by .gitignore
- **Temporary files**: Go to `/tmp/` or `/archive/`
- **Sensitive configs**: Use environment variables and .env files

### Clean Development Practices
1. **No loose scripts** - All Python files organized in proper directories
2. **No output files** - Results go to `/data/` or `/logs/`
3. **No duplicate configs** - Single source of truth for each setting
4. **Clear naming** - Descriptive file and directory names

## 🎯 Production Ready

The system is now:
- ✅ **Clean and organized** - Professional project structure
- ✅ **Scalable** - Docker-based microservices architecture
- ✅ **Observable** - Full monitoring and logging
- ✅ **Secure** - Encryption, authentication, secrets management
- ✅ **Performant** - Database optimized for real-time trading
- ✅ **Maintainable** - Clear separation of concerns

Ready for algorithmic trading at scale! 🚀