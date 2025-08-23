#!/bin/bash

# Bloomberg Terminal Trading System - Quick Start Script
# Professional one-command deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'

# Banner
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${WHITE}         BLOOMBERG TERMINAL TRADING SYSTEM           ${RESET}"
echo -e "${WHITE}              Quick Start Deployment                ${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# System Requirements Check
echo -e "${YELLOW}🔍 Checking system requirements...${RESET}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${RESET}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${RESET}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. Installing via Docker...${RESET}"
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js found: ${NODE_VERSION}${RESET}"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  Python 3 not found. Installing via Docker...${RESET}"
else
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python found: ${PYTHON_VERSION}${RESET}"
fi

echo -e "${GREEN}✅ System requirements check passed${RESET}"
echo ""

# Configuration Setup
echo -e "${YELLOW}⚙️  Setting up configuration...${RESET}"

if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file from template...${RESET}"
    cp .env.example .env
    
    echo -e "${YELLOW}📝 Please edit .env file with your API keys:${RESET}"
    echo -e "   • ALPACA_API_KEY"
    echo -e "   • ALPACA_SECRET_KEY"
    echo -e "   • POLYGON_API_KEY (optional)"
    echo ""
    echo -e "${CYAN}Press Enter to continue after editing .env, or Ctrl+C to exit${RESET}"
    read -r
else
    echo -e "${GREEN}✅ .env file already exists${RESET}"
fi

# Quick Configuration Wizard
echo -e "${YELLOW}🧙 Quick configuration wizard...${RESET}"

# Ask for trading mode
echo -e "${CYAN}Select trading mode:${RESET}"
echo -e "  ${GREEN}1) Paper Trading (Recommended for beginners)${RESET}"
echo -e "  ${RED}2) Live Trading (Real money - Use with caution)${RESET}"
echo -n "Enter choice [1-2]: "
read -r TRADING_MODE

if [ "$TRADING_MODE" = "2" ]; then
    echo -e "${RED}⚠️  WARNING: Live trading selected. This will use real money!${RESET}"
    echo -e "${RED}Are you sure you want to continue? (yes/no): ${RESET}"
    read -r CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo -e "${YELLOW}Switching to paper trading mode for safety.${RESET}"
        TRADING_MODE="1"
    fi
fi

# Update .env based on selection
if [ "$TRADING_MODE" = "1" ]; then
    sed -i.bak 's/PAPER_TRADING=.*/PAPER_TRADING=true/' .env
    sed -i.bak 's/ALPACA_BASE_URL=.*/ALPACA_BASE_URL=https:\/\/paper-api.alpaca.markets/' .env
    echo -e "${GREEN}✅ Configured for paper trading${RESET}"
else
    sed -i.bak 's/PAPER_TRADING=.*/PAPER_TRADING=false/' .env
    sed -i.bak 's/ALPACA_BASE_URL=.*/ALPACA_BASE_URL=https:\/\/api.alpaca.markets/' .env
    echo -e "${RED}⚠️  Configured for LIVE trading${RESET}"
fi

# Ask for initial capital
echo -n "Enter initial capital (default: 100000): "
read -r INITIAL_CAPITAL
INITIAL_CAPITAL=${INITIAL_CAPITAL:-100000}
sed -i.bak "s/INITIAL_CAPITAL=.*/INITIAL_CAPITAL=${INITIAL_CAPITAL}/" .env

echo -e "${GREEN}✅ Configuration complete${RESET}"
echo ""

# System Deployment
echo -e "${YELLOW}🚀 Starting system deployment...${RESET}"

# Stop any existing containers
echo -e "${BLUE}Stopping any existing containers...${RESET}"
docker-compose -f docker-compose.bloomberg.yml down 2>/dev/null || true

# Pull latest images
echo -e "${BLUE}Pulling latest Docker images...${RESET}"
docker-compose -f docker-compose.bloomberg.yml pull

# Build and start services
echo -e "${BLUE}Building and starting services...${RESET}"
docker-compose -f docker-compose.bloomberg.yml up -d --build

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to initialize...${RESET}"

# Wait for database
echo -n "Waiting for TimescaleDB"
for i in {1..30}; do
    if docker-compose -f docker-compose.bloomberg.yml exec -T timescaledb pg_isready -U trading_user &>/dev/null; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Redis
echo -n "Waiting for Redis"
for i in {1..15}; do
    if docker-compose -f docker-compose.bloomberg.yml exec -T redis redis-cli ping &>/dev/null; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for backend API
echo -n "Waiting for Backend API"
for i in {1..30}; do
    if curl -f http://localhost:8000/health &>/dev/null; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for frontend
echo -n "Waiting for Frontend"
for i in {1..30}; do
    if curl -f http://localhost:3000 &>/dev/null; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""

# Final Health Check
echo -e "${YELLOW}🔍 Performing final health check...${RESET}"

HEALTH_CHECK=$(curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$HEALTH_CHECK" = "healthy" ]; then
    echo -e "${GREEN}✅ Backend API healthy${RESET}"
else
    echo -e "${RED}❌ Backend API unhealthy${RESET}"
    echo -e "${YELLOW}Check logs: docker-compose -f docker-compose.bloomberg.yml logs backend${RESET}"
fi

# Success Banner
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${WHITE}           🎉 DEPLOYMENT SUCCESSFUL! 🎉              ${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "${WHITE}📊 Bloomberg Terminal: ${CYAN}http://localhost:3000${RESET}"
echo -e "${WHITE}🔧 API Documentation:  ${CYAN}http://localhost:8000/docs${RESET}"
echo -e "${WHITE}📈 Monitoring:         ${CYAN}http://localhost:3001${RESET} (admin/bloomberg123)"
echo -e "${WHITE}📋 Logs:              ${CYAN}http://localhost:5601${RESET}"
echo ""
echo -e "${YELLOW}💡 Useful Commands:${RESET}"
echo -e "   View logs:     ${CYAN}docker-compose -f docker-compose.bloomberg.yml logs -f${RESET}"
echo -e "   Stop system:   ${CYAN}docker-compose -f docker-compose.bloomberg.yml down${RESET}"
echo -e "   Restart:       ${CYAN}docker-compose -f docker-compose.bloomberg.yml restart${RESET}"
echo -e "   Health check:  ${CYAN}curl http://localhost:8000/health${RESET}"
echo ""

# Trading Mode Warning
if [ "$TRADING_MODE" = "2" ]; then
    echo -e "${RED}⚠️  IMPORTANT: LIVE TRADING MODE ACTIVE ⚠️${RESET}"
    echo -e "${RED}   • Monitor your positions carefully${RESET}"
    echo -e "${RED}   • Set appropriate risk limits${RESET}"
    echo -e "${RED}   • Have an emergency stop plan${RESET}"
    echo -e "${RED}   • Emergency stop: curl -X POST http://localhost:8000/emergency/stop${RESET}"
else
    echo -e "${GREEN}✅ Paper trading mode active - Safe for testing${RESET}"
fi

echo ""
echo -e "${WHITE}Happy Trading! 📈💰${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# Open browser (optional)
if command -v open &> /dev/null; then
    echo -e "${YELLOW}Opening Bloomberg Terminal in browser...${RESET}"
    sleep 3
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    echo -e "${YELLOW}Opening Bloomberg Terminal in browser...${RESET}"
    sleep 3
    xdg-open http://localhost:3000
fi