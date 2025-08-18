#!/bin/bash
set -e

# Create necessary directories
mkdir -p deploy/nginx/ssl
mkdir -p database/backups
mkdir -p logs

# Set permissions
chmod 755 .
chmod 755 deploy
chmod 600 deploy/nginx/ssl/* 2>/dev/null || true

# Create .env file if it doesn't exist
if [ ! -f .env.prod ]; then
    cat > .env.prod <<EOL
# Database
DB_NAME=trading_prod
DB_USER=trading_user
DB_PASSWORD=$(openssl rand -hex 32)
DB_HOST=postgres
DB_PORT=5432

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -hex 32)

# Django
DJANGO_SECRET_KEY=$(openssl rand -hex 50)
DJANGO_DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

# Trading APIs (fill these in)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 12)

# Environment
ENVIRONMENT=production
EOL
    echo "Created .env.prod file. Please update with your API keys and review the configuration."
    chmod 600 .env.prod
fi

echo "Production setup complete. Next steps:"
echo "1. Update .env.prod with your API keys and configuration"
echo "2. Run: docker-compose -f docker-compose.prod.yml up -d --build"
echo "3. Set up SSL certificates in deploy/nginx/ssl/"
echo "4. Monitor the application at http://localhost:3000 (Grafana)"
