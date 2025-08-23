#!/bin/sh

# Frontend health check script
set -e

# Check if nginx is running
if ! pgrep nginx > /dev/null; then
    echo "Nginx is not running"
    exit 1
fi

# Check if port 3000 is responding
if ! curl -f -s http://localhost:3000/health > /dev/null; then
    echo "Frontend health endpoint is not responding"
    exit 1
fi

# Check if main HTML file exists
if [ ! -f "/usr/share/nginx/html/index.html" ]; then
    echo "Main HTML file is missing"
    exit 1
fi

echo "Frontend is healthy"
exit 0