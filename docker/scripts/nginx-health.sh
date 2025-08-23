#!/bin/sh

# Nginx health check script
set -e

# Check if nginx master process is running
if ! pgrep -f "nginx: master process" > /dev/null; then
    echo "Nginx master process is not running"
    exit 1
fi

# Check if nginx worker processes are running
if ! pgrep -f "nginx: worker process" > /dev/null; then
    echo "Nginx worker processes are not running"
    exit 1
fi

# Check if HTTP port is responding
if ! curl -f -s http://localhost/health > /dev/null; then
    echo "HTTP health endpoint is not responding"
    exit 1
fi

# Check if HTTPS port is responding (ignore certificate errors for self-signed)
if ! curl -f -s -k https://localhost/health > /dev/null; then
    echo "HTTPS health endpoint is not responding"
    exit 1
fi

echo "Nginx is healthy"
exit 0