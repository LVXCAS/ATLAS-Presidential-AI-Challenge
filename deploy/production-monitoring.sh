#!/bin/bash

# Production Monitoring Setup for Hive Trade
# This script sets up comprehensive monitoring infrastructure

set -e

echo "🚀 Setting up Hive Trade Production Monitoring..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found. Please install Docker first." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose not found. Please install Docker Compose first." >&2; exit 1; }

# Create monitoring directories
echo "📁 Creating monitoring directory structure..."
mkdir -p monitoring/{grafana/{dashboards,provisioning/{dashboards,datasources}},prometheus,alertmanager,loki,promtail}

# Set up Grafana datasources
echo "⚙️  Configuring Grafana datasources..."
cat > monitoring/grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      httpMethod: "POST"
      
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true
    
  - name: Redis
    type: redis-datasource
    access: proxy
    url: redis://redis:6379
    editable: true
    jsonData:
      client: "standalone"
EOF

# Set up Grafana dashboard provisioning
echo "📊 Setting up Grafana dashboard provisioning..."
cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create comprehensive Prometheus alert rules
echo "🔔 Setting up Prometheus alert rules..."
cat > monitoring/prometheus/alert_rules.yml << EOF
groups:
  - name: trading_system_alerts
    rules:
      # High-priority trading alerts
      - alert: TradingSystemDown
        expr: up{job="hive-trade-core"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Trading system is down"
          description: "Core trading system has been down for more than 30 seconds"
          
      - alert: HighLatencyTrades
        expr: histogram_quantile(0.95, rate(trading_execution_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High trade execution latency detected"
          description: "95th percentile trade execution time is {{ \$value }}s"
          
      - alert: DatabaseConnectionFailure
        expr: postgres_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is unreachable"
          
      - alert: RedisConnectionFailure
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection failure"
          description: "Redis cache is unreachable"
          
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for 5 minutes"
          
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for 5 minutes"
          
      - alert: DiskSpaceLow
        expr: (node_filesystem_free_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Root disk space is below 15%"
          
      - alert: AIAgentFailure
        expr: up{job="ai-agents"} == 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "AI agents are down"
          description: "AI trading agents have been unavailable for more than 2 minutes"
          
      - alert: HighErrorRate
        expr: rate(trading_errors_total[5m]) > 0.1
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in trading system"
          description: "Error rate is {{ \$value }} errors per second"
          
      - alert: WebsocketConnectionsHigh
        expr: trading_websocket_connections > 100
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "High number of WebSocket connections"
          description: "{{ \$value }} WebSocket connections active"
EOF

# Configure AlertManager
echo "📢 Configuring AlertManager..."
cat > monitoring/alertmanager/alertmanager.yml << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@hive-trade.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://127.0.0.1:5001/'

- name: 'critical-alerts'
  webhook_configs:
  - url: 'http://127.0.0.1:5001/critical'
    send_resolved: true

- name: 'warning-alerts'
  webhook_configs:
  - url: 'http://127.0.0.1:5001/warning'
    send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

# Configure Loki for log aggregation
echo "📄 Configuring Loki for log aggregation..."
cat > monitoring/loki/loki.yml << EOF
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

ingester:
  wal:
    enabled: true
    dir: /loki/wal
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s
  max_transfer_retries: 0

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
EOF

# Configure Promtail for log collection
echo "📝 Configuring Promtail for log collection..."
cat > monitoring/promtail/promtail.yml << EOF
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: trading_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: trading-system
          __path__: /var/log/trading/*.log
    pipeline_stages:
      - match:
          selector: '{job="trading-system"}'
          stages:
          - regex:
              expression: '^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>\w+) - (?P<message>.*)$'
          - timestamp:
              source: timestamp
              format: '2006-01-02 15:04:05,000'
          - labels:
              level: level
              
  - job_name: docker_containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
EOF

# Start monitoring stack
echo "🔄 Starting monitoring infrastructure..."
docker-compose -f monitoring/docker-compose.monitoring.yml down
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Verify monitoring services
echo "🔍 Verifying monitoring services..."
services=(
    "http://localhost:9090/-/healthy"
    "http://localhost:3001/api/health" 
    "http://localhost:9093/-/healthy"
    "http://localhost:3100/ready"
)

for service in "${services[@]}"; do
    if curl -f -s "$service" > /dev/null; then
        echo "✅ Service at $service is healthy"
    else
        echo "❌ Service at $service is not responding"
    fi
done

echo ""
echo "🎉 Production monitoring setup complete!"
echo ""
echo "📊 Access your monitoring tools:"
echo "   • Grafana:      http://localhost:3001 (admin/admin123)"
echo "   • Prometheus:   http://localhost:9090"
echo "   • AlertManager: http://localhost:9093"
echo "   • Jaeger:       http://localhost:16686"
echo "   • Kibana:       http://localhost:5601"
echo ""
echo "🔧 Next steps:"
echo "   1. Import custom dashboards in Grafana"
echo "   2. Configure notification channels in AlertManager"
echo "   3. Set up log retention policies"
echo "   4. Configure backup schedules"
echo ""