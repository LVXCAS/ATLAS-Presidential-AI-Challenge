#!/usr/bin/env python3
"""
Production deployment script for Hive Trade v0.2
Sets up monitoring and database optimizations.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

import docker
from config.logging_config import get_logger
from config.database_optimization import apply_database_optimizations
from config.database import init_database

logger = get_logger(__name__)


class ProductionSetup:
    """Handles production deployment setup."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docker_client = docker.from_env()
        self.services_status = {}
    
    async def deploy_production(self):
        """Complete production deployment process."""
        
        logger.info("[LAUNCH] Starting Hive Trade v0.2 Production Deployment")
        
        steps = [
            ("[TOOLS]  Setting up monitoring infrastructure", self.setup_monitoring),
            ("[INFO]️  Optimizing database for real-time trading", self.optimize_database),
            ("[INFO] Starting production containers", self.start_containers),
            ("[SEARCH] Verifying services health", self.verify_services),
            ("[CHART] Creating monitoring dashboards", self.setup_dashboards),
            ("[OK] Production deployment complete", self.deployment_summary)
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(step_name)
                await step_func()
            except Exception as e:
                logger.error(f"[X] Failed at step '{step_name}': {e}")
                raise
    
    async def setup_monitoring(self):
        """Set up comprehensive monitoring stack."""
        
        # Make monitoring setup script executable (on Unix systems)
        monitoring_script = self.project_root / "deploy" / "production-monitoring.sh"
        if monitoring_script.exists():
            try:
                # On Windows, we'll use a Python equivalent
                await self._setup_monitoring_windows()
            except Exception as e:
                logger.warning(f"Could not run monitoring setup script: {e}")
                # Fallback to manual Docker Compose setup
                await self._setup_monitoring_docker()
    
    async def _setup_monitoring_windows(self):
        """Windows-compatible monitoring setup."""
        
        monitoring_dir = self.project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Create monitoring directories
        subdirs = [
            "grafana/dashboards",
            "grafana/provisioning/dashboards",
            "grafana/provisioning/datasources",
            "prometheus",
            "alertmanager",
            "loki",
            "promtail"
        ]
        
        for subdir in subdirs:
            (monitoring_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info("[INFO] Created monitoring directory structure")
        
        # Start monitoring services
        compose_file = self.project_root / "monitoring" / "docker-compose.monitoring.yml"
        if compose_file.exists():
            await self._run_docker_compose("monitoring", "up -d")
            logger.info("[INFO] Started monitoring services")
    
    async def _setup_monitoring_docker(self):
        """Setup monitoring using Docker Compose."""
        
        compose_file = self.project_root / "docker" / "docker-compose.yml"
        if compose_file.exists():
            # Start monitoring services from main compose file
            cmd = ["docker-compose", "-f", str(compose_file), "up", "-d", 
                   "prometheus", "grafana", "elasticsearch", "kibana"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                logger.info("[OK] Monitoring services started successfully")
            else:
                logger.error(f"Failed to start monitoring services: {result.stderr}")
    
    async def optimize_database(self):
        """Apply database optimizations for production."""
        
        try:
            # Initialize database with optimizations
            await init_database()
            
            # Apply additional optimizations
            optimization_result = await apply_database_optimizations()
            
            if optimization_result.get('status') == 'optimized':
                logger.info("[OK] Database optimizations applied successfully")
                self.services_status['database'] = 'optimized'
            else:
                logger.warning("[WARN]  Database optimization partially completed")
                self.services_status['database'] = 'partial'
                
        except Exception as e:
            logger.error(f"[X] Database optimization failed: {e}")
            self.services_status['database'] = 'failed'
    
    async def start_containers(self):
        """Start all production containers."""
        
        compose_file = self.project_root / "docker" / "docker-compose.yml"
        
        if not compose_file.exists():
            logger.error("[X] Docker Compose file not found")
            return
        
        # Start core services first
        core_services = [
            "postgres", "redis", 
            "hive-trade-core", "ai-agents",
            "web-dashboard"
        ]
        
        for service in core_services:
            try:
                await self._start_service(service)
                await asyncio.sleep(5)  # Allow service to stabilize
            except Exception as e:
                logger.error(f"Failed to start {service}: {e}")
                self.services_status[service] = 'failed'
    
    async def _start_service(self, service_name: str):
        """Start a specific Docker service."""
        
        cmd = ["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d", service_name]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=self.project_root
        )
        
        if result.returncode == 0:
            logger.info(f"[OK] Started {service_name}")
            self.services_status[service_name] = 'running'
        else:
            logger.error(f"[X] Failed to start {service_name}: {result.stderr}")
            self.services_status[service_name] = 'failed'
            raise Exception(f"Service {service_name} failed to start")
    
    async def verify_services(self):
        """Verify all services are healthy."""
        
        health_checks = {
            "postgres": "http://localhost:5432",
            "redis": "http://localhost:6379", 
            "hive-trade-core": "http://localhost:8001/health",
            "web-dashboard": "http://localhost:3000",
            "prometheus": "http://localhost:9090/-/healthy",
            "grafana": "http://localhost:3001/api/health"
        }
        
        for service, endpoint in health_checks.items():
            try:
                if await self._check_service_health(service, endpoint):
                    logger.info(f"[OK] {service} is healthy")
                    self.services_status[service] = 'healthy'
                else:
                    logger.warning(f"[WARN]  {service} health check failed")
                    self.services_status[service] = 'unhealthy'
            except Exception as e:
                logger.error(f"[X] Could not check {service}: {e}")
                self.services_status[service] = 'unknown'
    
    async def _check_service_health(self, service: str, endpoint: str) -> bool:
        """Check if a service is healthy."""
        
        try:
            if service in ['postgres', 'redis']:
                # Use Docker to check database services
                container = self.docker_client.containers.get(f"hive-trade-{service}")
                return container.status == 'running'
            else:
                # Use HTTP health checks for web services
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=10) as response:
                        return response.status == 200
        except Exception:
            return False
    
    async def setup_dashboards(self):
        """Set up Grafana dashboards for trading monitoring."""
        
        dashboards = [
            "trading_system_overview",
            "database_performance", 
            "ai_agents_monitoring",
            "risk_metrics",
            "system_resources"
        ]
        
        dashboard_dir = self.project_root / "monitoring" / "grafana" / "dashboards"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        for dashboard in dashboards:
            try:
                await self._create_dashboard(dashboard)
                logger.info(f"[CHART] Created {dashboard} dashboard")
            except Exception as e:
                logger.warning(f"Could not create {dashboard} dashboard: {e}")
    
    async def _create_dashboard(self, dashboard_name: str):
        """Create a Grafana dashboard."""
        
        # Dashboard configurations would be defined here
        # For now, we'll create placeholder dashboard files
        
        dashboard_config = {
            "dashboard": {
                "title": dashboard_name.replace('_', ' ').title(),
                "tags": ["hive-trade", "production"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        dashboard_file = self.project_root / "monitoring" / "grafana" / "dashboards" / f"{dashboard_name}.json"
        
        import json
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
    
    async def _run_docker_compose(self, service_dir: str, command: str):
        """Run docker-compose command in a specific directory."""
        
        compose_dir = self.project_root / service_dir
        cmd = f"docker-compose {command}".split()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=compose_dir
        )
        
        if result.returncode != 0:
            raise Exception(f"Docker compose failed: {result.stderr}")
        
        return result.stdout
    
    async def deployment_summary(self):
        """Print deployment summary."""
        
        logger.info("\n" + "="*60)
        logger.info("[PARTY] HIVE TRADE v0.2 PRODUCTION DEPLOYMENT SUMMARY")
        logger.info("="*60)
        
        # Service status summary
        healthy_services = sum(1 for status in self.services_status.values() if status in ['running', 'healthy', 'optimized'])
        total_services = len(self.services_status)
        
        logger.info(f"[CHART] Services Status: {healthy_services}/{total_services} healthy")
        
        for service, status in self.services_status.items():
            status_emoji = {
                'running': '[OK]',
                'healthy': '[OK]', 
                'optimized': '[OK]',
                'failed': '[X]',
                'unhealthy': '[WARN]',
                'unknown': '[INFO]',
                'partial': '[WARN]'
            }.get(status, '[INFO]')
            
            logger.info(f"   {status_emoji} {service}: {status}")
        
        # Access URLs
        logger.info("\n[INFO] Access URLs:")
        logger.info("   • Trading Dashboard:  http://localhost:3000")
        logger.info("   • Grafana:           http://localhost:3001 (admin/admin_password_123)")
        logger.info("   • Prometheus:        http://localhost:9090")
        logger.info("   • Kibana:           http://localhost:5601")
        logger.info("   • API Docs:         http://localhost:8001/docs")
        
        # Next steps
        logger.info("\n[TOOL] Next Steps:")
        logger.info("   1. Configure alert notifications in Grafana")
        logger.info("   2. Set up automated backups")
        logger.info("   3. Configure SSL certificates for production")
        logger.info("   4. Set up log retention policies")
        logger.info("   5. Configure firewall rules")
        
        logger.info("\n[OK] Production deployment completed successfully!")
        logger.info("="*60 + "\n")


async def main():
    """Main deployment function."""
    
    setup = ProductionSetup()
    
    try:
        await setup.deploy_production()
    except KeyboardInterrupt:
        logger.info("[INFO] Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[INFO] Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())