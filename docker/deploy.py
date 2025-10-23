#!/usr/bin/env python3
"""
Hive Trade Docker Deployment Manager
Comprehensive deployment automation for the trading system
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import yaml
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class HiveTradeDeployer:
    """
    Advanced deployment manager for Hive Trade system
    """
    
    def __init__(self, config_file: str = "docker/deploy-config.yml"):
        self.config_file = config_file
        self.config = self.load_config()
        self.services_status = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        return {
            'environment': 'development',
            'services': {
                'core': ['postgres', 'redis', 'hive-trade-core'],
                'ai': ['ai-agents'],
                'web': ['web-dashboard', 'nginx'],
                'monitoring': ['prometheus', 'grafana'],
                'logging': ['elasticsearch', 'logstash', 'kibana'],
                'backup': ['backup']
            },
            'health_check': {
                'timeout': 300,
                'retry_interval': 10,
                'endpoints': {
                    'core': 'http://localhost:8001/health',
                    'dashboard': 'http://localhost:3000',
                    'grafana': 'http://localhost:3001',
                    'kibana': 'http://localhost:5601'
                }
            },
            'deployment': {
                'build_timeout': 600,
                'start_timeout': 180,
                'stop_timeout': 60,
                'parallel_builds': True
            }
        }
    
    def run_command(self, command: List[str], cwd: str = None, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        try:
            logger.info(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command, result.stderr)
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites for deployment"""
        logger.info("Checking deployment prerequisites...")
        
        prerequisites = [
            ('docker', ['docker', '--version']),
            ('docker-compose', ['docker-compose', '--version']),
            ('git', ['git', '--version'])
        ]
        
        missing = []
        
        for name, command in prerequisites:
            try:
                self.run_command(command, timeout=10)
                logger.info(f"[OK] {name} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"[X] {name} is not available")
                missing.append(name)
        
        if missing:
            logger.error(f"Missing prerequisites: {', '.join(missing)}")
            return False
        
        # Check Docker daemon
        try:
            self.run_command(['docker', 'info'], timeout=10)
            logger.info("[OK] Docker daemon is running")
        except subprocess.CalledProcessError:
            logger.error("[X] Docker daemon is not running")
            return False
        
        # Check available resources
        self.check_system_resources()
        
        return True
    
    def check_system_resources(self) -> None:
        """Check available system resources"""
        try:
            # Check disk space
            result = self.run_command(['df', '-h', '.'], timeout=10)
            logger.info("Disk space check:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
            
            # Check memory
            try:
                result = self.run_command(['free', '-h'], timeout=10)
                logger.info("Memory check:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
            except:
                logger.info("Memory check not available on this system")
                
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
    
    def setup_environment(self) -> None:
        """Setup environment variables and configuration"""
        logger.info("Setting up deployment environment...")
        
        # Create necessary directories
        dirs = ['logs', 'data', 'backups', 'config', 'ssl']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
        
        # Generate environment file if it doesn't exist
        env_file = '.env'
        if not os.path.exists(env_file):
            self.generate_env_file(env_file)
        else:
            logger.info(f"Using existing {env_file}")
    
    def generate_env_file(self, env_file: str) -> None:
        """Generate environment configuration file"""
        logger.info(f"Generating {env_file}...")
        
        env_vars = {
            'COMPOSE_PROJECT_NAME': 'hivetrade',
            'NODE_ENV': self.config.get('environment', 'production'),
            'DB_NAME': 'trading_db',
            'DB_USER': 'trader',
            'DB_PASSWORD': 'secure_password_123',
            'REDIS_PASSWORD': 'redis_password_123',
            'ALPACA_API_KEY': 'your_alpaca_api_key',
            'ALPACA_SECRET_KEY': 'your_alpaca_secret_key',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
            'SLACK_WEBHOOK_URL': '',
            'DISCORD_WEBHOOK_URL': ''
        }
        
        with open(env_file, 'w') as f:
            f.write(f"# Hive Trade Environment Configuration\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Generated {env_file} - please update API keys before deployment")
    
    def build_services(self, services: Optional[List[str]] = None, parallel: bool = True) -> None:
        """Build Docker services"""
        logger.info("Building Docker services...")
        
        build_args = ['docker-compose', 'build']
        
        if parallel:
            build_args.append('--parallel')
        
        if services:
            build_args.extend(services)
        
        try:
            timeout = self.config.get('deployment', {}).get('build_timeout', 600)
            self.run_command(build_args, timeout=timeout)
            logger.info("Build completed successfully")
        except subprocess.CalledProcessError:
            logger.error("Build failed")
            raise
    
    def start_services(self, service_groups: Optional[List[str]] = None) -> None:
        """Start services in dependency order"""
        logger.info("Starting services...")
        
        if not service_groups:
            service_groups = ['core', 'ai', 'web', 'monitoring', 'logging']
        
        services_config = self.config.get('services', {})
        
        for group in service_groups:
            if group in services_config:
                services = services_config[group]
                logger.info(f"Starting {group} services: {', '.join(services)}")
                
                self.run_command([
                    'docker-compose', 'up', '-d', '--remove-orphans'
                ] + services)
                
                # Wait between service groups
                if group == 'core':
                    logger.info("Waiting for core services to stabilize...")
                    time.sleep(30)
                else:
                    time.sleep(10)
        
        logger.info("All services started")
    
    def stop_services(self, services: Optional[List[str]] = None, remove_volumes: bool = False) -> None:
        """Stop Docker services"""
        logger.info("Stopping services...")
        
        stop_args = ['docker-compose', 'down']
        
        if remove_volumes:
            stop_args.append('-v')
            logger.warning("Removing volumes - all data will be lost!")
        
        if services:
            # Stop specific services
            self.run_command(['docker-compose', 'stop'] + services)
        else:
            # Stop all services
            timeout = self.config.get('deployment', {}).get('stop_timeout', 60)
            self.run_command(stop_args, timeout=timeout)
        
        logger.info("Services stopped")
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of all services"""
        logger.info("Performing health checks...")
        
        endpoints = self.config.get('health_check', {}).get('endpoints', {})
        timeout = self.config.get('health_check', {}).get('timeout', 300)
        retry_interval = self.config.get('health_check', {}).get('retry_interval', 10)
        
        health_status = {}
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service, endpoint in endpoints.items():
                if service not in health_status or not health_status[service]:
                    try:
                        response = requests.get(endpoint, timeout=5)
                        if response.status_code == 200:
                            health_status[service] = True
                            logger.info(f"[OK] {service} is healthy")
                        else:
                            health_status[service] = False
                            all_healthy = False
                    except requests.RequestException:
                        health_status[service] = False
                        all_healthy = False
            
            if all_healthy:
                logger.info("All services are healthy")
                break
            
            logger.info(f"Waiting for services to be healthy... ({int(time.time() - start_time)}s)")
            time.sleep(retry_interval)
        
        # Final status report
        unhealthy = [service for service, status in health_status.items() if not status]
        if unhealthy:
            logger.warning(f"Unhealthy services: {', '.join(unhealthy)}")
        
        return health_status
    
    def show_status(self) -> None:
        """Show status of all services"""
        logger.info("Checking service status...")
        
        try:
            result = self.run_command(['docker-compose', 'ps'])
            print("\nService Status:")
            print("=" * 80)
            print(result.stdout)
        except subprocess.CalledProcessError:
            logger.error("Failed to get service status")
    
    def show_logs(self, service: str = None, tail: int = 100) -> None:
        """Show service logs"""
        log_args = ['docker-compose', 'logs', '--tail', str(tail)]
        
        if service:
            log_args.append(service)
            logger.info(f"Showing logs for {service}")
        else:
            logger.info("Showing logs for all services")
        
        try:
            result = self.run_command(log_args)
            print(result.stdout)
        except subprocess.CalledProcessError:
            logger.error("Failed to get logs")
    
    def backup_system(self) -> None:
        """Trigger system backup"""
        logger.info("Triggering system backup...")
        
        try:
            self.run_command([
                'docker-compose', 'exec', '-T', 'backup', 
                '/app/backup.sh', 'backup'
            ])
            logger.info("Backup completed successfully")
        except subprocess.CalledProcessError:
            logger.error("Backup failed")
            raise
    
    def deploy_full_stack(self, skip_build: bool = False, skip_health: bool = False) -> None:
        """Deploy the complete Hive Trade system"""
        logger.info("Starting full stack deployment...")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return
        
        # Setup environment
        self.setup_environment()
        
        # Build services
        if not skip_build:
            self.build_services(parallel=self.config.get('deployment', {}).get('parallel_builds', True))
        
        # Start services
        self.start_services()
        
        # Health checks
        if not skip_health:
            health_status = self.check_health()
            if not all(health_status.values()):
                logger.warning("Some services are not healthy")
        
        # Show final status
        self.show_status()
        
        logger.info("Deployment completed!")
        logger.info("Access points:")
        logger.info("  - Trading Dashboard: http://localhost:3000")
        logger.info("  - API Endpoint: http://localhost:8001")
        logger.info("  - Monitoring: http://localhost:3001")
        logger.info("  - Logs: http://localhost:5601")

def main():
    """Main deployment CLI"""
    parser = argparse.ArgumentParser(description='Hive Trade Deployment Manager')
    parser.add_argument('--config', default='docker/deploy-config.yml', 
                       help='Deployment configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy full stack')
    deploy_parser.add_argument('--skip-build', action='store_true', 
                              help='Skip building Docker images')
    deploy_parser.add_argument('--skip-health', action='store_true', 
                              help='Skip health checks')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build services')
    build_parser.add_argument('services', nargs='*', help='Specific services to build')
    build_parser.add_argument('--no-parallel', action='store_true', 
                             help='Disable parallel builds')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start services')
    start_parser.add_argument('groups', nargs='*', help='Service groups to start')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop services')
    stop_parser.add_argument('services', nargs='*', help='Specific services to stop')
    stop_parser.add_argument('--remove-volumes', action='store_true',
                            help='Remove volumes (WARNING: data loss)')
    
    # Status command
    subparsers.add_parser('status', help='Show service status')
    
    # Health command
    subparsers.add_parser('health', help='Check service health')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show service logs')
    logs_parser.add_argument('service', nargs='?', help='Specific service')
    logs_parser.add_argument('--tail', type=int, default=100, help='Number of lines to show')
    
    # Backup command
    subparsers.add_parser('backup', help='Trigger system backup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize deployer
    deployer = HiveTradeDeployer(args.config)
    
    try:
        if args.command == 'deploy':
            deployer.deploy_full_stack(args.skip_build, args.skip_health)
        elif args.command == 'build':
            services = args.services if args.services else None
            parallel = not args.no_parallel
            deployer.build_services(services, parallel)
        elif args.command == 'start':
            groups = args.groups if args.groups else None
            deployer.start_services(groups)
        elif args.command == 'stop':
            services = args.services if args.services else None
            deployer.stop_services(services, args.remove_volumes)
        elif args.command == 'status':
            deployer.show_status()
        elif args.command == 'health':
            deployer.check_health()
        elif args.command == 'logs':
            deployer.show_logs(args.service, args.tail)
        elif args.command == 'backup':
            deployer.backup_system()
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()