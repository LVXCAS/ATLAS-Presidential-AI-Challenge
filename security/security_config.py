"""
Hive Trade Security Configuration Manager
Comprehensive security hardening and configuration system
"""

import os
import sys
import json
import yaml
import secrets
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityMetrics:
    """Security configuration metrics"""
    password_strength_score: float
    encryption_enabled: bool
    ssl_configured: bool
    firewall_rules_count: int
    api_security_score: float
    database_security_score: float
    container_security_score: float
    overall_security_rating: str

class SecurityConfigManager:
    """
    Advanced security configuration and hardening system
    """
    
    def __init__(self, config_dir: str = "config/security"):
        self.config_dir = config_dir
        self.security_config = {}
        self.encryption_key = None
        os.makedirs(config_dir, exist_ok=True)
    
    def generate_secure_passwords(self, count: int = 10) -> Dict[str, str]:
        """Generate secure passwords for system components"""
        logger.info(f"Generating {count} secure passwords...")
        
        passwords = {}
        components = [
            'db_password',
            'redis_password', 
            'jwt_secret',
            'api_key',
            'admin_password',
            'grafana_admin_password',
            'elasticsearch_password',
            'backup_encryption_key',
            'ssl_cert_password',
            'webhook_secret'
        ]
        
        for i, component in enumerate(components[:count]):
            # Generate cryptographically secure password
            password = secrets.token_urlsafe(32)
            passwords[component] = password
            
        logger.info("Secure passwords generated")
        return passwords
    
    def create_encryption_key(self) -> bytes:
        """Create encryption key for sensitive data"""
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key()
        return self.encryption_key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive configuration data"""
        if not self.encryption_key:
            self.create_encryption_key()
        
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive configuration data"""
        if not self.encryption_key:
            raise ValueError("Encryption key not available")
        
        fernet = Fernet(self.encryption_key)
        decoded_data = base64.b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(decoded_data)
        return decrypted_data.decode()
    
    def generate_ssl_certificates(self, domain: str = "hive-trade.local") -> Dict[str, str]:
        """Generate self-signed SSL certificates for development"""
        logger.info(f"Generating SSL certificates for {domain}...")
        
        ssl_dir = os.path.join(self.config_dir, "ssl")
        os.makedirs(ssl_dir, exist_ok=True)
        
        cert_file = os.path.join(ssl_dir, "cert.pem")
        key_file = os.path.join(ssl_dir, "key.pem")
        
        # Generate private key and certificate
        openssl_cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", key_file, "-out", cert_file,
            "-days", "365", "-nodes",
            "-subj", f"/C=US/ST=State/L=City/O=HiveTrade/CN={domain}"
        ]
        
        try:
            subprocess.run(openssl_cmd, check=True, capture_output=True)
            logger.info("SSL certificates generated successfully")
            
            return {
                "cert_file": cert_file,
                "key_file": key_file,
                "domain": domain
            }
        except subprocess.CalledProcessError as e:
            logger.warning(f"OpenSSL not available, using dummy certificates: {e}")
            
            # Create dummy certificate files
            with open(cert_file, 'w') as f:
                f.write("# Dummy certificate file - replace with real SSL certificate\n")
            
            with open(key_file, 'w') as f:
                f.write("# Dummy private key file - replace with real SSL key\n")
            
            return {
                "cert_file": cert_file,
                "key_file": key_file,
                "domain": domain,
                "warning": "Dummy certificates created - replace with real SSL certificates"
            }
    
    def configure_api_security(self) -> Dict[str, Any]:
        """Configure API security settings"""
        logger.info("Configuring API security...")
        
        api_config = {
            "authentication": {
                "jwt_enabled": True,
                "jwt_algorithm": "HS256",
                "jwt_expiration_hours": 24,
                "require_api_key": True,
                "api_key_header": "X-API-Key"
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "burst_limit": 50
            },
            "cors": {
                "enabled": True,
                "allowed_origins": [
                    "http://localhost:3000",
                    "https://hive-trade.local"
                ],
                "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
                "allowed_headers": ["Content-Type", "Authorization", "X-API-Key"]
            },
            "input_validation": {
                "max_request_size_mb": 10,
                "sanitize_inputs": True,
                "validate_content_type": True
            },
            "security_headers": {
                "x_frame_options": "DENY",
                "x_content_type_options": "nosniff",
                "x_xss_protection": "1; mode=block",
                "strict_transport_security": "max-age=31536000; includeSubDomains",
                "content_security_policy": "default-src 'self'; script-src 'self' 'unsafe-inline'"
            }
        }
        
        return api_config
    
    def configure_database_security(self) -> Dict[str, Any]:
        """Configure database security settings"""
        logger.info("Configuring database security...")
        
        db_config = {
            "postgresql": {
                "ssl_mode": "require",
                "ssl_cert_file": "/etc/ssl/certs/postgresql.crt",
                "ssl_key_file": "/etc/ssl/private/postgresql.key",
                "authentication": "scram-sha-256",
                "max_connections": 100,
                "connection_timeout": 30,
                "statement_timeout": 300000,  # 5 minutes
                "idle_in_transaction_timeout": 60000,  # 1 minute
                "log_connections": True,
                "log_disconnections": True,
                "log_statement": "all",
                "log_min_duration_statement": 1000  # Log slow queries
            },
            "redis": {
                "require_password": True,
                "ssl_enabled": False,  # Enable for production
                "max_connections": 1000,
                "timeout": 5,
                "tcp_keepalive": 300,
                "maxmemory_policy": "allkeys-lru"
            },
            "backup_encryption": {
                "enabled": True,
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 90
            }
        }
        
        return db_config
    
    def configure_container_security(self) -> Dict[str, Any]:
        """Configure Docker container security"""
        logger.info("Configuring container security...")
        
        container_config = {
            "docker_daemon": {
                "user_namespace_enabled": True,
                "no_new_privileges": True,
                "seccomp_enabled": True,
                "apparmor_enabled": True
            },
            "container_defaults": {
                "read_only_root_fs": False,  # Some containers need write access
                "no_new_privileges": True,
                "drop_capabilities": ["ALL"],
                "add_capabilities": ["CHOWN", "DAC_OVERRIDE", "SETUID", "SETGID"],
                "user_id": 1000,
                "group_id": 1000
            },
            "network_security": {
                "enable_custom_bridge": True,
                "disable_inter_container_communication": False,
                "enable_network_encryption": True
            },
            "resource_limits": {
                "memory_limit": "4g",
                "cpu_limit": "2.0",
                "swap_limit": "1g",
                "max_files": 65536
            },
            "image_security": {
                "verify_signatures": True,
                "scan_vulnerabilities": True,
                "use_trusted_registries_only": True,
                "base_image_updates": True
            }
        }
        
        return container_config
    
    def configure_network_security(self) -> Dict[str, Any]:
        """Configure network security settings"""
        logger.info("Configuring network security...")
        
        network_config = {
            "firewall_rules": [
                {"port": 80, "protocol": "tcp", "description": "HTTP", "source": "0.0.0.0/0"},
                {"port": 443, "protocol": "tcp", "description": "HTTPS", "source": "0.0.0.0/0"},
                {"port": 8001, "protocol": "tcp", "description": "Trading API", "source": "172.20.0.0/16"},
                {"port": 3000, "protocol": "tcp", "description": "Dashboard", "source": "172.20.0.0/16"},
                {"port": 5432, "protocol": "tcp", "description": "PostgreSQL", "source": "172.20.0.0/16"},
                {"port": 6379, "protocol": "tcp", "description": "Redis", "source": "172.20.0.0/16"},
                {"port": 9090, "protocol": "tcp", "description": "Prometheus", "source": "127.0.0.1/32"},
                {"port": 3001, "protocol": "tcp", "description": "Grafana", "source": "172.20.0.0/16"},
                {"port": 5601, "protocol": "tcp", "description": "Kibana", "source": "172.20.0.0/16"}
            ],
            "intrusion_detection": {
                "enabled": True,
                "monitor_failed_logins": True,
                "block_suspicious_ips": True,
                "max_failed_attempts": 5,
                "block_duration_minutes": 60
            },
            "ddos_protection": {
                "enabled": True,
                "rate_limit_per_ip": 1000,
                "connection_limit_per_ip": 50,
                "enable_syn_flood_protection": True
            }
        }
        
        return network_config
    
    def configure_monitoring_security(self) -> Dict[str, Any]:
        """Configure monitoring and alerting security"""
        logger.info("Configuring monitoring security...")
        
        monitoring_config = {
            "prometheus": {
                "enable_basic_auth": True,
                "admin_user": "admin",
                "enable_https": True,
                "retention_days": 30,
                "scrape_interval": "15s"
            },
            "grafana": {
                "disable_signups": True,
                "require_https": True,
                "session_timeout_hours": 24,
                "enable_2fa": True,
                "password_policy": {
                    "min_length": 12,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_symbols": True
                }
            },
            "alerting": {
                "security_alerts": [
                    {
                        "name": "High Failed Login Attempts",
                        "condition": "failed_logins > 10",
                        "severity": "critical",
                        "channels": ["slack", "email"]
                    },
                    {
                        "name": "Unusual API Activity",
                        "condition": "api_requests_per_minute > 500",
                        "severity": "warning",
                        "channels": ["slack"]
                    },
                    {
                        "name": "Database Connection Anomaly",
                        "condition": "db_connections > 80",
                        "severity": "warning",
                        "channels": ["slack"]
                    }
                ]
            }
        }
        
        return monitoring_config
    
    def assess_password_strength(self, password: str) -> float:
        """Assess password strength (0-100 score)"""
        score = 0
        
        # Length scoring
        if len(password) >= 8:
            score += 25
        if len(password) >= 12:
            score += 15
        if len(password) >= 16:
            score += 10
        
        # Character variety
        if any(c.islower() for c in password):
            score += 10
        if any(c.isupper() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20
        
        return min(score, 100)
    
    def generate_security_report(self, config: Dict[str, Any]) -> SecurityMetrics:
        """Generate comprehensive security assessment"""
        logger.info("Generating security assessment...")
        
        # Assess password strength
        passwords = config.get('passwords', {})
        if passwords:
            password_scores = [self.assess_password_strength(pwd) for pwd in passwords.values()]
            avg_password_strength = sum(password_scores) / len(password_scores)
        else:
            avg_password_strength = 0
        
        # Check encryption
        encryption_enabled = config.get('encryption', {}).get('enabled', False)
        
        # Check SSL
        ssl_configured = 'ssl' in config and config['ssl'].get('cert_file')
        
        # Count firewall rules
        firewall_rules = config.get('network', {}).get('firewall_rules', [])
        firewall_rules_count = len(firewall_rules)
        
        # Calculate security scores
        api_config = config.get('api', {})
        api_security_score = 0
        if api_config.get('authentication', {}).get('jwt_enabled'):
            api_security_score += 25
        if api_config.get('rate_limiting', {}).get('enabled'):
            api_security_score += 25
        if api_config.get('cors', {}).get('enabled'):
            api_security_score += 25
        if api_config.get('security_headers'):
            api_security_score += 25
        
        db_config = config.get('database', {}).get('postgresql', {})
        db_security_score = 0
        if db_config.get('ssl_mode') == 'require':
            db_security_score += 30
        if db_config.get('authentication') == 'scram-sha-256':
            db_security_score += 30
        if db_config.get('log_connections'):
            db_security_score += 20
        if db_config.get('log_statement'):
            db_security_score += 20
        
        container_config = config.get('containers', {})
        container_security_score = 0
        if container_config.get('container_defaults', {}).get('no_new_privileges'):
            container_security_score += 25
        if container_config.get('network_security', {}).get('enable_custom_bridge'):
            container_security_score += 25
        if container_config.get('resource_limits'):
            container_security_score += 25
        if container_config.get('image_security', {}).get('scan_vulnerabilities'):
            container_security_score += 25
        
        # Overall rating
        overall_score = (avg_password_strength + api_security_score + db_security_score + container_security_score) / 4
        
        if overall_score >= 90:
            rating = "Excellent"
        elif overall_score >= 80:
            rating = "Good"
        elif overall_score >= 70:
            rating = "Acceptable"
        elif overall_score >= 60:
            rating = "Needs Improvement"
        else:
            rating = "Poor"
        
        return SecurityMetrics(
            password_strength_score=avg_password_strength,
            encryption_enabled=encryption_enabled,
            ssl_configured=ssl_configured,
            firewall_rules_count=firewall_rules_count,
            api_security_score=api_security_score,
            database_security_score=db_security_score,
            container_security_score=container_security_score,
            overall_security_rating=rating
        )
    
    def generate_security_config(self) -> Dict[str, Any]:
        """Generate comprehensive security configuration"""
        logger.info("Generating comprehensive security configuration...")
        
        # Generate secure passwords
        passwords = self.generate_secure_passwords()
        
        # Generate SSL certificates
        ssl_config = self.generate_ssl_certificates()
        
        # Configure all security components
        config = {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "passwords": passwords,
            "ssl": ssl_config,
            "api": self.configure_api_security(),
            "database": self.configure_database_security(),
            "containers": self.configure_container_security(),
            "network": self.configure_network_security(),
            "monitoring": self.configure_monitoring_security(),
            "encryption": {
                "enabled": True,
                "algorithm": "Fernet",
                "key_rotation_days": 90
            }
        }
        
        return config
    
    def save_security_config(self, config: Dict[str, Any], encrypt_passwords: bool = True) -> str:
        """Save security configuration to file"""
        config_file = os.path.join(self.config_dir, "security_config.yml")
        
        # Encrypt passwords if requested
        if encrypt_passwords and 'passwords' in config:
            self.create_encryption_key()
            encrypted_passwords = {}
            for key, password in config['passwords'].items():
                encrypted_passwords[key] = self.encrypt_sensitive_data(password)
            config['encrypted_passwords'] = encrypted_passwords
            del config['passwords']  # Remove plaintext passwords
            
            # Save encryption key separately
            key_file = os.path.join(self.config_dir, "encryption.key")
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            logger.info(f"Encryption key saved to {key_file}")
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Security configuration saved to {config_file}")
        return config_file
    
    def generate_security_report_text(self, metrics: SecurityMetrics, config: Dict[str, Any]) -> str:
        """Generate detailed security report"""
        
        report = f"""
HIVE TRADE SECURITY CONFIGURATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

SECURITY OVERVIEW:
{'*'*30}

Overall Security Rating:      {metrics.overall_security_rating}
Password Strength Score:      {metrics.password_strength_score:.1f}/100
Encryption Enabled:           {'Yes' if metrics.encryption_enabled else 'No'}
SSL Configured:               {'Yes' if metrics.ssl_configured else 'No'}
Firewall Rules Count:         {metrics.firewall_rules_count}

COMPONENT SECURITY SCORES:
{'*'*30}

API Security:                 {metrics.api_security_score:.1f}/100
Database Security:            {metrics.database_security_score:.1f}/100
Container Security:           {metrics.container_security_score:.1f}/100

DETAILED CONFIGURATION:
{'*'*30}

Authentication & Authorization:
  JWT Authentication:         {'Enabled' if config.get('api', {}).get('authentication', {}).get('jwt_enabled') else 'Disabled'}
  API Key Required:           {'Yes' if config.get('api', {}).get('authentication', {}).get('require_api_key') else 'No'}
  Session Timeout:            {config.get('api', {}).get('authentication', {}).get('jwt_expiration_hours', 24)} hours

Rate Limiting:
  Enabled:                    {'Yes' if config.get('api', {}).get('rate_limiting', {}).get('enabled') else 'No'}
  Requests per Minute:        {config.get('api', {}).get('rate_limiting', {}).get('requests_per_minute', 'N/A')}
  Burst Limit:                {config.get('api', {}).get('rate_limiting', {}).get('burst_limit', 'N/A')}

Database Security:
  SSL Mode:                   {config.get('database', {}).get('postgresql', {}).get('ssl_mode', 'Not configured')}
  Authentication:             {config.get('database', {}).get('postgresql', {}).get('authentication', 'Not configured')}
  Connection Logging:         {'Enabled' if config.get('database', {}).get('postgresql', {}).get('log_connections') else 'Disabled'}
  Statement Logging:          {'Enabled' if config.get('database', {}).get('postgresql', {}).get('log_statement') else 'Disabled'}

Container Security:
  No New Privileges:          {'Enabled' if config.get('containers', {}).get('container_defaults', {}).get('no_new_privileges') else 'Disabled'}
  Custom Bridge Network:      {'Enabled' if config.get('containers', {}).get('network_security', {}).get('enable_custom_bridge') else 'Disabled'}
  Resource Limits:            {'Configured' if config.get('containers', {}).get('resource_limits') else 'Not configured'}
  Vulnerability Scanning:     {'Enabled' if config.get('containers', {}).get('image_security', {}).get('scan_vulnerabilities') else 'Disabled'}

Network Security:
  Firewall Rules:             {len(config.get('network', {}).get('firewall_rules', []))} rules configured
  Intrusion Detection:        {'Enabled' if config.get('network', {}).get('intrusion_detection', {}).get('enabled') else 'Disabled'}
  DDoS Protection:           {'Enabled' if config.get('network', {}).get('ddos_protection', {}).get('enabled') else 'Disabled'}

SECURITY RECOMMENDATIONS:
{'*'*30}

High Priority:
"""
        
        recommendations = []
        
        if metrics.password_strength_score < 80:
            recommendations.append("  - Strengthen passwords (current average: {:.1f}/100)".format(metrics.password_strength_score))
        
        if not metrics.ssl_configured:
            recommendations.append("  - Configure SSL/TLS certificates")
        
        if metrics.api_security_score < 75:
            recommendations.append("  - Enhance API security configuration")
        
        if metrics.database_security_score < 80:
            recommendations.append("  - Improve database security settings")
        
        if metrics.firewall_rules_count < 5:
            recommendations.append("  - Add more granular firewall rules")
        
        if not recommendations:
            recommendations.append("  - No high-priority security issues identified")
        
        report += "\n".join(recommendations)
        
        report += f"""

Medium Priority:
  - Enable 2FA for admin accounts
  - Implement log aggregation and analysis
  - Set up automated security scanning
  - Configure backup encryption
  - Implement network segmentation

Low Priority:
  - Regular security audits
  - Penetration testing
  - Security awareness training
  - Incident response procedures
  - Compliance documentation

SECURITY CHECKLIST:
{'*'*30}

Authentication:
  {'[[OK]]' if config.get('api', {}).get('authentication', {}).get('jwt_enabled') else '[ ]'} JWT authentication enabled
  {'[[OK]]' if config.get('api', {}).get('authentication', {}).get('require_api_key') else '[ ]'} API key authentication
  {'[[OK]]' if config.get('monitoring', {}).get('grafana', {}).get('enable_2fa') else '[ ]'} Two-factor authentication

Encryption:
  {'[[OK]]' if metrics.ssl_configured else '[ ]'} SSL/TLS certificates
  {'[[OK]]' if metrics.encryption_enabled else '[ ]'} Data encryption at rest
  {'[[OK]]' if config.get('database', {}).get('backup_encryption', {}).get('enabled') else '[ ]'} Backup encryption

Network Security:
  {'[[OK]]' if len(config.get('network', {}).get('firewall_rules', [])) > 0 else '[ ]'} Firewall configured
  {'[[OK]]' if config.get('network', {}).get('intrusion_detection', {}).get('enabled') else '[ ]'} Intrusion detection
  {'[[OK]]' if config.get('containers', {}).get('network_security', {}).get('enable_custom_bridge') else '[ ]'} Network isolation

Monitoring:
  {'[[OK]]' if config.get('database', {}).get('postgresql', {}).get('log_connections') else '[ ]'} Database logging
  {'[[OK]]' if config.get('api', {}).get('rate_limiting', {}).get('enabled') else '[ ]'} Rate limiting
  {'[[OK]]' if len(config.get('monitoring', {}).get('alerting', {}).get('security_alerts', [])) > 0 else '[ ]'} Security alerting

NEXT STEPS:
{'*'*30}

1. Immediate Actions:
   - Review and update all default passwords
   - Enable SSL/TLS on all services
   - Configure firewall rules
   - Test backup and restore procedures

2. Short-term (1-2 weeks):
   - Implement comprehensive monitoring
   - Set up automated security scanning
   - Configure intrusion detection
   - Document incident response procedures

3. Long-term (1-3 months):
   - Conduct security audit
   - Implement advanced threat detection
   - Set up compliance monitoring
   - Regular security training

COMPLIANCE NOTES:
{'*'*30}

SOC 2 Type II:
  - Implement access controls: {'Partial' if metrics.api_security_score > 50 else 'Not implemented'}
  - Audit logging: {'Implemented' if config.get('database', {}).get('postgresql', {}).get('log_statement') else 'Not implemented'}
  - Encryption: {'Implemented' if metrics.encryption_enabled else 'Not implemented'}

ISO 27001:
  - Information security policy: Manual review required
  - Risk assessment: {'Basic' if metrics.overall_security_rating in ['Good', 'Excellent'] else 'Required'}
  - Security awareness: Training program needed

{'='*70}
Security Configuration Complete - Rating: {metrics.overall_security_rating}
"""
        
        return report

def main():
    """Main security configuration workflow"""
    
    print("HIVE TRADE SECURITY CONFIGURATION MANAGER")
    print("="*55)
    
    # Initialize security manager
    security_manager = SecurityConfigManager()
    
    print("1. Generating comprehensive security configuration...")
    
    # Generate security configuration
    config = security_manager.generate_security_config()
    print(f"   Generated configuration with {len(config.get('passwords', {}))} secure passwords")
    
    # Save configuration
    print("2. Saving security configuration...")
    config_file = security_manager.save_security_config(config, encrypt_passwords=True)
    print(f"   Configuration saved to: {config_file}")
    
    # Generate security assessment
    print("3. Performing security assessment...")
    metrics = security_manager.generate_security_report(config)
    print(f"   Overall security rating: {metrics.overall_security_rating}")
    
    # Generate detailed report
    print("4. Generating security report...")
    report = security_manager.generate_security_report_text(metrics, config)
    
    # Save report
    report_file = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSECURITY CONFIGURATION COMPLETE:")
    print(f"- Security Rating: {metrics.overall_security_rating}")
    print(f"- Password Strength: {metrics.password_strength_score:.1f}/100")
    print(f"- API Security: {metrics.api_security_score:.1f}/100")
    print(f"- Database Security: {metrics.database_security_score:.1f}/100")
    print(f"- Container Security: {metrics.container_security_score:.1f}/100")
    print(f"- Configuration saved: {config_file}")
    print(f"- Report saved: {report_file}")
    
    # Security status
    if metrics.overall_security_rating == "Excellent":
        print("\nSTATUS: Security configuration is excellent")
    elif metrics.overall_security_rating == "Good":
        print("\nSTATUS: Security configuration is good with minor improvements needed")
    elif metrics.overall_security_rating == "Acceptable":
        print("\nSTATUS: Security configuration needs improvement")
    else:
        print("\nSTATUS: Security configuration requires immediate attention")
    
    # Save metrics
    metrics_data = asdict(metrics)
    json_file = f"security_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"- Metrics saved: {json_file}")
    
    print(f"\nIMPORTANT NOTES:")
    print(f"- Encryption key saved to: config/security/encryption.key")
    print(f"- Update API keys in configuration before production deployment")
    print(f"- Replace dummy SSL certificates with real ones for production")
    print(f"- Review and customize firewall rules for your environment")

if __name__ == "__main__":
    main()