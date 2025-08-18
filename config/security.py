"""
Security and API Key Management for LangGraph Trading System.

This module provides:
- Secure API key storage and retrieval
- Data encryption/decryption
- Environment-based configuration
- Basic authentication and RBAC framework
"""

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    encryption_key_env: str = Field(default="TRADING_SYSTEM_ENCRYPTION_KEY")
    master_password_env: str = Field(default="TRADING_SYSTEM_MASTER_PASSWORD")
    secrets_file_path: str = Field(default="config/secrets.enc")
    use_aws_secrets: bool = Field(default=False)
    aws_region: str = Field(default="us-east-1")
    aws_secret_name: str = Field(default="trading-system-secrets")


class UserRole(BaseModel):
    """User role definition for RBAC."""
    
    name: str
    permissions: List[str]
    description: str


class User(BaseModel):
    """User model for authentication."""
    
    username: str
    password_hash: str
    roles: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class APIKeyMetadata(BaseModel):
    """Metadata for API keys."""
    
    name: str
    service: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True


class EncryptionManager:
    """Handles encryption and decryption of sensitive data."""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.getenv("TRADING_SYSTEM_MASTER_PASSWORD")
        if not self.master_password:
            # Debug output to see what environment variables are available
            logger.debug(f"Environment variables: {dict(os.environ)}")
            raise ValueError("Master password must be provided or set in environment")
        
        self._fernet = self._create_fernet_key()
    
    def _create_fernet_key(self) -> Fernet:
        """Create Fernet encryption key from master password."""
        password = self.master_password.encode()
        salt = b'trading_system_salt'  # In production, use random salt stored securely
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        try:
            encrypted_data = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data."""
        json_data = json.dumps(data)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)


class SecretManager:
    """Manages secure storage and retrieval of API keys and secrets."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_manager = EncryptionManager()
        self._secrets_cache: Dict[str, Any] = {}
        self._api_key_metadata: Dict[str, APIKeyMetadata] = {}
        
        # Initialize secrets storage
        self._initialize_secrets_storage()
    
    def _initialize_secrets_storage(self):
        """Initialize secrets storage (local file or AWS Secrets Manager)."""
        if self.config.use_aws_secrets:
            self._initialize_aws_secrets()
        else:
            self._initialize_local_secrets()
    
    def _initialize_local_secrets(self):
        """Initialize local encrypted secrets file."""
        secrets_dir = os.path.dirname(self.config.secrets_file_path)
        if not os.path.exists(secrets_dir):
            os.makedirs(secrets_dir, mode=0o700)
        
        if not os.path.exists(self.config.secrets_file_path):
            # Create empty encrypted secrets file
            empty_secrets = {"api_keys": {}, "metadata": {}}
            self._save_local_secrets(empty_secrets)
            logger.info("Created new encrypted secrets file")
    
    def _initialize_aws_secrets(self):
        """Initialize AWS Secrets Manager (placeholder for future implementation)."""
        # TODO: Implement AWS Secrets Manager integration
        logger.warning("AWS Secrets Manager not yet implemented, falling back to local storage")
        self.config.use_aws_secrets = False
        self._initialize_local_secrets()
    
    def _load_local_secrets(self) -> Dict[str, Any]:
        """Load secrets from local encrypted file."""
        try:
            with open(self.config.secrets_file_path, 'r') as f:
                encrypted_data = f.read()
            
            if not encrypted_data.strip():
                return {"api_keys": {}, "metadata": {}}
            
            return self.encryption_manager.decrypt_dict(encrypted_data)
        except FileNotFoundError:
            return {"api_keys": {}, "metadata": {}}
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise
    
    def _save_local_secrets(self, secrets: Dict[str, Any]):
        """Save secrets to local encrypted file."""
        try:
            encrypted_data = self.encryption_manager.encrypt_dict(secrets)
            
            # Write to temporary file first, then rename for atomic operation
            temp_file = f"{self.config.secrets_file_path}.tmp"
            with open(temp_file, 'w') as f:
                f.write(encrypted_data)
            
            # On Windows, need to remove target file first
            if os.path.exists(self.config.secrets_file_path):
                os.remove(self.config.secrets_file_path)
            os.rename(temp_file, self.config.secrets_file_path)
            os.chmod(self.config.secrets_file_path, 0o600)  # Read/write for owner only
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise
    
    def store_api_key(self, name: str, service: str, api_key: str, 
                     secret_key: Optional[str] = None, expires_at: Optional[datetime] = None):
        """Store API key securely."""
        secrets = self._load_local_secrets()
        
        key_data = {
            "api_key": api_key,
            "secret_key": secret_key,
            "service": service,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None
        }
        
        secrets["api_keys"][name] = key_data
        
        # Store metadata
        metadata = APIKeyMetadata(
            name=name,
            service=service,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        # Convert datetime objects to ISO strings for JSON serialization
        metadata_dict = metadata.model_dump()
        metadata_dict["created_at"] = metadata_dict["created_at"].isoformat()
        if metadata_dict["expires_at"]:
            metadata_dict["expires_at"] = metadata_dict["expires_at"].isoformat()
        secrets["metadata"][name] = metadata_dict
        
        self._save_local_secrets(secrets)
        logger.info(f"Stored API key for {service} service")
    
    def get_api_key(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve API key securely."""
        secrets = self._load_local_secrets()
        key_data = secrets["api_keys"].get(name)
        
        if key_data:
            # Update last used timestamp
            key_data["last_used"] = datetime.utcnow().isoformat()
            secrets["api_keys"][name] = key_data
            self._save_local_secrets(secrets)
            
            logger.debug(f"Retrieved API key for {name}")
        
        return key_data
    
    def list_api_keys(self) -> List[APIKeyMetadata]:
        """List all stored API keys metadata."""
        secrets = self._load_local_secrets()
        metadata_list = []
        
        for name, metadata_dict in secrets.get("metadata", {}).items():
            metadata = APIKeyMetadata(**metadata_dict)
            metadata_list.append(metadata)
        
        return metadata_list
    
    def delete_api_key(self, name: str) -> bool:
        """Delete API key."""
        secrets = self._load_local_secrets()
        
        if name in secrets["api_keys"]:
            del secrets["api_keys"][name]
            if name in secrets.get("metadata", {}):
                del secrets["metadata"][name]
            
            self._save_local_secrets(secrets)
            logger.info(f"Deleted API key: {name}")
            return True
        
        return False


class AuthenticationManager:
    """Handles user authentication and session management."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default system users."""
        # Create default admin user
        admin_user = User(
            username="admin",
            password_hash=self._hash_password("admin123"),  # Change in production!
            roles=["admin", "trader", "viewer"],
            created_at=datetime.utcnow()
        )
        self.users["admin"] = admin_user
        
        # Create default trader user
        trader_user = User(
            username="trader",
            password_hash=self._hash_password("trader123"),  # Change in production!
            roles=["trader", "viewer"],
            created_at=datetime.utcnow()
        )
        self.users["trader"] = trader_user
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        user = self.users.get(username)
        if not user or not user.is_active:
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user.password_hash:
            return None
        
        # Create session
        session_token = base64.urlsafe_b64encode(os.urandom(32)).decode()
        self.sessions[session_token] = {
            "username": username,
            "roles": user.roles,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=8)
        }
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        logger.info(f"User {username} authenticated successfully")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token."""
        session = self.sessions.get(session_token)
        if not session:
            return None
        
        if datetime.utcnow() > session["expires_at"]:
            del self.sessions[session_token]
            return None
        
        return session
    
    def logout(self, session_token: str) -> bool:
        """Logout user by invalidating session."""
        if session_token in self.sessions:
            username = self.sessions[session_token]["username"]
            del self.sessions[session_token]
            logger.info(f"User {username} logged out")
            return True
        return False


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.roles = self._initialize_roles()
    
    def _initialize_roles(self) -> Dict[str, UserRole]:
        """Initialize default roles and permissions."""
        roles = {}
        
        # Admin role - full access
        roles["admin"] = UserRole(
            name="admin",
            permissions=[
                "system.manage",
                "users.manage",
                "trading.execute",
                "trading.view",
                "portfolio.manage",
                "portfolio.view",
                "settings.manage",
                "api_keys.manage",
                "logs.view"
            ],
            description="Full system administrator access"
        )
        
        # Trader role - trading operations
        roles["trader"] = UserRole(
            name="trader",
            permissions=[
                "trading.execute",
                "trading.view",
                "portfolio.view",
                "strategies.manage",
                "risk.manage"
            ],
            description="Trading operations and portfolio management"
        )
        
        # Viewer role - read-only access
        roles["viewer"] = UserRole(
            name="viewer",
            permissions=[
                "trading.view",
                "portfolio.view",
                "logs.view"
            ],
            description="Read-only access to trading data"
        )
        
        return roles
    
    def check_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role and required_permission in role.permissions:
                return True
        return False
    
    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """Get all permissions for user roles."""
        permissions = set()
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        return list(permissions)


class SecurityManager:
    """Main security manager that coordinates all security components."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.secret_manager = SecretManager(self.config)
        self.auth_manager = AuthenticationManager()
        self.rbac_manager = RBACManager()
        
        logger.info("Security manager initialized")
    
    def initialize_api_keys_from_env(self):
        """Initialize API keys from environment variables."""
        api_keys_config = [
            ("alpaca", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "Alpaca Trading"),
            ("polygon", "POLYGON_API_KEY", None, "Polygon Market Data"),
            ("openai", "OPENAI_API_KEY", None, "OpenAI API"),
            ("google", "GOOGLE_API_KEY", None, "Google Gemini API"),
            ("deepseek", "DEEPSEEK_API_KEY", None, "DeepSeek API"),
        ]
        
        for name, api_key_env, secret_key_env, service in api_keys_config:
            api_key = os.getenv(api_key_env)
            secret_key = os.getenv(secret_key_env) if secret_key_env else None
            
            if api_key:
                self.secret_manager.store_api_key(
                    name=name,
                    service=service,
                    api_key=api_key,
                    secret_key=secret_key
                )
                logger.info(f"Stored {service} API key")
    
    def get_broker_credentials(self, broker: str) -> Optional[Dict[str, Any]]:
        """Get broker credentials securely."""
        return self.secret_manager.get_api_key(broker)
    
    def get_ai_credentials(self, service: str) -> Optional[Dict[str, Any]]:
        """Get AI service credentials securely."""
        return self.secret_manager.get_api_key(service)
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        return self.auth_manager.authenticate(username, password)
    
    def authorize_action(self, session_token: str, required_permission: str) -> bool:
        """Authorize user action based on session and required permission."""
        session = self.auth_manager.validate_session(session_token)
        if not session:
            return False
        
        return self.rbac_manager.check_permission(session["roles"], required_permission)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.secret_manager.encryption_manager.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.secret_manager.encryption_manager.decrypt(encrypted_data)


# Global security manager instance (lazy initialization)
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get or create the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

# For backward compatibility
security_manager = None