"""
Authentication and Security Framework

Enterprise-grade security system with JWT authentication, role-based access control,
API rate limiting, encryption, and comprehensive audit logging for trading systems.
"""

import asyncio
import jwt
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from functools import wraps
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import sqlite3
from contextlib import contextmanager
import ipaddress
from collections import defaultdict, deque
import threading
import warnings
warnings.filterwarnings('ignore')

# FastAPI security components
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

class UserRole(Enum):
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(Enum):
    # Trading permissions
    EXECUTE_TRADES = "execute_trades"
    VIEW_POSITIONS = "view_positions"
    MANAGE_STRATEGIES = "manage_strategies"

    # System permissions
    ADMIN_SYSTEM = "admin_system"
    VIEW_SYSTEM_HEALTH = "view_system_health"
    MANAGE_USERS = "manage_users"

    # Data permissions
    ACCESS_MARKET_DATA = "access_market_data"
    ACCESS_HISTORICAL_DATA = "access_historical_data"
    EXPORT_DATA = "export_data"

    # Configuration permissions
    MODIFY_SETTINGS = "modify_settings"
    VIEW_LOGS = "view_logs"
    MANAGE_API_KEYS = "manage_api_keys"

@dataclass
class User:
    """User account representation"""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[UserRole]
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    ip_whitelist: List[str] = field(default_factory=list)
    session_timeout: int = 3600  # seconds

@dataclass
class APIKey:
    """API key representation"""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: Set[Permission]
    rate_limit: int = 1000  # requests per hour
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_level: str = "low"

class SecurityConfig:
    """Security configuration settings"""
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_REQUIRE_NUMBERS = True
    PASSWORD_REQUIRE_UPPERCASE = True
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # seconds
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())

class PasswordManager:
    """Secure password management with bcrypt"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
        """Validate password meets security requirements"""
        errors = []

        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters")

        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if SecurityConfig.PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

class EncryptionManager:
    """Data encryption and decryption"""

    def __init__(self, key: bytes = None):
        self.key = key or SecurityConfig.ENCRYPTION_KEY
        if isinstance(self.key, str):
            self.key = self.key.encode()
        self.fernet = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        json_data = json.dumps(data)
        return self.encrypt(json_data)

    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt and parse as dictionary"""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)

class JWTManager:
    """JWT token management"""

    @staticmethod
    def create_token(user_id: str, permissions: List[str], expires_delta: timedelta = None) -> str:
        """Create JWT token"""
        if expires_delta is None:
            expires_delta = timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS)

        expire = datetime.utcnow() + expires_delta
        payload = {
            "sub": user_id,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }

        return jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logging.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logging.warning(f"Invalid JWT token: {e}")
            return None

    @staticmethod
    def refresh_token(token: str) -> Optional[str]:
        """Refresh JWT token if valid"""
        payload = JWTManager.verify_token(token)
        if payload:
            # Create new token with same permissions
            return JWTManager.create_token(
                payload['sub'],
                payload['permissions']
            )
        return None

class RateLimiter:
    """API rate limiting with Redis backend"""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.request_counts = defaultdict(lambda: deque())

    def is_allowed(self, identifier: str, limit: int = None, window: int = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit"""
        limit = limit or SecurityConfig.RATE_LIMIT_REQUESTS
        window = window or SecurityConfig.RATE_LIMIT_WINDOW

        current_time = time.time()
        key = f"rate_limit:{identifier}"

        try:
            # Use Redis for distributed rate limiting
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - window)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window)
            results = pipe.execute()

            current_requests = results[1]
            allowed = current_requests < limit

            return allowed, {
                'allowed': allowed,
                'current_requests': current_requests,
                'limit': limit,
                'window': window,
                'reset_time': current_time + window
            }

        except Exception as e:
            logging.error(f"Rate limiter error: {e}")
            # Fallback to in-memory rate limiting
            return self._fallback_rate_limit(identifier, limit, window)

    def _fallback_rate_limit(self, identifier: str, limit: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Fallback in-memory rate limiting"""
        current_time = time.time()
        requests = self.request_counts[identifier]

        # Remove old requests outside the window
        while requests and requests[0] < current_time - window:
            requests.popleft()

        # Add current request
        requests.append(current_time)

        allowed = len(requests) <= limit

        return allowed, {
            'allowed': allowed,
            'current_requests': len(requests),
            'limit': limit,
            'window': window,
            'reset_time': current_time + window
        }

class IPWhitelistManager:
    """IP address whitelist management"""

    def __init__(self):
        self.whitelisted_ips = set()
        self.whitelisted_networks = []

    def add_ip(self, ip_address: str):
        """Add IP address to whitelist"""
        try:
            ip = ipaddress.ip_address(ip_address)
            self.whitelisted_ips.add(str(ip))
            logging.info(f"Added IP to whitelist: {ip}")
        except ValueError as e:
            logging.error(f"Invalid IP address: {ip_address} - {e}")

    def add_network(self, network: str):
        """Add network range to whitelist"""
        try:
            net = ipaddress.ip_network(network, strict=False)
            self.whitelisted_networks.append(net)
            logging.info(f"Added network to whitelist: {net}")
        except ValueError as e:
            logging.error(f"Invalid network: {network} - {e}")

    def is_allowed(self, ip_address: str) -> bool:
        """Check if IP address is whitelisted"""
        try:
            ip = ipaddress.ip_address(ip_address)

            # Check direct IP match
            if str(ip) in self.whitelisted_ips:
                return True

            # Check network ranges
            for network in self.whitelisted_networks:
                if ip in network:
                    return True

            return False

        except ValueError:
            logging.error(f"Invalid IP address for checking: {ip_address}")
            return False

class AuditLogger:
    """Security audit logging system"""

    def __init__(self, db_path: str = "security_audit.db"):
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Setup audit log database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    risk_level TEXT DEFAULT 'low'
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON security_events(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON security_events(user_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON security_events(event_type)
            """)

    def log_event(self, event: SecurityEvent):
        """Log security event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO security_events
                    (event_id, event_type, user_id, ip_address, user_agent, timestamp, details, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type,
                    event.user_id,
                    event.ip_address,
                    event.user_agent,
                    event.timestamp.isoformat(),
                    json.dumps(event.details),
                    event.risk_level
                ))

                logging.info(f"Security event logged: {event.event_type} - {event.event_id}")

        except Exception as e:
            logging.error(f"Failed to log security event: {e}")

    def get_events(self,
                   user_id: str = None,
                   event_type: str = None,
                   risk_level: str = None,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   limit: int = 100) -> List[SecurityEvent]:
        """Retrieve security events with filters"""

        query = "SELECT * FROM security_events WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)

                events = []
                for row in cursor.fetchall():
                    events.append(SecurityEvent(
                        event_id=row['event_id'],
                        event_type=row['event_type'],
                        user_id=row['user_id'],
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        details=json.loads(row['details']),
                        risk_level=row['risk_level']
                    ))

                return events

        except Exception as e:
            logging.error(f"Failed to retrieve security events: {e}")
            return []

class UserManager:
    """User account management"""

    def __init__(self, db_path: str = "users.db", encryption_manager: EncryptionManager = None):
        self.db_path = db_path
        self.encryption_manager = encryption_manager or EncryptionManager()
        self._setup_database()

    def _setup_database(self):
        """Setup user database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    permissions TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TEXT,
                    api_keys TEXT,
                    ip_whitelist TEXT,
                    session_timeout INTEGER DEFAULT 3600
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    rate_limit INTEGER DEFAULT 1000,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    expires_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

    def create_user(self, username: str, email: str, password: str, roles: List[UserRole]) -> User:
        """Create new user account"""
        # Validate password
        is_valid, errors = PasswordManager.validate_password_strength(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {', '.join(errors)}")

        # Hash password
        password_hash = PasswordManager.hash_password(password)

        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            permissions=self._get_permissions_for_roles(roles)
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO users
                (user_id, username, email, password_hash, roles, permissions,
                 is_active, is_verified, created_at, failed_login_attempts,
                 api_keys, ip_whitelist, session_timeout)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.user_id,
                user.username,
                user.email,
                user.password_hash,
                json.dumps([role.value for role in user.roles]),
                json.dumps([perm.value for perm in user.permissions]),
                user.is_active,
                user.is_verified,
                user.created_at.isoformat(),
                user.failed_login_attempts,
                json.dumps(user.api_keys),
                json.dumps(user.ip_whitelist),
                user.session_timeout
            ))

        logging.info(f"Created user: {username}")
        return user

    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[User]:
        """Authenticate user credentials"""
        user = self.get_user_by_username(username)

        if not user:
            logging.warning(f"Authentication failed - user not found: {username}")
            return None

        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logging.warning(f"Authentication failed - account locked: {username}")
            return None

        # Check IP whitelist if configured
        if user.ip_whitelist and ip_address not in user.ip_whitelist:
            logging.warning(f"Authentication failed - IP not whitelisted: {username} from {ip_address}")
            return None

        # Verify password
        if not PasswordManager.verify_password(password, user.password_hash):
            # Increment failed attempts
            self._increment_failed_attempts(user.user_id)
            logging.warning(f"Authentication failed - invalid password: {username}")
            return None

        # Reset failed attempts on successful login
        self._reset_failed_attempts(user.user_id)

        # Update last login
        user.last_login = datetime.utcnow()
        self._update_last_login(user.user_id)

        logging.info(f"User authenticated successfully: {username}")
        return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_user(row)

        except Exception as e:
            logging.error(f"Error getting user by username: {e}")

        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_user(row)

        except Exception as e:
            logging.error(f"Error getting user by ID: {e}")

        return None

    def create_api_key(self, user_id: str, name: str, permissions: List[Permission],
                      rate_limit: int = 1000, expires_days: int = None) -> Tuple[str, APIKey]:
        """Create API key for user"""
        # Generate API key
        api_key = f"hive_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # Create API key object
        api_key_obj = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=set(permissions),
            rate_limit=rate_limit,
            expires_at=expires_at
        )

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_keys
                (key_id, key_hash, name, user_id, permissions, rate_limit,
                 is_active, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key_obj.key_id,
                api_key_obj.key_hash,
                api_key_obj.name,
                api_key_obj.user_id,
                json.dumps([perm.value for perm in api_key_obj.permissions]),
                api_key_obj.rate_limit,
                api_key_obj.is_active,
                api_key_obj.created_at.isoformat(),
                api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None
            ))

        logging.info(f"Created API key for user {user_id}: {name}")
        return api_key, api_key_obj

    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return associated key object"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM api_keys
                    WHERE key_hash = ? AND is_active = TRUE
                """, (key_hash,))
                row = cursor.fetchone()

                if row:
                    # Check expiration
                    expires_at = None
                    if row['expires_at']:
                        expires_at = datetime.fromisoformat(row['expires_at'])
                        if datetime.utcnow() > expires_at:
                            logging.warning(f"API key expired: {row['key_id']}")
                            return None

                    # Update last used
                    conn.execute("""
                        UPDATE api_keys SET last_used = ? WHERE key_id = ?
                    """, (datetime.utcnow().isoformat(), row['key_id']))

                    return APIKey(
                        key_id=row['key_id'],
                        key_hash=row['key_hash'],
                        name=row['name'],
                        user_id=row['user_id'],
                        permissions=set(Permission(p) for p in json.loads(row['permissions'])),
                        rate_limit=row['rate_limit'],
                        is_active=row['is_active'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                        expires_at=expires_at
                    )

        except Exception as e:
            logging.error(f"Error verifying API key: {e}")

        return None

    def _get_permissions_for_roles(self, roles: List[UserRole]) -> Set[Permission]:
        """Get permissions for user roles"""
        permissions = set()

        for role in roles:
            if role == UserRole.ADMIN:
                permissions.update(Permission)  # Admin gets all permissions
            elif role == UserRole.TRADER:
                permissions.update([
                    Permission.EXECUTE_TRADES,
                    Permission.VIEW_POSITIONS,
                    Permission.MANAGE_STRATEGIES,
                    Permission.ACCESS_MARKET_DATA,
                    Permission.VIEW_SYSTEM_HEALTH
                ])
            elif role == UserRole.ANALYST:
                permissions.update([
                    Permission.VIEW_POSITIONS,
                    Permission.ACCESS_MARKET_DATA,
                    Permission.ACCESS_HISTORICAL_DATA,
                    Permission.EXPORT_DATA,
                    Permission.VIEW_SYSTEM_HEALTH
                ])
            elif role == UserRole.VIEWER:
                permissions.update([
                    Permission.VIEW_POSITIONS,
                    Permission.ACCESS_MARKET_DATA,
                    Permission.VIEW_SYSTEM_HEALTH
                ])
            elif role == UserRole.API_USER:
                permissions.update([
                    Permission.ACCESS_MARKET_DATA,
                    Permission.VIEW_POSITIONS
                ])

        return permissions

    def _increment_failed_attempts(self, user_id: str):
        """Increment failed login attempts"""
        with sqlite3.connect(self.db_path) as conn:
            # Get current attempts
            cursor = conn.execute(
                "SELECT failed_login_attempts FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if row:
                attempts = row[0] + 1
                locked_until = None

                # Lock account if too many failed attempts
                if attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
                    locked_until = datetime.utcnow() + timedelta(minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES)

                conn.execute("""
                    UPDATE users
                    SET failed_login_attempts = ?, locked_until = ?
                    WHERE user_id = ?
                """, (attempts, locked_until.isoformat() if locked_until else None, user_id))

    def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users
                SET failed_login_attempts = 0, locked_until = NULL
                WHERE user_id = ?
            """, (user_id,))

    def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users SET last_login = ? WHERE user_id = ?
            """, (datetime.utcnow().isoformat(), user_id))

    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            roles=[UserRole(r) for r in json.loads(row['roles'])],
            permissions=set(Permission(p) for p in json.loads(row['permissions'] or '[]')),
            is_active=bool(row['is_active']),
            is_verified=bool(row['is_verified']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
            failed_login_attempts=row['failed_login_attempts'],
            locked_until=datetime.fromisoformat(row['locked_until']) if row['locked_until'] else None,
            api_keys=json.loads(row['api_keys'] or '[]'),
            ip_whitelist=json.loads(row['ip_whitelist'] or '[]'),
            session_timeout=row['session_timeout']
        )

class SecurityManager:
    """Main security manager coordinating all security components"""

    def __init__(self, redis_client: redis.Redis = None):
        self.user_manager = UserManager()
        self.rate_limiter = RateLimiter(redis_client) if redis_client else None
        self.ip_whitelist = IPWhitelistManager()
        self.audit_logger = AuditLogger()
        self.encryption_manager = EncryptionManager()

    def authenticate_request(self, token: str = None, api_key: str = None,
                           ip_address: str = "", user_agent: str = "") -> Tuple[bool, Optional[User], Dict[str, Any]]:
        """Authenticate incoming request"""

        event_details = {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'authentication_method': 'token' if token else 'api_key'
        }

        try:
            if token:
                # JWT token authentication
                payload = JWTManager.verify_token(token)
                if not payload:
                    self._log_security_event("authentication_failed", None, ip_address, user_agent,
                                           {**event_details, 'reason': 'invalid_token'}, "medium")
                    return False, None, {'error': 'Invalid or expired token'}

                user = self.user_manager.get_user_by_id(payload['sub'])
                if not user or not user.is_active:
                    self._log_security_event("authentication_failed", payload['sub'], ip_address, user_agent,
                                           {**event_details, 'reason': 'inactive_user'}, "medium")
                    return False, None, {'error': 'User account inactive'}

                # Check rate limiting
                if self.rate_limiter:
                    allowed, rate_info = self.rate_limiter.is_allowed(f"user:{user.user_id}")
                    if not allowed:
                        self._log_security_event("rate_limit_exceeded", user.user_id, ip_address, user_agent,
                                               {**event_details, 'rate_info': rate_info}, "high")
                        return False, None, {'error': 'Rate limit exceeded', 'rate_info': rate_info}

                self._log_security_event("authentication_success", user.user_id, ip_address, user_agent, event_details)
                return True, user, {'permissions': [p.value for p in user.permissions]}

            elif api_key:
                # API key authentication
                api_key_obj = self.user_manager.verify_api_key(api_key)
                if not api_key_obj:
                    self._log_security_event("authentication_failed", None, ip_address, user_agent,
                                           {**event_details, 'reason': 'invalid_api_key'}, "medium")
                    return False, None, {'error': 'Invalid API key'}

                user = self.user_manager.get_user_by_id(api_key_obj.user_id)
                if not user or not user.is_active:
                    self._log_security_event("authentication_failed", api_key_obj.user_id, ip_address, user_agent,
                                           {**event_details, 'reason': 'inactive_user'}, "medium")
                    return False, None, {'error': 'User account inactive'}

                # Check API key rate limiting
                if self.rate_limiter:
                    allowed, rate_info = self.rate_limiter.is_allowed(
                        f"api_key:{api_key_obj.key_id}",
                        api_key_obj.rate_limit
                    )
                    if not allowed:
                        self._log_security_event("rate_limit_exceeded", user.user_id, ip_address, user_agent,
                                               {**event_details, 'rate_info': rate_info}, "high")
                        return False, None, {'error': 'API rate limit exceeded', 'rate_info': rate_info}

                self._log_security_event("authentication_success", user.user_id, ip_address, user_agent,
                                       {**event_details, 'api_key_id': api_key_obj.key_id})
                return True, user, {'permissions': [p.value for p in api_key_obj.permissions]}

            else:
                self._log_security_event("authentication_failed", None, ip_address, user_agent,
                                       {**event_details, 'reason': 'no_credentials'}, "low")
                return False, None, {'error': 'No authentication credentials provided'}

        except Exception as e:
            logging.error(f"Authentication error: {e}")
            self._log_security_event("authentication_error", None, ip_address, user_agent,
                                   {**event_details, 'error': str(e)}, "high")
            return False, None, {'error': 'Authentication system error'}

    def _log_security_event(self, event_type: str, user_id: Optional[str],
                          ip_address: str, user_agent: str,
                          details: Dict[str, Any], risk_level: str = "low"):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            details=details,
            risk_level=risk_level
        )

        self.audit_logger.log_event(event)

# FastAPI Security Middleware
class SecurityMiddleware:
    """Security middleware for FastAPI applications"""

    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager

    def create_security_dependency(self, required_permissions: List[Permission] = None):
        """Create FastAPI dependency for security"""

        def security_dependency(request: Request,
                              credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):

            # Extract authentication info
            token = credentials.credentials if credentials else None
            api_key = request.headers.get("X-API-Key")
            ip_address = request.client.host
            user_agent = request.headers.get("User-Agent", "")

            # Authenticate request
            authenticated, user, auth_info = self.security_manager.authenticate_request(
                token=token,
                api_key=api_key,
                ip_address=ip_address,
                user_agent=user_agent
            )

            if not authenticated:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=auth_info.get('error', 'Authentication failed'),
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Check permissions if required
            if required_permissions:
                user_permissions = set(Permission(p) for p in auth_info.get('permissions', []))
                required_perms = set(required_permissions)

                if not required_perms.issubset(user_permissions):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )

            # Add user to request state
            request.state.user = user
            request.state.auth_info = auth_info

            return user

        return security_dependency

# Example usage and setup
def create_secure_app() -> FastAPI:
    """Create FastAPI app with security middleware"""

    app = FastAPI(title="HiveTrading Secure API", version="1.0.0")

    # Security middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

    # Initialize security manager
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    security_manager = SecurityManager(redis_client)
    security_middleware = SecurityMiddleware(security_manager)

    # Security dependencies
    require_auth = security_middleware.create_security_dependency()
    require_trader = security_middleware.create_security_dependency([Permission.EXECUTE_TRADES])
    require_admin = security_middleware.create_security_dependency([Permission.ADMIN_SYSTEM])

    @app.post("/auth/login")
    async def login(username: str, password: str, request: Request):
        """User login endpoint"""
        ip_address = request.client.host
        user = security_manager.user_manager.authenticate_user(username, password, ip_address)

        if user:
            token = JWTManager.create_token(user.user_id, [p.value for p in user.permissions])
            return {"token": token, "user_id": user.user_id}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    @app.get("/auth/me")
    async def get_current_user(user: User = Depends(require_auth)):
        """Get current user info"""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "permissions": [p.value for p in user.permissions]
        }

    @app.post("/trading/execute")
    async def execute_trade(trade_data: dict, user: User = Depends(require_trader)):
        """Execute trade (requires trader permissions)"""
        return {"message": "Trade executed", "user": user.username}

    @app.get("/admin/users")
    async def list_users(user: User = Depends(require_admin)):
        """List users (admin only)"""
        return {"message": "Users list", "requested_by": user.username}

    return app

if __name__ == "__main__":
    # Example setup
    app = create_secure_app()

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")