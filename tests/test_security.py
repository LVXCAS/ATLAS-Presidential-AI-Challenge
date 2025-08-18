"""
Tests for the security system.
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from config.security import (
    SecurityManager, 
    SecurityConfig, 
    EncryptionManager,
    AuthenticationManager,
    RBACManager
)
from config.auth_middleware import (
    security_middleware,
    require_auth,
    require_permission,
    AuthenticationError,
    AuthorizationError
)


class TestEncryptionManager:
    """Test encryption functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        os.environ["TRADING_SYSTEM_MASTER_PASSWORD"] = "test_password_123"
        self.encryption_manager = EncryptionManager()
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        original_data = "sensitive_api_key_12345"
        encrypted = self.encryption_manager.encrypt(original_data)
        decrypted = self.encryption_manager.decrypt(encrypted)
        
        assert decrypted == original_data
        assert encrypted != original_data
    
    def test_encrypt_decrypt_dict(self):
        """Test dictionary encryption and decryption."""
        original_data = {
            "api_key": "test_key_123",
            "secret": "test_secret_456",
            "service": "test_service"
        }
        
        encrypted = self.encryption_manager.encrypt_dict(original_data)
        decrypted = self.encryption_manager.decrypt_dict(encrypted)
        
        assert decrypted == original_data
    
    def teardown_method(self):
        """Clean up test environment."""
        if "TRADING_SYSTEM_MASTER_PASSWORD" in os.environ:
            del os.environ["TRADING_SYSTEM_MASTER_PASSWORD"]


class TestAuthenticationManager:
    """Test authentication functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auth_manager = AuthenticationManager()
    
    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials."""
        session_token = self.auth_manager.authenticate("admin", "admin123")
        assert session_token is not None
        assert len(session_token) > 0
    
    def test_authenticate_invalid_user(self):
        """Test authentication with invalid credentials."""
        session_token = self.auth_manager.authenticate("admin", "wrong_password")
        assert session_token is None
    
    def test_validate_session(self):
        """Test session validation."""
        session_token = self.auth_manager.authenticate("admin", "admin123")
        session = self.auth_manager.validate_session(session_token)
        
        assert session is not None
        assert session["username"] == "admin"
        assert "admin" in session["roles"]
    
    def test_logout(self):
        """Test user logout."""
        session_token = self.auth_manager.authenticate("admin", "admin123")
        assert self.auth_manager.validate_session(session_token) is not None
        
        success = self.auth_manager.logout(session_token)
        assert success is True
        assert self.auth_manager.validate_session(session_token) is None


class TestRBACManager:
    """Test role-based access control."""
    
    def setup_method(self):
        """Set up test environment."""
        self.rbac_manager = RBACManager()
    
    def test_admin_permissions(self):
        """Test admin role permissions."""
        admin_roles = ["admin"]
        
        assert self.rbac_manager.check_permission(admin_roles, "system.manage")
        assert self.rbac_manager.check_permission(admin_roles, "trading.execute")
        assert self.rbac_manager.check_permission(admin_roles, "trading.view")
    
    def test_trader_permissions(self):
        """Test trader role permissions."""
        trader_roles = ["trader"]
        
        assert not self.rbac_manager.check_permission(trader_roles, "system.manage")
        assert self.rbac_manager.check_permission(trader_roles, "trading.execute")
        assert self.rbac_manager.check_permission(trader_roles, "trading.view")
    
    def test_viewer_permissions(self):
        """Test viewer role permissions."""
        viewer_roles = ["viewer"]
        
        assert not self.rbac_manager.check_permission(viewer_roles, "system.manage")
        assert not self.rbac_manager.check_permission(viewer_roles, "trading.execute")
        assert self.rbac_manager.check_permission(viewer_roles, "trading.view")
    
    def test_get_user_permissions(self):
        """Test getting all permissions for user roles."""
        admin_permissions = self.rbac_manager.get_user_permissions(["admin"])
        trader_permissions = self.rbac_manager.get_user_permissions(["trader"])
        
        assert len(admin_permissions) > len(trader_permissions)
        assert "system.manage" in admin_permissions
        assert "system.manage" not in trader_permissions


class TestSecurityManager:
    """Test the main security manager."""
    
    def setup_method(self):
        """Set up test environment."""
        os.environ["TRADING_SYSTEM_MASTER_PASSWORD"] = "test_password_123"
        
        # Use temporary file for secrets
        self.temp_dir = tempfile.mkdtemp()
        config = SecurityConfig(secrets_file_path=f"{self.temp_dir}/secrets.enc")
        self.security_manager = SecurityManager(config)
    
    def test_api_key_storage_retrieval(self):
        """Test API key storage and retrieval."""
        # Store API key
        self.security_manager.secret_manager.store_api_key(
            name="test_service",
            service="Test Service",
            api_key="test_api_key_123",
            secret_key="test_secret_key_456"
        )
        
        # Retrieve API key
        credentials = self.security_manager.get_broker_credentials("test_service")
        
        assert credentials is not None
        assert credentials["api_key"] == "test_api_key_123"
        assert credentials["secret_key"] == "test_secret_key_456"
        assert credentials["service"] == "Test Service"
    
    def test_user_authentication_authorization(self):
        """Test user authentication and authorization."""
        # Authenticate user
        session_token = self.security_manager.authenticate_user("admin", "admin123")
        assert session_token is not None
        
        # Test authorization
        assert self.security_manager.authorize_action(session_token, "system.manage")
        assert self.security_manager.authorize_action(session_token, "trading.execute")
    
    def teardown_method(self):
        """Clean up test environment."""
        if "TRADING_SYSTEM_MASTER_PASSWORD" in os.environ:
            del os.environ["TRADING_SYSTEM_MASTER_PASSWORD"]


class TestAuthMiddleware:
    """Test authentication middleware."""
    
    def setup_method(self):
        """Set up test environment."""
        os.environ["TRADING_SYSTEM_MASTER_PASSWORD"] = "test_password_123"
    
    def test_require_auth_decorator(self):
        """Test authentication decorator."""
        @require_auth
        def protected_function(session_token: str, current_user: dict):
            return {"user": current_user["username"]}
        
        # Get valid session token
        from config.auth_middleware import get_env_manager
        env_manager = get_env_manager()
        session_token = env_manager.secure_settings.authenticate_user("admin", "admin123")
        
        # Test with valid token
        result = protected_function(session_token=session_token)
        assert result["user"] == "admin"
        
        # Test with invalid token
        with pytest.raises(AuthenticationError):
            protected_function(session_token="invalid_token")
    
    def test_require_permission_decorator(self):
        """Test permission decorator."""
        @require_permission("system.manage")
        def admin_function(session_token: str):
            return {"status": "authorized"}
        
        # Get valid session tokens
        from config.auth_middleware import get_env_manager
        env_manager = get_env_manager()
        admin_token = env_manager.secure_settings.authenticate_user("admin", "admin123")
        trader_token = env_manager.secure_settings.authenticate_user("trader", "trader123")
        
        # Test with admin token (should work)
        result = admin_function(session_token=admin_token)
        assert result["status"] == "authorized"
        
        # Test with trader token (should fail)
        with pytest.raises(AuthorizationError):
            admin_function(session_token=trader_token)
    
    def teardown_method(self):
        """Clean up test environment."""
        if "TRADING_SYSTEM_MASTER_PASSWORD" in os.environ:
            del os.environ["TRADING_SYSTEM_MASTER_PASSWORD"]


if __name__ == "__main__":
    pytest.main([__file__])