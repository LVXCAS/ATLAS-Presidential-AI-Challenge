"""
Authentication and authorization middleware for the trading system.
"""

from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import logging
from config.secure_config import get_env_manager

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class AuthorizationError(Exception):
    """Authorization failed."""
    pass


class SecurityMiddleware:
    """Security middleware for request authentication and authorization."""
    
    def __init__(self):
        self.env_manager = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _get_env_manager(self):
        """Get or create environment manager instance."""
        if self.env_manager is None:
            self.env_manager = get_env_manager()
        return self.env_manager
    
    def authenticate_request(self, session_token: str) -> Dict[str, Any]:
        """Authenticate request using session token."""
        if not session_token:
            raise AuthenticationError("No session token provided")
        
        session = self._get_env_manager().secure_settings._get_security_manager().auth_manager.validate_session(session_token)
        if not session:
            raise AuthenticationError("Invalid or expired session token")
        
        return session
    
    def authorize_request(self, session_token: str, required_permission: str) -> bool:
        """Authorize request for specific permission."""
        try:
            session = self.authenticate_request(session_token)
            return self._get_env_manager().secure_settings.authorize_action(session_token, required_permission)
        except AuthenticationError:
            return False
    
    def require_authentication(self, func: Callable) -> Callable:
        """Decorator to require authentication for function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_token = kwargs.get('session_token') or (args[0] if args else None)
            
            if not session_token:
                raise AuthenticationError("Session token required")
            
            try:
                session = self.authenticate_request(session_token)
                kwargs['current_user'] = session
                return func(*args, **kwargs)
            except AuthenticationError as e:
                logger.warning(f"Authentication failed for {func.__name__}: {e}")
                raise
        
        return wrapper
    
    def require_permission(self, permission: str) -> Callable:
        """Decorator to require specific permission for function calls."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                session_token = kwargs.get('session_token') or (args[0] if args else None)
                
                if not session_token:
                    raise AuthenticationError("Session token required")
                
                if not self.authorize_request(session_token, permission):
                    raise AuthorizationError(f"Permission '{permission}' required")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


# Global security middleware instance
security_middleware = SecurityMiddleware()


# Convenience decorators
def require_auth(func: Callable) -> Callable:
    """Require authentication for function."""
    return security_middleware.require_authentication(func)


def require_permission(permission: str) -> Callable:
    """Require specific permission for function."""
    return security_middleware.require_permission(permission)


def require_admin(func: Callable) -> Callable:
    """Require admin permission for function."""
    return require_permission("system.manage")(func)


def require_trader(func: Callable) -> Callable:
    """Require trader permission for function."""
    return require_permission("trading.execute")(func)


def require_viewer(func: Callable) -> Callable:
    """Require viewer permission for function."""
    return require_permission("trading.view")(func)


# Example usage functions
class SecureAPIEndpoints:
    """Example secure API endpoints demonstrating authentication and authorization."""
    
    @require_auth
    def get_user_profile(self, session_token: str, current_user: Dict[str, Any]) -> Dict[str, Any]:
        """Get current user profile (requires authentication)."""
        return {
            "username": current_user["username"],
            "roles": current_user["roles"],
            "permissions": get_env_manager().secure_settings._get_security_manager().rbac_manager.get_user_permissions(current_user["roles"])
        }
    
    @require_trader
    def execute_trade(self, session_token: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade (requires trader permission)."""
        logger.info(f"Trade execution authorized for session: {session_token}")
        return {"status": "authorized", "trade_data": trade_data}
    
    @require_admin
    def manage_system(self, session_token: str, action: str) -> Dict[str, Any]:
        """Manage system (requires admin permission)."""
        logger.info(f"System management action '{action}' authorized for session: {session_token}")
        return {"status": "authorized", "action": action}
    
    @require_viewer
    def view_portfolio(self, session_token: str) -> Dict[str, Any]:
        """View portfolio (requires viewer permission)."""
        return {"status": "authorized", "data": "portfolio_data"}


# Example authentication helper functions
def login_user(username: str, password: str) -> Dict[str, Any]:
    """Login user and return session information."""
    try:
        session_token = get_env_manager().secure_settings.authenticate_user(username, password)
        if session_token:
            session = get_env_manager().secure_settings._get_security_manager().auth_manager.validate_session(session_token)
            return {
                "success": True,
                "session_token": session_token,
                "username": session["username"],
                "roles": session["roles"],
                "expires_at": session["expires_at"].isoformat()
            }
        else:
            return {"success": False, "error": "Invalid credentials"}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return {"success": False, "error": "Login failed"}


def logout_user(session_token: str) -> Dict[str, Any]:
    """Logout user and invalidate session."""
    try:
        success = get_env_manager().secure_settings._get_security_manager().auth_manager.logout(session_token)
        return {"success": success}
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        return {"success": False, "error": "Logout failed"}


def validate_session(session_token: str) -> Dict[str, Any]:
    """Validate session token."""
    try:
        session = security_middleware.authenticate_request(session_token)
        return {
            "valid": True,
            "username": session["username"],
            "roles": session["roles"],
            "expires_at": session["expires_at"].isoformat()
        }
    except AuthenticationError:
        return {"valid": False}