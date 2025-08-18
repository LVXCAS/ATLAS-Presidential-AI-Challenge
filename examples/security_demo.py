#!/usr/bin/env python3
"""
Security System Demonstration Script

This script demonstrates how to use the security features of the
LangGraph Trading System including:
- API key management
- User authentication
- Role-based authorization
- Data encryption
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.security import SecurityManager, SecurityConfig
from config.secure_config import EnvironmentManager
from config.auth_middleware import (
    login_user, 
    logout_user, 
    validate_session,
    SecureAPIEndpoints,
    require_auth,
    require_trader,
    require_admin
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_api_key_management():
    """Demonstrate API key management."""
    print("\n🔑 API Key Management Demo")
    print("=" * 50)
    
    # Set up security manager
    os.environ["TRADING_SYSTEM_MASTER_PASSWORD"] = "demo_password_123"
    security_manager = SecurityManager()
    
    # Store some demo API keys
    demo_keys = [
        ("alpaca_demo", "Alpaca Trading", "ALPACA_API_KEY_123", "ALPACA_SECRET_456"),
        ("polygon_demo", "Polygon Data", "POLYGON_API_KEY_789", None),
        ("openai_demo", "OpenAI API", "OPENAI_API_KEY_ABC", None),
    ]
    
    print("Storing demo API keys...")
    for name, service, api_key, secret_key in demo_keys:
        security_manager.secret_manager.store_api_key(
            name=name,
            service=service,
            api_key=api_key,
            secret_key=secret_key
        )
        print(f"✅ Stored {service} credentials")
    
    # List stored keys
    print("\nStored API keys:")
    api_keys = security_manager.secret_manager.list_api_keys()
    for key_metadata in api_keys:
        print(f"• {key_metadata.name} ({key_metadata.service}) - Created: {key_metadata.created_at}")
    
    # Retrieve a key
    print("\nRetrieving Alpaca credentials...")
    alpaca_creds = security_manager.get_broker_credentials("alpaca_demo")
    if alpaca_creds:
        print(f"✅ API Key: {alpaca_creds['api_key'][:10]}...")
        print(f"✅ Secret Key: {alpaca_creds['secret_key'][:10]}...")
    
    return security_manager


def demo_authentication():
    """Demonstrate user authentication."""
    print("\n🔐 Authentication Demo")
    print("=" * 50)
    
    # Test login with different users
    users_to_test = [
        ("admin", "admin123"),
        ("trader", "trader123"),
        ("admin", "wrong_password"),  # This should fail
    ]
    
    sessions = {}
    
    for username, password in users_to_test:
        print(f"\nTesting login: {username}")
        result = login_user(username, password)
        
        if result["success"]:
            print(f"✅ Login successful")
            print(f"   Session token: {result['session_token'][:20]}...")
            print(f"   Roles: {result['roles']}")
            print(f"   Expires: {result['expires_at']}")
            sessions[username] = result["session_token"]
        else:
            print(f"❌ Login failed: {result['error']}")
    
    return sessions


def demo_authorization(sessions):
    """Demonstrate role-based authorization."""
    print("\n🛡️ Authorization Demo")
    print("=" * 50)
    
    # Create secure API endpoints instance
    api = SecureAPIEndpoints()
    
    # Test different operations with different users
    operations = [
        ("get_user_profile", "admin", lambda token: api.get_user_profile(session_token=token)),
        ("get_user_profile", "trader", lambda token: api.get_user_profile(session_token=token)),
        ("execute_trade", "admin", lambda token: api.execute_trade(session_token=token, trade_data={"symbol": "AAPL", "qty": 100})),
        ("execute_trade", "trader", lambda token: api.execute_trade(session_token=token, trade_data={"symbol": "AAPL", "qty": 100})),
        ("manage_system", "admin", lambda token: api.manage_system(session_token=token, action="restart")),
        ("manage_system", "trader", lambda token: api.manage_system(session_token=token, action="restart")),  # Should fail
        ("view_portfolio", "admin", lambda token: api.view_portfolio(session_token=token)),
        ("view_portfolio", "trader", lambda token: api.view_portfolio(session_token=token)),
    ]
    
    for operation, user, func in operations:
        if user not in sessions:
            continue
            
        print(f"\nTesting {operation} as {user}:")
        try:
            result = func(sessions[user])
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Failed: {e}")


def demo_data_encryption():
    """Demonstrate data encryption."""
    print("\n🔒 Data Encryption Demo")
    print("=" * 50)
    
    # Set up environment manager
    env_manager = EnvironmentManager()
    
    # Test data to encrypt
    sensitive_data = [
        "Trading strategy parameters: momentum_threshold=0.75",
        "Portfolio allocation: AAPL=25%, GOOGL=30%, MSFT=20%, TSLA=25%",
        "Risk limits: max_daily_loss=5%, position_size_limit=10%"
    ]
    
    print("Encrypting sensitive trading data...")
    encrypted_data = []
    
    for i, data in enumerate(sensitive_data, 1):
        encrypted = env_manager.secure_settings.encrypt_data(data)
        encrypted_data.append(encrypted)
        print(f"✅ Data {i} encrypted: {encrypted[:50]}...")
    
    print("\nDecrypting data...")
    for i, encrypted in enumerate(encrypted_data, 1):
        decrypted = env_manager.secure_settings.decrypt_data(encrypted)
        print(f"✅ Data {i} decrypted: {decrypted}")


def demo_session_management(sessions):
    """Demonstrate session management."""
    print("\n⏰ Session Management Demo")
    print("=" * 50)
    
    # Validate sessions
    for user, token in sessions.items():
        print(f"\nValidating session for {user}:")
        session_info = validate_session(token)
        
        if session_info["valid"]:
            print(f"✅ Session valid")
            print(f"   Username: {session_info['username']}")
            print(f"   Roles: {session_info['roles']}")
            print(f"   Expires: {session_info['expires_at']}")
        else:
            print(f"❌ Session invalid")
    
    # Logout users
    print("\nLogging out users...")
    for user, token in sessions.items():
        result = logout_user(token)
        if result["success"]:
            print(f"✅ {user} logged out successfully")
        else:
            print(f"❌ Failed to logout {user}")
    
    # Validate sessions after logout
    print("\nValidating sessions after logout:")
    for user, token in sessions.items():
        session_info = validate_session(token)
        status = "valid" if session_info["valid"] else "invalid"
        print(f"• {user}: {status}")


def demo_environment_configuration():
    """Demonstrate environment-based configuration."""
    print("\n⚙️ Environment Configuration Demo")
    print("=" * 50)
    
    env_manager = EnvironmentManager()
    
    print(f"Current environment: {env_manager.environment}")
    print(f"Is production: {env_manager.is_production()}")
    print(f"Is development: {env_manager.is_development()}")
    
    # Show configuration validation
    print("\nConfiguration validation:")
    validation_results = env_manager.validate_configuration()
    for component, is_valid in validation_results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"  {component}: {status}")
    
    # Show different config sections
    configs = [
        ("Database", env_manager.get_database_config()),
        ("Redis", env_manager.get_redis_config()),
        ("Trading", env_manager.get_trading_config()),
        ("Risk", env_manager.get_risk_config()),
    ]
    
    for name, config in configs:
        print(f"\n{name} Configuration:")
        for key, value in config.items():
            # Mask sensitive values
            if "password" in key.lower() or "secret" in key.lower():
                value = "***masked***"
            print(f"  {key}: {value}")


def main():
    """Run all security demos."""
    print("🚀 LangGraph Trading System - Security Demo")
    print("=" * 60)
    
    try:
        # Demo 1: API Key Management
        security_manager = demo_api_key_management()
        
        # Demo 2: Authentication
        sessions = demo_authentication()
        
        # Demo 3: Authorization
        demo_authorization(sessions)
        
        # Demo 4: Data Encryption
        demo_data_encryption()
        
        # Demo 5: Session Management
        demo_session_management(sessions)
        
        # Demo 6: Environment Configuration
        demo_environment_configuration()
        
        print("\n🎉 Security demo completed successfully!")
        print("\n📝 Key Features Demonstrated:")
        print("• Secure API key storage with encryption")
        print("• User authentication with session management")
        print("• Role-based access control (RBAC)")
        print("• Data encryption/decryption")
        print("• Environment-based configuration")
        print("• Session validation and logout")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()