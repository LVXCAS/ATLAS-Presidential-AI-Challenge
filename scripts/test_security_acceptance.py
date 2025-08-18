#!/usr/bin/env python3
"""
Security Acceptance Test Script

This script tests all the acceptance criteria for task 1.3:
- API keys encrypted
- Environment configs working
- Secure API key storage (local vault)
- Encryption for sensitive data
- Environment-based configuration
- Basic authentication and RBAC framework
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.security import get_security_manager
from config.secure_config import get_env_manager
from config.auth_middleware import login_user, logout_user, validate_session
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_key_encryption():
    """Test that API keys are encrypted in storage."""
    print("🔐 Testing API Key Encryption...")
    
    security_manager = get_security_manager()
    
    # Store a test API key
    test_api_key = "test_encrypted_key_12345"
    security_manager.secret_manager.store_api_key(
        name="test_encryption",
        service="Test Encryption Service",
        api_key=test_api_key,
        secret_key="test_secret_67890"
    )
    
    # Check that the key is stored encrypted
    secrets_file = security_manager.secret_manager.config.secrets_file_path
    if os.path.exists(secrets_file):
        with open(secrets_file, 'r') as f:
            encrypted_content = f.read()
        
        # Verify the original key is not in the file (it's encrypted)
        if test_api_key not in encrypted_content:
            print("✅ API keys are properly encrypted in storage")
            return True
        else:
            print("❌ API keys are not encrypted (found in plaintext)")
            return False
    else:
        print("❌ Secrets file not found")
        return False


def test_environment_configuration():
    """Test environment-based configuration."""
    print("🌍 Testing Environment-Based Configuration...")
    
    env_manager = get_env_manager()
    
    # Test environment detection
    environment = env_manager.environment
    is_dev = env_manager.is_development()
    is_prod = env_manager.is_production()
    
    print(f"   Environment: {environment}")
    print(f"   Is Development: {is_dev}")
    print(f"   Is Production: {is_prod}")
    
    # Test configuration loading
    db_config = env_manager.get_database_config()
    trading_config = env_manager.get_trading_config()
    risk_config = env_manager.get_risk_config()
    
    if all([db_config, trading_config, risk_config]):
        print("✅ Environment-based configuration working")
        return True
    else:
        print("❌ Environment-based configuration failed")
        return False


def test_secure_api_key_storage():
    """Test secure API key storage and retrieval."""
    print("🔑 Testing Secure API Key Storage...")
    
    security_manager = get_security_manager()
    
    # Store multiple API keys
    test_keys = [
        ("alpaca_test", "Alpaca Test", "ALPACA_KEY_123", "ALPACA_SECRET_456"),
        ("polygon_test", "Polygon Test", "POLYGON_KEY_789", None),
        ("openai_test", "OpenAI Test", "OPENAI_KEY_ABC", None),
    ]
    
    stored_count = 0
    for name, service, api_key, secret_key in test_keys:
        try:
            security_manager.secret_manager.store_api_key(
                name=name,
                service=service,
                api_key=api_key,
                secret_key=secret_key
            )
            stored_count += 1
        except Exception as e:
            print(f"   Failed to store {name}: {e}")
    
    # Retrieve and verify
    retrieved_count = 0
    for name, service, expected_api_key, expected_secret_key in test_keys:
        try:
            credentials = security_manager.get_broker_credentials(name)
            if credentials and credentials.get("api_key") == expected_api_key:
                retrieved_count += 1
        except Exception as e:
            print(f"   Failed to retrieve {name}: {e}")
    
    if stored_count == len(test_keys) and retrieved_count == len(test_keys):
        print("✅ Secure API key storage working")
        return True
    else:
        print(f"❌ API key storage failed (stored: {stored_count}, retrieved: {retrieved_count})")
        return False


def test_data_encryption():
    """Test encryption for sensitive data."""
    print("🔒 Testing Data Encryption...")
    
    env_manager = get_env_manager()
    
    # Test data to encrypt
    sensitive_data = [
        "Trading strategy: momentum_threshold=0.75",
        "Portfolio: AAPL=25%, GOOGL=30%",
        "API_KEY_SENSITIVE_12345"
    ]
    
    encryption_success = 0
    for data in sensitive_data:
        try:
            # Encrypt
            encrypted = env_manager.secure_settings.encrypt_data(data)
            
            # Verify it's different from original
            if encrypted != data:
                # Decrypt and verify
                decrypted = env_manager.secure_settings.decrypt_data(encrypted)
                if decrypted == data:
                    encryption_success += 1
        except Exception as e:
            print(f"   Encryption failed for data: {e}")
    
    if encryption_success == len(sensitive_data):
        print("✅ Data encryption working")
        return True
    else:
        print(f"❌ Data encryption failed ({encryption_success}/{len(sensitive_data)})")
        return False


def test_authentication_framework():
    """Test basic authentication framework."""
    print("🔐 Testing Authentication Framework...")
    
    # Test user authentication
    test_users = [
        ("admin", "admin123", True),
        ("trader", "trader123", True),
        ("admin", "wrong_password", False),
        ("nonexistent", "password", False)
    ]
    
    auth_success = 0
    sessions = []
    
    for username, password, should_succeed in test_users:
        result = login_user(username, password)
        
        if should_succeed and result["success"]:
            auth_success += 1
            if "session_token" in result:
                sessions.append(result["session_token"])
        elif not should_succeed and not result["success"]:
            auth_success += 1
    
    # Test session validation
    session_validation_success = 0
    for session_token in sessions:
        session_info = validate_session(session_token)
        if session_info["valid"]:
            session_validation_success += 1
    
    if auth_success == len(test_users) and session_validation_success == len(sessions):
        print("✅ Authentication framework working")
        return True
    else:
        print(f"❌ Authentication failed (auth: {auth_success}/{len(test_users)}, sessions: {session_validation_success}/{len(sessions)})")
        return False


def test_rbac_framework():
    """Test Role-Based Access Control framework."""
    print("🛡️ Testing RBAC Framework...")
    
    # Test different user roles and permissions
    test_cases = [
        ("admin", "admin123", "system.manage", True),
        ("admin", "admin123", "trading.execute", True),
        ("trader", "trader123", "trading.execute", True),
        ("trader", "trader123", "system.manage", False),
    ]
    
    rbac_success = 0
    
    for username, password, permission, should_have_permission in test_cases:
        # Login user
        login_result = login_user(username, password)
        if not login_result["success"]:
            continue
        
        session_token = login_result["session_token"]
        
        # Test permission
        env_manager = get_env_manager()
        has_permission = env_manager.secure_settings.authorize_action(session_token, permission)
        
        if has_permission == should_have_permission:
            rbac_success += 1
        
        # Logout
        logout_user(session_token)
    
    if rbac_success == len(test_cases):
        print("✅ RBAC framework working")
        return True
    else:
        print(f"❌ RBAC failed ({rbac_success}/{len(test_cases)})")
        return False


def main():
    """Run all acceptance tests."""
    print("🧪 Security System Acceptance Tests")
    print("=" * 60)
    print("Testing Task 1.3 Acceptance Criteria:")
    print("- API keys encrypted")
    print("- Environment configs working")
    print("- Secure API key storage (local vault)")
    print("- Encryption for sensitive data")
    print("- Environment-based configuration")
    print("- Basic authentication and RBAC framework")
    print("=" * 60)
    
    tests = [
        ("API Key Encryption", test_api_key_encryption),
        ("Environment Configuration", test_environment_configuration),
        ("Secure API Key Storage", test_secure_api_key_storage),
        ("Data Encryption", test_data_encryption),
        ("Authentication Framework", test_authentication_framework),
        ("RBAC Framework", test_rbac_framework),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"   Test failed: {test_name}")
        except Exception as e:
            print(f"   Test error: {test_name} - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All acceptance criteria met!")
        print("\n✅ Task 1.3 Implementation Complete:")
        print("   • Secure API key storage implemented")
        print("   • Data encryption working")
        print("   • Environment-based configuration active")
        print("   • Authentication and RBAC framework operational")
        return True
    else:
        print("❌ Some acceptance criteria not met")
        return False


if __name__ == "__main__":
    # Set master password for testing
    os.environ["TRADING_SYSTEM_MASTER_PASSWORD"] = "development_master_password_123"
    
    success = main()
    sys.exit(0 if success else 1)