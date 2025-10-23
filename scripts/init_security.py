#!/usr/bin/env python3
"""
Security initialization script for the LangGraph Trading System.

This script initializes the security system, creates encrypted storage,
and sets up default users and permissions.
"""

import os
import sys
import getpass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.security import SecurityManager, SecurityConfig, get_security_manager
from config.secure_config import EnvironmentManager, get_env_manager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_security_system():
    """Initialize the security system."""
    print("[SECURE] LangGraph Trading System - Security Initialization")
    print("=" * 60)
    
    # Check if master password is set
    master_password = os.getenv("TRADING_SYSTEM_MASTER_PASSWORD")
    if not master_password:
        print("\n[WARN]  Master password not found in environment variables.")
        print("Please set TRADING_SYSTEM_MASTER_PASSWORD in your .env file")
        
        # Prompt for master password
        while True:
            password1 = getpass.getpass("Enter master password: ")
            password2 = getpass.getpass("Confirm master password: ")
            
            if password1 == password2:
                if len(password1) < 12:
                    print("[X] Password must be at least 12 characters long")
                    continue
                master_password = password1
                break
            else:
                print("[X] Passwords don't match. Please try again.")
        
        # Update .env file
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, "a") as f:
                f.write(f"\nTRADING_SYSTEM_MASTER_PASSWORD={master_password}\n")
        else:
            with open(env_file, "w") as f:
                f.write(f"TRADING_SYSTEM_MASTER_PASSWORD={master_password}\n")
        
        print("[OK] Master password saved to .env file")
    
    try:
        # Initialize security manager
        print("\n[TOOL] Initializing security manager...")
        security_manager = get_security_manager()
        
        # Initialize environment manager
        print("[TOOL] Initializing environment manager...")
        env_manager = get_env_manager()
        
        # Validate configuration
        print("[TOOL] Validating configuration...")
        validation_results = env_manager.validate_configuration()
        
        print("\n[CHART] Configuration Validation Results:")
        print("-" * 40)
        for component, is_valid in validation_results.items():
            status = "[OK] VALID" if is_valid else "[X] INVALID"
            print(f"{component:20} {status}")
        
        # List stored API keys
        print("\n[INFO] Stored API Keys:")
        print("-" * 40)
        api_keys = security_manager.secret_manager.list_api_keys()
        if api_keys:
            for key_metadata in api_keys:
                print(f"• {key_metadata.name:15} ({key_metadata.service})")
        else:
            print("No API keys stored yet")
        
        # Test authentication
        print("\n[SECURE] Testing Authentication System:")
        print("-" * 40)
        
        # Test admin login
        admin_token = security_manager.authenticate_user("admin", "admin123")
        if admin_token:
            print("[OK] Admin authentication: SUCCESS")
            
            # Test authorization
            can_manage = security_manager.authorize_action(admin_token, "system.manage")
            can_trade = security_manager.authorize_action(admin_token, "trading.execute")
            
            print(f"[OK] Admin system.manage permission: {'GRANTED' if can_manage else 'DENIED'}")
            print(f"[OK] Admin trading.execute permission: {'GRANTED' if can_trade else 'DENIED'}")
        else:
            print("[X] Admin authentication: FAILED")
        
        # Test trader login
        trader_token = security_manager.authenticate_user("trader", "trader123")
        if trader_token:
            print("[OK] Trader authentication: SUCCESS")
            
            # Test authorization
            can_manage = security_manager.authorize_action(trader_token, "system.manage")
            can_trade = security_manager.authorize_action(trader_token, "trading.execute")
            
            print(f"[OK] Trader system.manage permission: {'DENIED' if not can_manage else 'GRANTED'}")
            print(f"[OK] Trader trading.execute permission: {'GRANTED' if can_trade else 'DENIED'}")
        else:
            print("[X] Trader authentication: FAILED")
        
        # Test encryption
        print("\n[LOCK] Testing Encryption System:")
        print("-" * 40)
        test_data = "This is sensitive trading data"
        encrypted = security_manager.encrypt_sensitive_data(test_data)
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        
        if decrypted == test_data:
            print("[OK] Encryption/Decryption: SUCCESS")
        else:
            print("[X] Encryption/Decryption: FAILED")
        
        print("\n[PARTY] Security system initialization completed successfully!")
        print("\n[NOTE] Next Steps:")
        print("1. Update your .env file with real API keys")
        print("2. Change default passwords for admin and trader users")
        print("3. Review and customize RBAC permissions as needed")
        print("4. Test the system with your specific configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"Security initialization failed: {e}")
        print(f"\n[X] Security initialization failed: {e}")
        return False


def show_security_status():
    """Show current security system status."""
    try:
        env_manager = get_env_manager()
        validation_results = env_manager.validate_configuration()
        
        print("[SECURE] Security System Status")
        print("=" * 40)
        
        total_components = len(validation_results)
        valid_components = sum(validation_results.values())
        
        print(f"Configuration Status: {valid_components}/{total_components} components valid")
        print(f"Environment: {env_manager.environment}")
        print(f"Production Mode: {'Yes' if env_manager.is_production() else 'No'}")
        
        print("\nComponent Status:")
        for component, is_valid in validation_results.items():
            status = "[OK]" if is_valid else "[X]"
            print(f"  {status} {component}")
        
    except Exception as e:
        print(f"[X] Failed to get security status: {e}")


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_security_status()
    else:
        initialize_security_system()


if __name__ == "__main__":
    main()