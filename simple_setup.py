#!/usr/bin/env python3
"""
Simple Setup - Create directories and basic configuration for acceleration
"""

import os
import json
from datetime import datetime

def simple_setup():
    """Create basic setup for acceleration"""
    print("SIMPLE ACCELERATION SETUP")
    print("=" * 40)
    
    # Create directories
    dirs_created = 0
    directories = [
        'models',
        'models/base',
        'models/transfer', 
        'models/quick',
        'data',
        'data/historical',
        'logs/learning'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            dirs_created += 1
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating {directory}: {e}")
    
    # Create configuration file
    config = {
        "setup_date": datetime.now().isoformat(),
        "acceleration_enabled": True,
        "transfer_learning": True,
        "realtime_learning": True,
        "model_update_frequency": 3600,  # 1 hour
        "symbols_for_training": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        "min_accuracy_threshold": 0.52,
        "max_position_risk": 0.02,
        "learning_rate": 0.1,
        "rebalance_threshold": 0.05
    }
    
    try:
        with open('acceleration_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Created acceleration_config.json")
    except Exception as e:
        print(f"Error creating config: {e}")
    
    # Create learning status file
    status = {
        "setup_complete": True,
        "last_update": datetime.now().isoformat(),
        "models_trained": 0,
        "trades_processed": 0,
        "current_accuracy": 0.5,
        "learning_stage": "initialized"
    }
    
    try:
        with open('learning_status.json', 'w') as f:
            json.dump(status, f, indent=2)
        print("Created learning_status.json")
    except Exception as e:
        print(f"Error creating status: {e}")
    
    print("\n" + "=" * 40)
    print("SETUP COMPLETE!")
    print(f"Created {dirs_created} directories")
    print("Configuration files created")
    print("\nREADY FOR STEP 2: Initialize Real-Time Learning")
    print("\nNext command:")
    print("python initialize_learning.py")
    
    return True

if __name__ == "__main__":
    simple_setup()