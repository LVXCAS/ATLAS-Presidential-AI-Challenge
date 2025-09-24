"""
SETUP LEAN PROJECT
=================
Create proper LEAN project structure for real backtesting
"""

import os
import json
import shutil

def create_lean_project(strategy_name, strategy_file):
    """Create a LEAN project directory for a strategy"""

    # Create project directory
    project_dir = f"lean_projects/{strategy_name}"
    os.makedirs(project_dir, exist_ok=True)

    # Copy strategy file as main.py
    main_py_path = f"{project_dir}/main.py"
    shutil.copy(strategy_file, main_py_path)

    # Create lean.json config file for the project
    lean_config = {
        "environment": "backtesting",
        "algorithm-type-name": strategy_name,
        "algorithm-language": "Python",
        "algorithm-location": "main.py"
    }

    config_path = f"{project_dir}/lean.json"
    with open(config_path, 'w') as f:
        json.dump(lean_config, f, indent=2)

    # Create config.json for algorithm settings
    algo_config = {
        "start-date": "2020-01-01",
        "end-date": "2024-09-18",
        "cash": 100000,
        "data-folder": "../../lean_engine/Data"
    }

    algo_config_path = f"{project_dir}/config.json"
    with open(algo_config_path, 'w') as f:
        json.dump(algo_config, f, indent=2)

    print(f"Created LEAN project: {project_dir}")
    return project_dir

def main():
    """Create LEAN projects for all strategies"""
    print("CREATING LEAN PROJECTS")
    print("=" * 40)

    strategies = [
        ('RealMomentumStrategy', 'real_momentum_strategy.py'),
        ('RealMeanReversionStrategy', 'real_mean_reversion_strategy.py'),
        ('RealVolatilityStrategy', 'real_volatility_strategy.py')
    ]

    # Create base directory
    os.makedirs("lean_projects", exist_ok=True)

    created_projects = []

    for strategy_name, strategy_file in strategies:
        if os.path.exists(strategy_file):
            project_dir = create_lean_project(strategy_name, strategy_file)
            created_projects.append(project_dir)
        else:
            print(f"[WARNING] Strategy file not found: {strategy_file}")

    print(f"\\nCreated {len(created_projects)} LEAN projects:")
    for project in created_projects:
        print(f"  - {project}")

    print("\\n[OK] LEAN projects ready for backtesting!")
    print("Run: lean backtest lean_projects/RealMomentumStrategy")

if __name__ == "__main__":
    main()