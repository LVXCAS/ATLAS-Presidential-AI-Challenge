"""
Check LEAN Installation Status
Quick check to see if LEAN is installed and configured
"""

def check_lean_status():
    """Check if LEAN is installed and configured"""

    print("LEAN INSTALLATION STATUS CHECK")
    print("=" * 50)

    # Check 1: LEAN Python package
    try:
        import lean
        lean_version = getattr(lean, '__version__', 'unknown')
        print(f"LEAN Python Package: INSTALLED (v{lean_version})")
        lean_installed = True
    except ImportError:
        print("LEAN Python Package: NOT INSTALLED")
        lean_installed = False

    # Check 2: LEAN CLI
    import subprocess
    try:
        result = subprocess.run(['lean', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"LEAN CLI: INSTALLED ({result.stdout.strip()})")
            lean_cli_installed = True
        else:
            print("LEAN CLI: NOT WORKING")
            lean_cli_installed = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("LEAN CLI: NOT INSTALLED")
        lean_cli_installed = False

    # Check 3: Configuration files
    import os
    config_files = [
        'lean_config.json',
        'lean_config_paper_alpaca.json',
        'lean_config_live_alpaca.json',
        'lean_master_algorithm.py'
    ]

    configs_exist = 0
    print(f"\nLEAN Configuration Files:")
    for config in config_files:
        if os.path.exists(config):
            print(f"  {config}: EXISTS")
            configs_exist += 1
        else:
            print(f"  {config}: MISSING")

    # Check 4: LEAN algorithms directory
    if os.path.exists('lean_algorithms'):
        print(f"  lean_algorithms/: EXISTS")
        configs_exist += 1
    else:
        print(f"  lean_algorithms/: MISSING")

    # Overall status
    print(f"\nOVERALL LEAN STATUS:")
    print("-" * 30)

    if lean_installed and lean_cli_installed and configs_exist >= 3:
        status = "FULLY INSTALLED AND CONFIGURED"
        ready = True
    elif lean_installed or lean_cli_installed:
        status = "PARTIALLY INSTALLED"
        ready = False
    else:
        status = "NOT INSTALLED"
        ready = False

    print(f"Status: {status}")
    print(f"Ready for Trading: {'YES' if ready else 'NO'}")

    # Next steps
    print(f"\nNEXT STEPS:")
    if not lean_installed and not lean_cli_installed:
        print("1. Install LEAN: pip install lean")
        print("2. Install LEAN CLI: https://lean.io/docs/v2/lean-cli/installation")
        print("3. Run setup: python lean_local_setup.py")
        print("4. Configure API keys")
    elif not ready:
        print("1. Run setup: python lean_local_setup.py")
        print("2. Configure API keys in config files")
        print("3. Test with: python lean_runner.py backtest")
    else:
        print("1. Verify API keys in config files")
        print("2. Test with: python lean_runner.py backtest")
        print("3. Run paper trading: python lean_runner.py paper")
        print("4. Go live: python lean_runner.py live")

    return {
        'lean_installed': lean_installed,
        'lean_cli_installed': lean_cli_installed,
        'configs_exist': configs_exist,
        'ready': ready,
        'status': status
    }

if __name__ == "__main__":
    results = check_lean_status()