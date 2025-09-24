"""
RISK INTEGRATION CHECK
Verify what risk management is actually integrated in your autonomous systems
"""

import os
import importlib.util
import json

def check_risk_libraries():
    """Check which risk management libraries are available"""

    print("RISK MANAGEMENT LIBRARIES CHECK")
    print("=" * 50)

    # Core risk libraries
    risk_libraries = {
        'numpy': 'Core mathematical operations',
        'pandas': 'Data manipulation and analysis',
        'scipy': 'Statistical functions and optimization',
        'sklearn': 'Machine learning and covariance estimation',
        'cvxpy': 'Convex optimization for portfolio constraints',
        'pypfopt': 'Portfolio optimization (Markowitz, etc.)',
        'riskfolio': 'Advanced risk parity and portfolio methods',
        'empyrical': 'Risk metrics and performance analytics',
        'quantstats': 'Trading performance and risk analysis',
        'arch': 'GARCH models for volatility forecasting',
        'statsmodels': 'Statistical modeling',
        'VaR': 'Value at Risk calculations'
    }

    available_libraries = {}

    for lib, description in risk_libraries.items():
        try:
            if lib == 'VaR':
                # VaR is usually part of other libraries
                import scipy.stats
                available_libraries[lib] = True
                print(f"✓ {lib:15s} - {description}")
            else:
                __import__(lib)
                available_libraries[lib] = True
                print(f"✓ {lib:15s} - {description}")
        except ImportError:
            available_libraries[lib] = False
            print(f"✗ {lib:15s} - {description} (NOT INSTALLED)")

    return available_libraries

def check_autonomous_risk_integration():
    """Check if autonomous systems have risk management integrated"""

    print(f"\nAUTONOMOUS SYSTEM RISK INTEGRATION:")
    print("=" * 50)

    # Key autonomous system files
    autonomous_files = [
        'truly_autonomous_system.py',
        'autonomous_market_open_system.py',
        'autonomous_options_discovery.py',
        'mega_discovery_engine.py',
        'realistic_40_percent_system.py',
        'intelligent_rebalancer.py'
    ]

    risk_integration_status = {}

    for filename in autonomous_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()

            # Check for risk management keywords
            risk_keywords = [
                'risk', 'Risk', 'VAR', 'VaR', 'drawdown', 'stop_loss',
                'position_size', 'max_allocation', 'correlation', 'volatility',
                'risk_manager', 'emergency', 'circuit_breaker'
            ]

            found_keywords = [kw for kw in risk_keywords if kw in content]

            # Check for actual risk management implementation
            has_position_limits = 'max_allocation' in content or 'position_size' in content
            has_stop_loss = 'stop_loss' in content or 'emergency' in content
            has_risk_metrics = 'volatility' in content or 'correlation' in content

            risk_score = len(found_keywords)

            if risk_score >= 5 and (has_position_limits or has_stop_loss):
                risk_level = "GOOD"
            elif risk_score >= 3:
                risk_level = "BASIC"
            else:
                risk_level = "MINIMAL"

            risk_integration_status[filename] = {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'has_position_limits': has_position_limits,
                'has_stop_loss': has_stop_loss,
                'has_risk_metrics': has_risk_metrics,
                'keywords_found': found_keywords
            }

            print(f"{filename:35s} - {risk_level:8s} (Score: {risk_score:2d})")
            if has_position_limits:
                print(f"  ✓ Position limits implemented")
            if has_stop_loss:
                print(f"  ✓ Emergency controls present")
            if has_risk_metrics:
                print(f"  ✓ Risk metrics calculated")

        else:
            print(f"{filename:35s} - NOT FOUND")

    return risk_integration_status

def check_40_percent_system_risks():
    """Specific risk analysis for the 40% ROI system"""

    print(f"\n40% ROI SYSTEM RISK ANALYSIS:")
    print("=" * 40)

    try:
        with open('realistic_40_percent_plan.json', 'r') as f:
            plan = json.load(f)

        print("Current 40% ROI Plan Risk Assessment:")

        # Position concentration
        if 'deployment_phases' in plan:
            total_positions = plan.get('total_positions', 0)
            print(f"Total positions: {total_positions}")

            if total_positions < 5:
                print("⚠️  HIGH CONCENTRATION RISK - Less than 5 positions")
            elif total_positions < 10:
                print("⚡ MODERATE RISK - Good diversification")
            else:
                print("✓ LOW CONCENTRATION RISK - Well diversified")

        # Target ROI risk
        target_roi = plan.get('target_monthly_roi', 0)
        if target_roi > 0.3:  # 30%+
            print(f"⚠️  EXTREME ROI TARGET - {target_roi:.1%} monthly is very aggressive")

        # Check if risk controls exist
        risk_controls_present = False

        # Look for risk management in the system files
        if os.path.exists('realistic_40_percent_system.py'):
            with open('realistic_40_percent_system.py', 'r') as f:
                content = f.read()

            if 'max_single_position' in content:
                print("✓ Position size limits implemented")
                risk_controls_present = True

            if 'risk_level' in content:
                print("✓ Risk level assessment included")
                risk_controls_present = True

            if 'leverage' in content:
                print("⚠️  Leverage detected - increases risk")

        if not risk_controls_present:
            print("❌ LIMITED RISK CONTROLS - Need better risk management")

    except FileNotFoundError:
        print("40% ROI plan file not found")

def generate_risk_recommendations():
    """Generate recommendations for better risk management"""

    print(f"\nRISK MANAGEMENT RECOMMENDATIONS:")
    print("=" * 50)

    recommendations = [
        "1. POSITION SIZING: Limit any single position to 15% of portfolio",
        "2. STOP LOSSES: Implement automatic stop losses at -20% per position",
        "3. PORTFOLIO VaR: Monitor daily Value at Risk (1% VaR < 5% of portfolio)",
        "4. CORRELATION LIMITS: Avoid positions with >70% correlation",
        "5. VOLATILITY MONITORING: Alert if portfolio volatility >25% annualized",
        "6. LIQUIDITY CHECKS: Ensure positions can be closed within 24 hours",
        "7. DRAWDOWN PROTECTION: Auto-reduce positions if drawdown >15%",
        "8. LEVERAGE LIMITS: Maximum 2x leverage on any strategy",
        "9. EMERGENCY STOPS: Circuit breakers for extreme market moves",
        "10. REAL-TIME MONITORING: Update risk metrics every 15 minutes"
    ]

    for rec in recommendations:
        print(rec)

    print(f"\nCRITICAL FOR 40% ROI TARGET:")
    print("- Your target is EXTREMELY aggressive")
    print("- Requires sophisticated risk management")
    print("- Consider starting with 20% monthly target")
    print("- Scale up gradually as system proves profitable")

def main():
    """Run complete risk integration check"""

    # Check available libraries
    available_libs = check_risk_libraries()

    # Check autonomous system integration
    risk_status = check_autonomous_risk_integration()

    # Check 40% system specific risks
    check_40_percent_system_risks()

    # Generate recommendations
    generate_risk_recommendations()

    # Summary
    print(f"\nRISK MANAGEMENT SUMMARY:")
    print("=" * 30)

    lib_coverage = sum(available_libs.values()) / len(available_libs)
    print(f"Library Coverage: {lib_coverage:.1%}")

    systems_with_good_risk = sum(1 for status in risk_status.values()
                                if status['risk_level'] == 'GOOD')
    total_systems = len(risk_status)

    if total_systems > 0:
        integration_score = systems_with_good_risk / total_systems
        print(f"Risk Integration Score: {integration_score:.1%}")

        if integration_score >= 0.8:
            print("✓ EXCELLENT risk management integration")
        elif integration_score >= 0.6:
            print("⚡ GOOD risk management, some improvements needed")
        else:
            print("⚠️  POOR risk management - urgent improvements needed")

    return {
        'libraries': available_libs,
        'integration': risk_status,
        'recommendations': 'See output above'
    }

if __name__ == "__main__":
    main()