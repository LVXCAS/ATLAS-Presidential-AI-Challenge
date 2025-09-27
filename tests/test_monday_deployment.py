"""
TEST MONDAY DEPLOYMENT SYSTEM
============================
Windows-compatible test of the Monday deployment validation
"""

import asyncio
import sys
import os
from datetime import datetime

# Import the deployment system
from MONDAY_DEPLOYMENT_SYSTEM import MondayDeploymentSystem

async def test_monday_deployment():
    """Test the Monday deployment system"""
    print("="*80)
    print("MONDAY DEPLOYMENT VALIDATION SYSTEM")
    print("Getting your autonomous trading empire ready for Monday!")
    print("="*80)

    # Initialize deployment system
    deployment = MondayDeploymentSystem()

    print(f"\nRunning comprehensive validation...")
    print(f"This validates all systems for Monday deployment")

    # Run comprehensive validation
    await deployment.run_comprehensive_validation()

    # Show results
    if deployment.deployment_ready:
        print(f"\n[SUCCESS] Your GTX 1660 Super autonomous trading empire is READY for Monday!")
        print(f"All systems validated and operational.")
    else:
        print(f"\n[NEEDS WORK] Some components need setup before Monday deployment.")
        print(f"Check the deployment report for details.")

    # Show component status
    print(f"\nComponent Status Summary:")
    for name, status in deployment.component_status.items():
        status_text = "[OK]" if status.status == 'READY' else "[SETUP]" if status.status == 'NEEDS_SETUP' else "[FAIL]"
        print(f"  {status_text} {name}: {status.details}")

    print(f"\nValidation complete! Check the generated deployment report for full details.")

if __name__ == "__main__":
    asyncio.run(test_monday_deployment())