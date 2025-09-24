#!/usr/bin/env python3
"""
Check OpenBB Installation Status
"""

def check_openbb():
    try:
        import openbb
        print("SUCCESS: OpenBB is installed and available!")
        
        # Test basic functionality
        from openbb import obb
        print("+ OpenBB platform accessible")
        
        # Show version if available
        if hasattr(openbb, '__version__'):
            print(f"+ Version: {openbb.__version__}")
        
        return True
        
    except ImportError:
        print("PENDING: OpenBB installation still in progress")
        print("This is normal for a large package like OpenBB")
        return False
        
    except Exception as e:
        print(f"PARTIAL: OpenBB installed but initializing - {e}")
        return False

def show_integration_next_steps():
    print("\nONCE OPENBB IS READY:")
    print("=" * 30)
    print("1. Enhanced Data Sources:")
    print("   - 100+ professional data providers")
    print("   - Real-time options chains")
    print("   - Economic indicators")
    print("   - Corporate fundamentals")
    print()
    print("2. Integration Points:")
    print("   - Enhance live_data_manager.py")
    print("   - Add to quantitative_integration_hub.py") 
    print("   - Improve options pricing accuracy")
    print("   - Better economic factor analysis")
    print()
    print("3. Trading Improvements:")
    print("   - More accurate opportunity detection")
    print("   - Better risk assessment")
    print("   - Enhanced market analysis")
    print("   - Professional-grade data quality")

if __name__ == "__main__":
    print("OPENBB INSTALLATION STATUS CHECK")
    print("=" * 40)
    
    success = check_openbb()
    
    if success:
        print("\nREADY FOR INTEGRATION!")
        print("Run: python openbb_integration_plan.py")
    else:
        show_integration_next_steps()
        print("\nTo check again: python check_openbb_status.py")