"""
Quick check if forex bot is still running
Checks for pythonw.exe processes
"""
import subprocess
import sys

print("Checking if forex bot is running...")
print()

try:
    # Windows command to find pythonw processes
    result = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq pythonw.exe'],
        capture_output=True,
        text=True
    )

    output = result.stdout

    if 'pythonw.exe' in output:
        lines = [l for l in output.split('\n') if 'pythonw.exe' in l]
        print(f"[FOUND] {len(lines)} pythonw.exe process(es) running")
        print()
        print("Forex bot is likely running in background.")
        print()
        print("To see positions: python VIEW_FOREX_POSITIONS.py")
        print("To stop bot: KILL_ALL_PYTHON.bat")
    else:
        print("[NOT FOUND] No pythonw.exe processes running")
        print()
        print("Bot may have stopped. To restart:")
        print("  python WORKING_FOREX_OANDA.py")
        print("Or:")
        print("  SIMPLE_FOREX_LAUNCH.bat")

except Exception as e:
    print(f"Error checking processes: {e}")
    print()
    print("Try manually: Open Task Manager and look for python/pythonw")
