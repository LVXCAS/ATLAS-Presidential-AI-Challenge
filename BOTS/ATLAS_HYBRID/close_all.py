import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
import os
load_dotenv('../../.env')
os.environ['OANDA_API_KEY'] = os.getenv('OANDA_API_KEY')
os.environ['OANDA_ACCOUNT_ID'] = os.getenv('OANDA_ACCOUNT_ID')

from adapters.oanda_adapter import OandaAdapter

o = OandaAdapter()
positions = o.get_open_positions()

print(f"Open positions: {len(positions)}")
for p in positions:
    pair = p['instrument']
    direction = 'long' if p['units'] > 0 else 'short'
    print(f"Closing {pair} {direction}...")
    o.close_position(pair, direction)

print(f"Final positions: {len(o.get_open_positions())}")
print(f"Balance: ${o.get_account_balance():,.2f}")
