"""
SETUP REAL LEAN DATA
===================
Download and configure real historical data for LEAN backtesting
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

class RealLEANDataSetup:
    """Set up real historical data for LEAN backtesting"""

    def __init__(self):
        self.data_folder = "./lean_engine/Data"
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        self.start_date = "2020-01-01"
        self.end_date = "2024-09-19"  # Fixed date

        # Create data folders
        self.create_data_structure()

    def create_data_structure(self):
        """Create LEAN data folder structure"""
        folders = [
            self.data_folder,
            f"{self.data_folder}/equity",
            f"{self.data_folder}/equity/usa",
            f"{self.data_folder}/equity/usa/daily",
            f"{self.data_folder}/equity/usa/minute"
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}")

    def download_real_data(self):
        """Download real historical data using yfinance"""
        print("=" * 60)
        print("DOWNLOADING REAL HISTORICAL DATA")
        print("=" * 60)

        for symbol in self.symbols:
            try:
                print(f"Downloading {symbol}...")

                # Download daily data
                ticker = yf.Ticker(symbol)
                daily_data = ticker.history(start=self.start_date, end=self.end_date)

                if daily_data.empty:
                    print(f"No data for {symbol}")
                    continue

                # Save daily data in LEAN format
                self.save_lean_daily_data(symbol, daily_data)

                # Download minute data for recent period (last 60 days for faster processing)
                recent_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                minute_data = ticker.history(start=recent_start, end=self.end_date, interval="1m")

                if not minute_data.empty:
                    self.save_lean_minute_data(symbol, minute_data)

                print(f"[OK] {symbol}: {len(daily_data)} daily bars, {len(minute_data)} minute bars")

            except Exception as e:
                print(f"[ERROR] Error downloading {symbol}: {e}")

    def save_lean_daily_data(self, symbol, data):
        """Save daily data in LEAN CSV format"""
        # LEAN daily format: Date,Open,High,Low,Close,Volume
        lean_data = pd.DataFrame({
            'date': data.index.strftime('%Y%m%d'),
            'open': (data['Open'] * 10000).astype(int),  # LEAN uses scaled prices
            'high': (data['High'] * 10000).astype(int),
            'low': (data['Low'] * 10000).astype(int),
            'close': (data['Close'] * 10000).astype(int),
            'volume': data['Volume'].astype(int)
        })

        # Save to LEAN daily folder
        output_path = f"{self.data_folder}/equity/usa/daily/{symbol.lower()}.csv"
        lean_data.to_csv(output_path, index=False, header=False)

        return len(lean_data)

    def save_lean_minute_data(self, symbol, data):
        """Save minute data in LEAN CSV format"""
        # LEAN minute format: DateTime,Open,High,Low,Close,Volume
        lean_data = pd.DataFrame({
            'datetime': data.index.strftime('%Y%m%d %H:%M'),
            'open': (data['Open'] * 10000).astype(int),
            'high': (data['High'] * 10000).astype(int),
            'low': (data['Low'] * 10000).astype(int),
            'close': (data['Close'] * 10000).astype(int),
            'volume': data['Volume'].astype(int)
        })

        # Group by date for LEAN file structure
        for date, group in lean_data.groupby(lean_data['datetime'].str[:8]):
            date_folder = f"{self.data_folder}/equity/usa/minute/{symbol.lower()}"
            os.makedirs(date_folder, exist_ok=True)

            output_path = f"{date_folder}/{date}.csv"
            group.to_csv(output_path, index=False, header=False)

    def create_security_database(self):
        """Create security database for LEAN"""
        print("\nCreating security database...")

        securities = []
        for symbol in self.symbols:
            securities.append({
                "Symbol": symbol,
                "ID": symbol,
                "Market": "usa",
                "SecurityType": "Equity",
                "DatabaseType": "TradeBar"
            })

        # Save securities database
        db_path = f"{self.data_folder}/securities.json"
        with open(db_path, 'w') as f:
            json.dump(securities, f, indent=2)

        print(f"[OK] Security database created: {db_path}")

    def validate_data(self):
        """Validate downloaded data"""
        print("\n" + "=" * 60)
        print("VALIDATING DOWNLOADED DATA")
        print("=" * 60)

        total_files = 0
        total_records = 0

        for symbol in self.symbols:
            daily_file = f"{self.data_folder}/equity/usa/daily/{symbol.lower()}.csv"

            if os.path.exists(daily_file):
                daily_df = pd.read_csv(daily_file, header=None)
                total_files += 1
                total_records += len(daily_df)

                print(f"[OK] {symbol}: {len(daily_df)} daily records")

                # Show sample data
                if len(daily_df) > 0:
                    first_row = daily_df.iloc[0]
                    last_row = daily_df.iloc[-1]
                    print(f"   First: {first_row[0]} | Last: {last_row[0]}")
            else:
                print(f"[MISSING] {symbol}: No data file found")

        print(f"\nDATA SUMMARY:")
        print(f"Files created: {total_files}")
        print(f"Total records: {total_records}")
        print(f"Date range: {self.start_date} to {self.end_date}")

        return total_files > 0 and total_records > 0

def main():
    """Set up real LEAN data"""
    print("REAL LEAN DATA SETUP")
    print("=" * 60)

    setup = RealLEANDataSetup()

    # Download real data
    setup.download_real_data()

    # Create security database
    setup.create_security_database()

    # Validate
    success = setup.validate_data()

    if success:
        print("\nREAL DATA SETUP COMPLETE!")
        print("[OK] Historical data downloaded and formatted for LEAN")
        print("[OK] Ready for real backtesting with actual market data")
    else:
        print("\n[ERROR] Data setup failed - check errors above")

if __name__ == "__main__":
    main()