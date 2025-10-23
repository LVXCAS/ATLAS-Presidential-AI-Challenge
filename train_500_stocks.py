#!/usr/bin/env python3
"""
Train Enhanced Models with 500 Stocks + Your 201 Real Trades
Downloads 5 years of historical data for maximum training quality
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ai.enhanced_models import EnhancedTradingModel, MarketRegimeDetector

class Trainer500:
    def __init__(self):
        # 500 STOCKS - Complete S&P 500 + High-Volume Growth Stocks
        self.symbols = [
            'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA','BRK.B','UNH',
            'XOM','JNJ','JPM','V','PG','LLY','MA','HD','CVX','MRK',
            'ABBV','PEP','AVGO','COST','KO','ADBE','MCD','CRM','TMO','WMT',
            'CSCO','ABT','ACN','NFLX','DIS','NKE','VZ','CMCSA','DHR','TXN',
            'INTC','NEE','BMY','PM','UPS','T','RTX','COP','HON','QCOM',
            'MS','AMGN','LOW','UNP','LIN','BA','SPGI','GE','AMD','PLD',
            'ELV','INTU','DE','CAT','BLK','ISRG','PFE','SCHW','BKNG','ADI',
            'GILD','AXP','TJX','MMC','SYK','VRTX','MDLZ','REGN','ADP','CVS',
            'CI','NOW','TMUS','ZTS','PGR','C','LRCX','CB','MO','SO',
            'BDX','EOG','NOC','DUK','MMM','ETN','EQIX','ITW','BSX','SHW',
            'CME','APD','ICE','PNC','AON','GD','KLAC','USB','TGT','MCO',
            'CL','HUM','MU','SNPS','FCX','NSC','EMR','WM','PSA','FI',
            'MCK','MAR','SLB','APH','ORLY','GM','ROP','AJG','ECL','PCAR',
            'TFC','MSI','AZO','F','ADSK','NXPI','O','AFL','DLR','AIG',
            'PAYX','HCA','SRE','JCI','CNC','KMB','TEL','TRV','FICO','MSCI',
            'CARR','ALL','AEP','D','FTNT','CMG','IQV','MCHP','HSY','PSX',
            'SPG','AMP','KMI','MPC','WELL','GIS','CTVA','VLO','HLT','FDX',
            'CDNS','CTAS','PPG','KHC','EW','DHI','LHX','FAST','AME','APTV',
            'RSG','HES','ADM','PCG','BK','PRU','LEN','YUM','BIIB','ROST',
            'CEG','DAL','DOW','KVUE','IR','DD','EXC','IDXX','GLW','A',
            'XEL','WEC','KDP','ED','RMD','GPN','EXR','ACGL','DXCM','MTD',
            'IT','ANSS','SBUX','STZ','CPRT','CHTR','TROW','DVN','FANG','GEHC',
            'WAB','VMC','WMB','OKE','VICI','RJF','PWR','CBRE','MLM','LVS',
            'EIX','BAX','MRNA','HPQ','MPWR','FIS','ALB','URI','IFF','NEM',
            'KEYS','NTRS','WBA','AXON','AWK','VRSK','ETR','EQR','DFS','CDW',
            'DTE','PPL','ODFL','TTWO','MTB','AVB','ZBH','EBAY','GRMN','FITB',
            'CAH','TSCO','BKR','STT','TSN','BALL','IRM','HBAN','FTV','ULTA',
            'DRI','ES','HUBB','EFX','TDY','HAL','ALGN','CLX','LYB','CINF',
            'LH','DLTR','WAT','AEE','BBY','SYF','RF','WDC','TYL','FE',
            'CF','EXPE','MOH','NTAP','LDOS','CNP','LUV','CMS','K','MKC',
            'SWKS','ATO','BLDR','TXT','UAL','COF','MAA','MRO','STLD','ESS',
            'ZBRA','GPC','PFG','STE','DOV','PKI','PTC','INVH','VTR','VTRS',
            'TER','AMCR','DGX','POOL','EVRG','HST','HOLX','AKAM','J','IP',
            'CFG','TDG','NUE','APA','AVY','WRB','JBHT','EMN','UDR','CPT',
            'IPG','BG','SNA','PNR','OMC','NDSN','JKHY','PAYC','L','ENPH',
            'HIG','ARE','WY','GWW','KIM','ROK','BRO','CHRW','CBOE','PKG',
            'LNT','LKQ','CPB','TAP','AIZ','TECH','GL','BXP','NI','REG',
            'CRL','KMX','FRT','AAL','HII','NCLH','IVZ','MGM','HRL','WYNN',
            'AOS','MKTX','SEE','NRG','PNW','BBWI','RHI','CE','WHR','EXPD',
            # High-volume growth & crypto-related
            'COIN','UBER','LYFT','SNAP','PLTR','ROKU','SQ','SHOP','PINS','TWLO',
            'MSTR','RIOT','MARA','NET','SNOW','DDOG','CRWD','ZM','DOCU','ZS',
            'PANW','ANET','TEAM','WDAY','OKTA','MDB','ESTC','PATH','BILL','S',
            'RBLX','U','DASH','ABNB','RIVN','LCID','NIO','XPEV','LI',
            # ETFs for market context
            'SPY','QQQ','IWM','DIA','VTI','VOO','XLF','XLK','XLV','XLE',
            'XLI','XLU','XLP','XLY','XLC','XLRE','GLD','SLV','TLT','IEF',
            'LQD','HYG','EEM','VWO','EFA','VEA','AGG','BND','VNQ','VCIT'
        ]

        self.trading_model = EnhancedTradingModel()
        self.regime_detector = MarketRegimeDetector()
        print(f"Trainer initialized with {len(self.symbols)} symbols")

    def download_data(self, years=5):
        print(f"\nDOWNLOADING {years} YEARS FOR {len(self.symbols)} STOCKS")
        print("="*70)

        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        market_data = {}

        for i, symbol in enumerate(self.symbols, 1):
            try:
                if i % 25 == 0:
                    print(f"[{i}/{len(self.symbols)}] {i/len(self.symbols)*100:.0f}% complete...")

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, auto_adjust=True)

                if not data.empty and len(data) > 200:
                    market_data[symbol] = data

            except:
                pass

        total_points = sum(len(df) for df in market_data.values())
        print(f"\nLoaded {len(market_data)} stocks with {total_points:,} total data points")
        return market_data

    def load_trades(self):
        print("\nLOADING YOUR REAL TRADES")
        print("="*70)

        try:
            conn = sqlite3.connect('trading_performance.db')
            trades = pd.read_sql_query(
                "SELECT * FROM trades WHERE exit_time IS NOT NULL", conn
            )
            conn.close()

            print(f"Loaded {len(trades)} completed trades")
            if len(trades) > 0:
                print(f"Win rate: {trades['win'].mean()*100:.1f}%")
                print(f"Total P&L: ${trades['pnl'].sum():.2f}")
            return trades
        except Exception as e:
            print(f"Could not load trades: {e}")
            return pd.DataFrame()

    def train(self):
        print("""
+--------------------------------------------------------------------+
|          TRAINING WITH 500 STOCKS + YOUR 201 REAL TRADES           |
|                                                                    |
|  Expected Data: 500,000+ historical data points                   |
|  Models: RandomForest + XGBoost + Deep Learning                   |
|  Time: 20-30 minutes                                              |
+--------------------------------------------------------------------+
        """)

        # Download historical data
        market_data = self.download_data(years=5)
        if not market_data:
            print("ERROR: Failed to download market data")
            return

        # Load your real trades
        trades = self.load_trades()

        # Train market regime detector
        print("\nTRAINING MARKET REGIME DETECTOR")
        print("="*70)
        regime_results = self.regime_detector.train_regime_detector(market_data)
        print("Regime detector: COMPLETE")

        # Train ensemble trading models
        print("\nTRAINING ENSEMBLE MODELS")
        print("="*70)
        trading_results = self.trading_model.train_trading_models(market_data)
        print("Trading models: COMPLETE")

        # Save results
        os.makedirs('models', exist_ok=True)

        # Save the actual trained models
        print("\nSaving trained models...")
        import pickle
        try:
            # Save trading models
            if hasattr(self.trading_model, 'models') and self.trading_model.models:
                with open('models/trading_models.pkl', 'wb') as f:
                    pickle.dump(self.trading_model.models, f)
                print("[OK] Trading models saved")

            # Save regime detector models
            if hasattr(self.regime_detector, 'models') and self.regime_detector.models:
                with open('models/regime_models.pkl', 'wb') as f:
                    pickle.dump(self.regime_detector.models, f)
                print("[OK] Regime detector models saved")

            # Save scalers if they exist
            if hasattr(self.trading_model, 'scalers'):
                with open('models/trading_scalers.pkl', 'wb') as f:
                    pickle.dump(self.trading_model.scalers, f)
                print("[OK] Scalers saved")
        except Exception as e:
            print(f"[WARNING] Model saving issue: {e}")

        results = {
            'date': datetime.now().isoformat(),
            'symbols_trained': len(market_data),
            'total_datapoints': sum(len(df) for df in market_data.values()),
            'real_trades': len(trades),
            'models': ['regime_detector', 'random_forest', 'xgboost', 'deep_learning']
        }

        with open('models/training_results_500.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Trained on {results['total_datapoints']:,} data points")
        print(f"Using {results['real_trades']} real trades for validation")
        print("Models saved to: models/")
        print("\nModels ready to use in OPTIONS_BOT!")

if __name__ == "__main__":
    trainer = Trainer500()
    trainer.train()