import requests
import pandas as pd

class EconomicDataAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_credit_card_data(self, sector):
        # Placeholder for fetching credit card spending data
        print(f"Fetching credit card data for sector: {sector}")
        # Dummy data
        return pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-08']),
            'spending_growth': [0.02, 0.03]
        })

    def get_economic_indicator(self, indicator_name):
        # Placeholder for fetching economic indicators (e.g., GDP, CPI)
        print(f"Fetching economic indicator: {indicator_name}")
        # Dummy data
        return pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-04-01']),
            'value': [25.5, 25.8]  # Trillions of USD for GDP
        })

    def generate_signals(self, data, data_type):
        signals = []
        if data_type == 'credit_card':
            for index, row in data.iterrows():
                if row['spending_growth'] > 0.025:
                    signals.append({'signal': 'BUY', 'reason': f"High credit card spending growth in sector"})
        elif data_type == 'gdp':
            if data['value'].pct_change().iloc[-1] > 0.01:
                 signals.append({'signal': 'BUY', 'reason': 'Strong GDP growth'})
        return pd.DataFrame(signals)

if __name__ == '__main__':
    agent = EconomicDataAgent(api_key="YOUR_DATA_PROVIDER_API_KEY")
    
    # Credit card data analysis
    credit_card_data = agent.get_credit_card_data("Retail")
    credit_card_signals = agent.generate_signals(credit_card_data, 'credit_card')
    print("Credit Card Signals:\n", credit_card_signals)

    # Economic indicator analysis
    gdp_data = agent.get_economic_indicator("GDP")
    gdp_signals = agent.generate_signals(gdp_data, 'gdp')
    print("GDP Signals:\n", gdp_signals)
