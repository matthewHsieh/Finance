import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from config import Config

class DataLoader:
    def __init__(self):
        Config.check_keys()
        self.fred = None
        if Config.FRED_API_KEY:
            self.fred = Fred(api_key=Config.FRED_API_KEY)
        
    def fetch_stock_data(self, ticker: str, period: str = "10y") -> pd.Series:
        """
        Fetch Adjusted Close price for a stock.
        """
        print(f"📉 Fetching Stock Data for {ticker}...")
        cache_path = os.path.join(Config.RAW_DATA_DIR, f"{ticker}.parquet")
        
        # Check Cache (implement simple cache logic if needed, skipping for MVP integrity 1st run)
        
        try:
            df = yf.download(ticker, period=period, progress=False, interval="1d")
            if df.empty:
                print(f"⚠️ Warning: No data found for {ticker}")
                return pd.Series()
            
            # yfinance often returns MultiIndex columns if list is passed, but single ticker is simple.
            # However, recent yfinance versions might include Ticker level.
            # Safe access to 'Adj Close'
            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                series = df.iloc[:, 0] # Fallback
                
            series.name = ticker
            return series
            
        except Exception as e:
            print(f"❌ Error fetching stock {ticker}: {e}")
            return pd.Series()

    def fetch_macro_data(self, code: str) -> pd.Series:
        """
        Fetch Macro data from FRED.
        Example Code: 'PCOPPUSDM' (Copper), 'GDP'
        """
        print(f"🏦 Fetching FRED Data for {code}...")
        
        if not self.fred:
            print("❌ FRED API Key not initialized.")
            return pd.Series()
            
        try:
            # FRED API returns a Series by default
            series = self.fred.get_series(code)
            series.name = code
            return series
            
        except Exception as e:
            print(f"❌ Error fetching FRED {code}: {e}")
            return pd.Series()

    def fetch_and_align(self, stock_ticker: str, macro_code: str) -> pd.DataFrame:
        """
        Fetch both and align timestamps.
        """
        stock = self.fetch_stock_data(stock_ticker)
        macro = self.fetch_macro_data(macro_code)
        
        if stock.empty or macro.empty:
            return pd.DataFrame()
            
        # Align
        # Join on index (Date)
        # Macro data (e.g. Monthly) needs to be forward filled to match daily stock data
        
        df = pd.concat([stock, macro], axis=1)
        df.columns = ['Stock', 'Macro']
        
        # Sort and Fill
        # Sort
        df = df.sort_index()
        
        # User Request: Use Linear Interpolation + Gaussian Noise for finer granularity on Monthly data
        # instead of step-function ffill() which causes zero-variance issues.
        df['Macro'] = df['Macro'].interpolate(method='time')
        
        # Fallback to ffill for leading NaNs or if interpolate missed edges
        df['Macro'] = df['Macro'].ffill().bfill()
        
        # Add Gaussian Noise (Jitter)
        # Scale noise to 0.1% of the values to ensure valid math (std > 0) but minimal distortion
        noise_level = 0.001 
        noise = np.random.normal(loc=0, scale=df['Macro'].mean() * noise_level, size=len(df))
        df['Macro'] = df['Macro'] + noise

        df = df.dropna() # Drop remaining NaNs (e.g. if stock data is missing)
        
        return df

if __name__ == "__main__":
    # Test
    loader = DataLoader()
    print("Testing YFinance...")
    s = loader.fetch_stock_data("1605.TW")
    print(s.tail())
