import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.engines.data_loader import DataLoader
from config import Config

def test_pipeline():
    print("🚀 Starting Data Pipeline Test...")
    
    # 1. Check Config
    print(f"Checking API Keys...")
    if not Config.FRED_API_KEY:
        print("⚠️ FRED_API_KEY is missing. FRED fetch will fail.")
    else:
        print("✅ FRED_API_KEY found.")

    loader = DataLoader()

    # 2. Test Stock Fetch (Should work without keys)
    ticker = "2330.TW" # TSMC
    print(f"\n--- Testing Stock Fetch ({ticker}) ---")
    stock = loader.fetch_stock_data(ticker)
    if not stock.empty:
        print(f"✅ Successfully fetched {len(stock)} rows for {ticker}")
        print(stock.tail(3))
    else:
        print("❌ Stock fetch failed.")

    # 3. Test Macro Fetch (Needs Key)
    macro_code = "DCOILWTICO" # Crude Oil
    print(f"\n--- Testing Macro Fetch ({macro_code}) ---")
    if Config.FRED_API_KEY:
        macro = loader.fetch_macro_data(macro_code)
        if not macro.empty:
            print(f"✅ Successfully fetched {len(macro)} rows for {macro_code}")
            print(macro.tail(3))
        else:
            print("❌ Macro fetch failed (API Error?).")
    else:
        print("⏭️ Skipping Macro Fetch (No Key).")

    # 4. Test Alignment
    if not stock.empty and Config.FRED_API_KEY:
        print(f"\n--- Testing Data Alignment ---")
        aligned = loader.fetch_and_align(ticker, macro_code)
        if not aligned.empty:
            print(f"✅ Successfully aligned data. Shape: {aligned.shape}")
            print(aligned.tail())
        else:
            print("❌ Alignment failed.")

if __name__ == "__main__":
    test_pipeline()
