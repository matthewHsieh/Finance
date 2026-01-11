import sys
import os
# Add project root to path for CLI execution (before imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Dict
import pandas as pd
import numpy as np
from src.engines.data_loader import DataLoader
from src.engines.semantic_validator import SemanticValidator
from src.utils.math_utils import compute_max_lag_correlation

class DetectiveEngine:
    def __init__(self, macro_universe_codes: List[str] = None):
        """
        macro_universe_codes: List of FRED codes to scan.
        If None, uses a default small set for demo.
        """
        self.loader = DataLoader()
        # Default mini-universe for MVP
        # Default mini-universe for MVP (Mixed FRED Codes & Yahoo Futures)
        self.macro_codes = macro_universe_codes or [
            'HG=F', # Copper Futures (Yahoo)
            'ALI=F', # Aluminum Futures (Yahoo)
            'PNICKUSDM', # Global Nickel Price (FRED/IMF) - Monthly & Laggy
            'VALE', # Vale S.A. (Nickel Miner Proxy) - Real-time daily data
            # 'SN=F', # Tin Futures (Delisted/Unstable)
            'GC=F', # Gold Futures
            'SI=F', # Silver Futures
            'CL=F', # Crude Oil Futures
            'T10Y2Y', # 10Y-2Y Yield Spread (Recession Signal)
            'DGS10', # 10-Year Treasury Yield (Benchmark Interest Rate)
            'FEDFUNDS', # Federal Funds Effective Rate (Central Bank Rate)
            'VIXCLS', # VIX (Volatility)
            'M2SL', # M2 Money Supply
        ]
        self.validator = SemanticValidator()

    def analyze(self, ticker: str, skip_validation: bool = False) -> List[Dict]:
        """
        Scans macro variables to find optimal drivers.
        Returns top 5 sorted by correlation strength.
        skip_validation: If True, returns Math candidates without asking LLM (for UI apps).
        """
        print(f"🕵️‍♂️ Detective analyzing: {ticker}...")
        
        # 1. Fetch Target Data
        stock_series = self.loader.fetch_stock_data(ticker)
        if stock_series.empty:
            print(f"❌ Could not fetch stock data for {ticker}")
            return []
            
        findings = []
        
        # 2. Brute Force Scan
        print(f"   Scanning {len(self.macro_codes)} macro variables...")
        for code in self.macro_codes:
            try:
                # 3.1 Fetch Data (Handle Source Routing)
                # VALE is a stock ticker (Yahoo), not a FRED code, but doesn't have '='
                is_yahoo = '=' in code or code == 'VALE'
                
                if is_yahoo: 
                    macro_series = self.loader.fetch_stock_data(code)
                    if macro_series.empty: continue
                    # Align manually 
                    # Fetch Stock
                    stock_series = self.loader.fetch_stock_data(ticker)
                    
                    df = pd.concat([stock_series, macro_series], axis=1).dropna()
                    df.columns = ['Stock', 'Macro']
                else: 
                     # FRED Code
                     df = self.loader.fetch_and_align(ticker, code)
                
                if df.empty or len(df) < 30: continue
                
                # Filter for User's Regime (2020-Now)
                # This focuses on the post-COVID inflation/commodity cycle
                df = df.loc['2020-01-01':]
                
                if df.empty or len(df) < 30: continue

                stock_data = df['Stock']
                macro_data = df['Macro']
                
                # 3.2 Math Analysis (Long Term & Short Term)
                
                # A. Long Term (Full Window ~2y)
                best_lag, max_corr = compute_max_lag_correlation(stock_data, macro_data)
                
                # B. Short Term (Last 60 Days) - To capture recent "30% jump"
                recent_df = df.tail(60) 
                if len(recent_df) > 20:
                     _, recent_corr = compute_max_lag_correlation(recent_df['Stock'], recent_df['Macro'], max_lookback=10)
                else:
                     recent_corr = 0
                
                # Score: Use the higher of Long or Short term to detect "Emerging Logic"
                # If Recent > 0.8 but Long is 0.2, it's an "Emerging Driver".
                
                final_score = max(abs(max_corr), abs(recent_corr))
                
                # Filter - Lower threshold slightly as we have less data but it's more relevant
                # Force include PNICKUSDM (Nickel) because user knows it's relevant and metrics are skewed by monthly granularity
                # Force include PNICKUSDM (Nickel) because user knows it's relevant
                # Handle NaN scores (common with monthly flat data) by defaulting to 0.0
                if final_score > 0.15 or code == 'PNICKUSDM': 
                    findings.append({
                        'code': code,
                        'max_corr': round(max_corr, 4) if not np.isnan(max_corr) else 0.0,
                        'recent_corr': round(recent_corr, 4) if not np.isnan(recent_corr) else 0.0,
                        'best_lag': best_lag,
                        'sample_size': len(df)
                    })
                else:
                    print(f"     [DEBUG] Skipped {code}: Score {final_score:.4f} < 0.15")
                    
            except Exception as e:
                print(f"   ⚠️ Error analyzing {code}: {e}")
                continue
        
        # 3. Sort Results & Apply Semantic Check
        findings.sort(key=lambda x: max(abs(x['max_corr']), abs(x['recent_corr'])), reverse=True)
        
        # User Request: Show ALL findings, not just top 5 (Nickel might be lower down)
        top_findings = findings 
        validated_findings = []
        
        print("\n🔎 Top Findings (Math Only - LT=LongTerm, ST=ShortTerm):")
        for f in top_findings:
            print(f"   - {f['code']}: LT {f['max_corr']} | ST {f['recent_corr']} (Lag {f['best_lag']}d)")
            
            # Semantic Check
            if skip_validation:
                 f['is_logical'] = None
                 f['logic_reason'] = "Pending Verification"
                 validated_findings.append(f)
            else:
                is_valid, reason = self.validator.check_causality(ticker, f['code'])
                f['is_logical'] = is_valid
                f['logic_reason'] = reason
                
                if is_valid:
                    print(f"     ✅ Logic Verified: {reason}")
                    validated_findings.append(f)
                else:
                    print(f"     ❌ Logic Failed: {reason}")
            
        return validated_findings

if __name__ == "__main__":
    # Test
    engine = DetectiveEngine()
    engine.analyze("1605.TW")
