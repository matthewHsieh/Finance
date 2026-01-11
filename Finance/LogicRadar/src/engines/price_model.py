import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta

class PriceModel:
    def __init__(self):
        self.model = LinearRegression()
        self.drivers = {} # Stores {code: {'lag': int, 'data': pd.Series}}
        self.target_ticker = ""

    def load_data(self, stock_series: pd.Series, macros_data: dict):
        """
        Stock Series: The target (Y)
        macros_data: {
            'HG=F': {'series': pd.Series, 'lag': 40},
            'JJN': ...
        }
        """
        self.stock_series = stock_series.dropna()
        self.drivers = macros_data
        
    def train(self, cutoff_date='2020-01-01'):
        """
        Trains the Multivariate Model.
        Returns metrics and the result DataFrame (History + Forecast).
        """
        # 1. Prepare Feature Matrix (X) and Target (Y)
        # Robust Logic: Handle Series or DataFrame (yfinance sometimes returns DF)
        if isinstance(self.stock_series, pd.DataFrame):
            df = self.stock_series.copy()
            df.columns = ['Y'] # Force rename
        else:
            df = self.stock_series.to_frame(name='Y')
        
        driver_cols = []
        max_lag = 0
        
        # Align Drivers with Lag
        # Crucial: If Macro leads by 40 days, we shift Macro FORWARD by 40 days to match Stock today.
        # Training Equation: Stock(t) = Alpha + Beta * Macro(t-40)
        # So in the dataframe, the row for Today must contain Stock(Today) and Macro(Today-40).
        # We achieve this by shifting Macro data.
        
        for code, info in self.drivers.items():
            lag = info['lag']
            series = info['series']
            if lag > max_lag:
                max_lag = lag
                
            # Shift Logic:
            # If shift(lag), positive lag means shifting logic forward?
            # Let's verify.
            # Stock(t) matches Macro(t-L). 
            # So if we take Macro Series and .shift(L), the value at t becomes the value from t-L.
            # Correct.
            
            shifted_series = series.shift(lag)
            shifted_series.name = code
            
            # Merge
            df = df.join(shifted_series, how='outer')
            driver_cols.append(code)

        # 2. Filter for Training (Data that actually exists for both Y and X)
        # We want to train on 2020+ mainly
        df_train = df.loc[cutoff_date:].dropna()
        
        if df_train.empty:
            return None, None
            
        X = df_train[driver_cols]
        y = df_train['Y']
        
        # 3. Fit
        self.model.fit(X, y)
        r2 = r2_score(y, self.model.predict(X))
        
        # 4. Predict (Fair Value) - for ALL available X data (including future if lags allow)
        # We re-construct X from the FULL df (which might have future dates for Y if X leads)
        # Wait, 'outer' join on timestamps limited by stock history? 
        # No, we need to extend the index into the future if lags exist.
        
        # Smart Forecasting Logic:
        # If Lag=40, we have Macro data for today... that Macro data corresponds to Stock price 40 days later.
        # So we effectively can predict 40 days into the future.
        
        # Re-build full X with macro data that extends beyond stock data
        # We need a master time index that covers all Macro valid dates
        
        full_df = pd.DataFrame()
        for code in driver_cols:
            # Original unshifted data has the latest dates
            # Shifted data moves it to the future
            series = self.drivers[code]['series']
            shifted = series.shift(self.drivers[code]['lag'])
            shifted.name = code
            if full_df.empty:
                full_df = pd.DataFrame(shifted)
            else:
                full_df = full_df.join(shifted, how='outer')
                
        # Filter: Start from 2020
        full_df = full_df.loc[cutoff_date:]
        
        # Drop rows where we don't have ALL driver data (cannot predict)
        full_X = full_df.dropna().copy()
        
        # Predict
        full_X['Fair_Value'] = self.model.predict(full_X[driver_cols])
        
        # Join back with Actual Stock Price
        # Robust Join: Handle stock_series being Series or DF
        if isinstance(self.stock_series, pd.DataFrame):
            actual_df = self.stock_series.copy()
            actual_df.columns = ['Actual']
        else:
            actual_df = self.stock_series.to_frame(name='Actual')

        result_df = full_X[['Fair_Value']].join(actual_df, how='left')
        
        # Calculate Deviation %
        result_df['Deviation'] = (result_df['Actual'] - result_df['Fair_Value']) / result_df['Fair_Value']
        
        # Metrics
        coefs = dict(zip(driver_cols, self.model.coef_))
        metrics = {
            'R2': r2,
            'Intercept': self.model.intercept_,
            'Coefficients': coefs,
            'Max_Lag': max_lag
        }
        
        return metrics, result_df

if __name__ == "__main__":
    pass
