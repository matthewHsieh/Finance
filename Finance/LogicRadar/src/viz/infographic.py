import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.dates as mdates
import numpy as np

def generate_logic_card(stock_df: pd.Series, macro_df: pd.Series, ticker: str, macro_code: str, lag_days: int, corr_score: float, logic_valid: bool = True, output_dir: str = "data/processed/") -> str:
    """
    Generates a high-contrast infographic with a 'Projection Zone'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Align Data WITHOUT dropping future (The Forecast Fix)
    
    # Clean series individually first
    stock_df = stock_df.dropna()
    macro_df = macro_df.dropna()
    
    # Filter for Visualization (Regime Focus)
    cutoff_date = '2020-01-01'
    if isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df = stock_df.loc[cutoff_date:]
    
    if isinstance(macro_df.index, pd.DatetimeIndex):
         macro_df = macro_df.loc[cutoff_date:]
    
    # Calculate Z-Score Parameters based on the OVERLAPPING period only
    # This ensures fairness. If we use recent spiked data for mean/std, it might distort history.
    common_idx = stock_df.index.intersection(macro_df.index)
    
    if len(common_idx) < 30:
        print("⚠️ Not enough common data for Z-Score normalization")
        return ""
        
    stock_mean = stock_df.loc[common_idx].mean()
    stock_std = stock_df.loc[common_idx].std()
    
    macro_mean = macro_df.loc[common_idx].mean()
    macro_std = macro_df.loc[common_idx].std()
    
    # Apply Z-Score to FULL series
    stock_z = (stock_df - stock_mean) / stock_std
    macro_z = (macro_df - macro_mean) / macro_std
    
    # Create the Shifted Macro Line (The "Prediction")
    # Shift forward by Lag Days. 
    # Interpretation: Macro(t) predicts Stock(t+Lag).
    # We shift the Macro SERIES forward in time.
    # Note: tshift is deprecated, using shift with freq if index is DatetimeIndex
    macro_z_shifted = macro_z.shift(periods=lag_days, freq='D')
    
    # 2. Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # A. Stock (History) - Cyan
    ax.plot(stock_z.index, stock_z, color='#00d4ff', linewidth=2.5, label=f"{ticker} (Price)")
    
    # B. Macro (Prediction) - Orange
    ax.plot(macro_z_shifted.index, macro_z_shifted, color='#ffaa00', linewidth=2, linestyle='--', alpha=0.9, label=f"{macro_code} (Shifted {lag_days}d)")
    
    # 3. The "Forecast Zone" Visualization
    # Find the end of Stock data
    last_stock_date = stock_z.index[-1]
    last_macro_date = macro_z_shifted.index[-1]
    
    # Check if we have future projection
    # Ensure both are timestamps
    if last_macro_date > last_stock_date:
        # Highlight the area from Today to Future
        ax.axvspan(last_stock_date, last_macro_date, color='#ffaa00', alpha=0.15, label='FORECAST ZONE')
        
        # Annotation
        # Calculate mid-point correctly for dates
        time_diff = last_macro_date - last_stock_date
        mid_date = last_stock_date + time_diff / 2
        
        y_pos = ax.get_ylim()[1] * 0.8
        plt.text(mid_date, y_pos, "FUTURE\nTREND", 
                 color='#ffaa00', ha='center', fontsize=10, fontweight='bold', alpha=0.8)

    # 4. Viral Elements
    title_color = '#00ff00' if logic_valid else '#ff0000'
    prefix = "LOGIC VERIFIED" if logic_valid else "LOGIC WARNING"
    plt.title(f"{prefix}: {ticker} vs {macro_code}", fontsize=18, fontweight='bold', color='white', pad=20)
    
    stats_text = f"Correlation: {corr_score}\nLead/Lag: {lag_days} Days\nLogic Check: {'PASS' if logic_valid else 'FAIL'}"
    plt.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
             fontsize=11, family='monospace', color='white', bbox=dict(facecolor='#222', edgecolor=title_color, alpha=0.8))
    
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.legend(loc='lower left')
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    filename = os.path.join(output_dir, f"{ticker}_{macro_code}_logic_card.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"   🖼️ Logic Card Generated: {filename}")
    return filename

def generate_composite_card(stock_df: pd.Series, macros_data: dict, ticker: str, output_dir: str = "data/processed/") -> str:
    """
    Plots Stock vs Multiple Macro Variables (Z-Score Normalized).
    macros_data structure:
    {
        'MacroCode': {
            'series': pd.Series,
            'lag': int,
            'corr': float
        },
        ...
    }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Plot Stock (Base) - Cyan, Thick
    # Clean stock first
    stock_df = stock_df.dropna()
    
    # Filter for Viz
    cutoff_date = '2020-01-01'
    if isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df = stock_df.loc[cutoff_date:]
        
    stock_mean = stock_df.mean()
    stock_std = stock_df.std()
    stock_z = (stock_df - stock_mean) / stock_std
    
    ax.plot(stock_z.index, stock_z, color='#00d4ff', linewidth=3, label=f"{ticker} (Price)", zorder=10)
    
    # 2. Plot Macros
    # Use a colormap for distinct lines
    colors = plt.cm.autumn(np.linspace(0, 1, len(macros_data)))
    
    for idx, (code, data) in enumerate(macros_data.items()):
        m_series = data['series'].dropna()
        
        # Filter for Viz (Regime Focus)
        cutoff_date = '2020-01-01'
        if isinstance(m_series.index, pd.DatetimeIndex):
             m_series = m_series.loc[cutoff_date:]
             
        lag = data['lag']
        
        # Calculate Z-Score (using its own history IN THIS REGIME)
        m_mean = m_series.mean()
        m_std = m_series.std()
        m_z = (m_series - m_mean) / m_std
        
        # Shift
        m_z_shifted = m_z.shift(periods=lag, freq='D')
        
        # Plot
        color = colors[idx]
        label = f"{code} (Lag {lag}d, r={data['corr']:.2f})"
        ax.plot(m_z_shifted.index, m_z_shifted, color=color, linewidth=1.5, linestyle='--', alpha=0.8, label=label)
    
    # 3. Formatting
    plt.title(f"LOGIC COMPOSITE: {ticker} vs Selected Drivers", fontsize=18, fontweight='bold', color='white', pad=20)
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.legend(loc='upper left', fontsize=9, framealpha=0.2)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    filename = os.path.join(output_dir, f"{ticker}_composite_logic_card.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename

if __name__ == "__main__":
    pass
