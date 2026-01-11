import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

def plot_valuation(result_df: pd.DataFrame, ticker: str, r2: float, drivers: list, output_dir: str = "data/processed/") -> str:
    """
    Plots Actual vs Fair Value Model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Filter for Viz (2020+) is already done in engine, but safe check
    viz_df = result_df.loc['2020-01-01':]
    
    # 1. Main Price Chart
    # Actual
    ax1.plot(viz_df.index, viz_df['Actual'], color='#00d4ff', linewidth=2.5, label='Actual Price', zorder=5)
    
    # Fair Value (Model)
    ax1.plot(viz_df.index, viz_df['Fair_Value'], color='#aa00ff', linewidth=2, linestyle='--', label=f'Fair Value (R²={r2:.2f})', alpha=0.9)
    
    # Forecast Zone (Where Actual is NaN but Fair Value exists)
    last_actual_date = viz_df['Actual'].last_valid_index()
    forecast_df = viz_df.loc[last_actual_date:]
    
    if len(forecast_df) > 1:
        ax1.fill_between(forecast_df.index, forecast_df['Fair_Value'], color='#aa00ff', alpha=0.2, label='Projected Zone')
        # Highlight End Target
        target_price = forecast_df['Fair_Value'].iloc[-1]
        target_date = forecast_df.index[-1]
        ax1.scatter(target_date, target_price, color='#aa00ff', s=100, zorder=10, marker='*')
        ax1.annotate(f"Target: {target_price:.1f}", (target_date, target_price), xytext=(10, 10), textcoords='offset points', color='white', fontweight='bold')

    ax1.set_title(f"PRICE PROJECTOR: {ticker} Valuation Model", fontsize=18, fontweight='bold', color='white', pad=15)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.2)
    ax1.set_ylabel("Price")

    # 2. Deviation Chart (Margin of Safety)
    # Deviation = (Actual - Fair) / Fair
    # If Deviation < -0.2 (Price is 20% below Fair Value) -> Strong Buy (Green)
    
    deviation = viz_df['Deviation'] * 100 # In %
    
    # Color logic
    colors = []
    for x in deviation:
        if pd.isna(x): colors.append('gray')
        elif x < -15: colors.append('#00ff00') # Undervalued (Green)
        elif x > 15: colors.append('#ff0000') # Overvalued (Red)
        else: colors.append('gray')
        
    ax2.bar(deviation.index, deviation, color=colors, width=2, alpha=0.6)
    ax2.axhline(0, color='white', linewidth=0.5)
    ax2.axhline(15, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(-15, color='green', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel("Deviation %")
    ax2.set_title(f"Valuation Gap (Green = Undervalued > 15%)", fontsize=10, color='gray')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    
    filename = os.path.join(output_dir, f"{ticker}_valuation_model.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename
