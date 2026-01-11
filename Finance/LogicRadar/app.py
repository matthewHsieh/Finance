import streamlit as st
import pandas as pd
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.engines.detective_engine import DetectiveEngine
from src.viz.infographic import generate_logic_card, generate_composite_card
from src.viz.valuation_plot import plot_valuation
from src.engines.semantic_validator import SemanticValidator
from src.engines.price_model import PriceModel

st.set_page_config(page_title="LogicRadar V3", page_icon="📡", layout="wide")

st.title("📡 LogicRadar V3: The Smart Detective")
st.markdown("### Discover Hidden Variables Driving Your Stock")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", value="1605.TW")
    
    st.divider()
    st.markdown("**System Status**")
    st.info("LLM Provider: Terminal/Manual")
    
    if st.button("🚀 Scan Logic"):
        st.session_state['scan_triggered'] = True

# Helper for caching the expensive scan
# Helper for caching the expensive scan
# Changed to allow_output_mutation or similar if needed, but here just rename or clear manually.
# User can clear cache from UI, but let's change logic slightly to invalidate.
@st.cache_data(ttl=3600)
def run_scan(ticker_input, regime='2020-Now'): # Added dummy arg to potentially invalidate old cache signatures
    engine = DetectiveEngine()
    # Ensure skip_validation=True for the math scan
    return engine.analyze(ticker_input, skip_validation=True), engine

# Main Logic
if 'scan_triggered' in st.session_state and st.session_state['scan_triggered']:
    st.subheader(f"🔍 Analyzing {ticker}...")
    
    with st.spinner("Scanning macro universe (FRED + Futures)..."):
        # Use cached Function
        results, detective_instance = run_scan(ticker, regime='2020-Now')
    
    if not results:
        st.error("No significant correlations found.")
    else:
        st.success(f"Found {len(results)} potential drivers.")
        
        # Prepare Data for Table
        # We need to run Logic Check for the UI display
        # In a real app, this might be async. For MVP, we do it here or let user click.
        # Plan says: "Display ✅/❌". So we should verify them.
        # Since we are in 'Terminal' mode, we can't pop a terminal.
        # We will Mock "Auto Approve" for the purpose of the Demo App unless we build a Streamlit UI for it.
        # Compromise: We show "Pending" and let user "Verify" by clicking?
        # Or, simpler: We execute the check in 'manual' mode but via Streamlit widgets? 
        # Writing a full chat interface is complex for MVP.
        # Let's auto-verify with a "Note" for now, or just mark them as '?' and let user decide.
        
        # Better: Let's assume for the "App" experience, we want to see the result.
        # We will add a column "Logic Check" that says "Verify Needed".
        
        display_data = []
        for r in results:
            # Determine best score
            lt_corr = r.get('max_corr', 0)
            st_corr = r.get('recent_corr', 0)
            best_score = max(abs(lt_corr), abs(st_corr))
            
            display_data.append({
                "Macro Variable": r['code'],
                "Correlation (LT)": lt_corr,
                "Correlation (ST)": st_corr,
                "Best Lag (Days)": r['best_lag'],
                "Score": best_score,
                "Logic Status": "❓ Verify" 
            })
            
        df_results = pd.DataFrame(display_data)
        
        # Display Table
        st.dataframe(df_results.style.format({
            "Correlation (LT)": "{:.4f}",
            "Correlation (ST)": "{:.4f}",
            "Score": "{:.4f}"
        }))
        
        # Visualization Section
        st.divider()
        st.subheader("🎨 Generate Logic Card")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Multi-select for Composite View
            selected_macros = st.multiselect("Select Drivers to Visualize (Composite)", df_results['Macro Variable'], default=None)
            
            # User Manual Validation
            st.info("Composite View: Plots all selected variables on one chart.")
            
            if st.button("✨ Generate Composite Infographic"):
                if not selected_macros:
                    st.warning("Please select at least one variable.")
                else:
                    with st.spinner("Generating Art..."):
                        macros_data = {}
                        
                        # Fetch Stock Data once
                        stock_series = detective_instance.loader.fetch_stock_data(ticker)
                        
                        for macro_code in selected_macros:
                            # Find result meta
                            target_result = next(r for r in results if r['code'] == macro_code)
                            
                            is_yahoo = '=' in macro_code or macro_code == 'VALE'
                            
                            # Use detective_instance from cache if possible, or new loader
                            # Since detective_instance is returned by run_scan, we can use it, 
                            # but run_scan returns 'engine' object which is DetectiveEngine.
                            # We can access engine.loader
                            loader = detective_instance.loader

                            if is_yahoo:
                                m_series = loader.fetch_stock_data(macro_code)
                            else:
                                m_series = loader.fetch_macro_data(macro_code)
                                
                            macros_data[macro_code] = {
                                'series': m_series,
                                'lag': target_result['best_lag'],
                                'corr': display_data[next(i for i, x in enumerate(display_data) if x['Macro Variable'] == macro_code)]['Score']
                            }
                            
                        # Generate Composite
                        img_path = generate_composite_card(stock_series, macros_data, ticker)
                        st.image(img_path, caption=f"Composite Logic Card: {ticker}")
                        
                        # --- Feature: Price Projector (Valuation Model) ---
                        st.divider()
                        st.subheader("💰 Price Projector (Valuation Model)")
                        st.write("Does the Logic justify the Price? Let's build a Regression Model.")
                        
                        # Automated Build (No Button)
                        with st.spinner("Training Regression AI to find Fair Value..."):
                            pm = PriceModel()
                            # Prepare data dictionary for price model
                            pm_data = {}
                            for k, v in macros_data.items():
                                pm_data[k] = {'series': v['series'], 'lag': v['lag']}
                                
                            pm.load_data(stock_series, pm_data)
                            # Train on 2020+
                            metrics, res_df = pm.train(cutoff_date='2020-01-01')
                            
                            if metrics:
                                st.success(f"Model Trained! R² Confidence: {metrics['R2']:.2%}")
                                
                                # Show Formula
                                formula = []
                                for code, coef in metrics['Coefficients'].items():
                                    formula.append(f"{coef:.2f} * {code}")
                                st.code(f"Fair Value = {metrics['Intercept']:.2f} + " + " + ".join(formula))
                                
                                # Plot
                                val_img = plot_valuation(res_df, ticker, metrics['R2'], selected_macros)
                                st.image(val_img, caption="Fair Value vs Actual Price")
                                
                                # Metrics Highlight
                                curr_price = res_df['Actual'].iloc[-1]
                                fv_price = res_df['Fair_Value'].iloc[-1]
                                
                                # Try to find future target
                                last_valid_fv_date = res_df.index[-1]
                                target_price = res_df['Fair_Value'].iloc[-1] # This is the furthest projected
                                
                                # Is target in future?
                                last_actual_date = stock_series.dropna().index[-1]
                                is_future = last_valid_fv_date > last_actual_date
                                
                                st.metric("Current Fair Value", f"{fv_price:.2f}", delta=f"{(fv_price-curr_price)/curr_price:.1%} vs Market")
                                
                                if is_future:
                                    days_out = (last_valid_fv_date - last_actual_date).days
                                    st.metric(f"Projected Target (+{days_out} days)", f"{target_price:.2f}", delta="Model Forecast", delta_color="inverse")
                                
                            else:
                                st.error("Model failed to train. (Possible causes: Insufficient Data Overlap or JJN delisted)")

else:
    st.write("👈 Enter a ticker and click Scan to start.")
