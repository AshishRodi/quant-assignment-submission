import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import the logic module we just created
try:
    from quant_engine import QuantEngine
except ImportError:
    st.error("🚨 Missing 'quant_engine.py'. Please create this file to run the analytics.")
    st.stop()

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AlphaStream Quant Engine",
    page_icon="📈",
    layout="wide",  # Professional dashboard layout
    initial_sidebar_state="expanded"
)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Strategy Controls")
    
    # Input Source
    data_source = st.radio("Data Source", ["Live Stream (Simulation)", "Upload CSV"], index=0)
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload OHLC CSV", type=['csv'])
        st.info("💡 CSV requires 'close' column. If only one asset is uploaded, a synthetic benchmark is generated for OLS.")
    else:
        st.caption("📡 Simulating Binance WebSocket Feed...")
        ticker = st.selectbox("Primary Asset", ["BTCUSDT", "ETHUSDT"])
        benchmark = st.selectbox("Benchmark Asset", ["ETHUSDT", "BTCUSDT"], index=0)
    
    st.divider()
    
    # Analytics Parameters [cite: 25]
    st.markdown("### 🎛️ Parameters")
    window_size = st.slider("Rolling Window", 10, 100, 20)
    z_threshold = st.number_input("Z-Score Threshold", value=2.0, step=0.1)
    
    st.divider()
    
    # Action Buttons
    if st.button("🔄 Reset / Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# --- 3. DATA PREPARATION ---
st.title("⚡ Quant Developer Evaluation Dashboard")

df = pd.DataFrame()

# CASE A: LIVE SIMULATION
if data_source == "Live Stream (Simulation)":
    # Generate synthetic correlated data for demo purposes
    # In production, this would be replaced by the Database Query
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
    
    # Create two correlated series (Cointegrated)
    np.random.seed(42)
    # Asset A (e.g., BTC)
    price_a = 50000 + np.cumsum(np.random.randn(200) * 50)
    # Asset B (e.g., ETH) - correlated with A plus some noise
    price_b = (price_a * 0.05) + np.random.normal(0, 50, 200) + 1000
    
    df = pd.DataFrame({'timestamp': dates, 'close': price_a, 'benchmark_close': price_b})

# CASE B: CSV UPLOAD
elif data_source == "Upload CSV" and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' in df.columns:
            # If user didn't upload a benchmark, simulate one so OLS doesn't break
            if 'benchmark_close' not in df.columns:
                st.toast("⚠️ No benchmark column found. Generating synthetic benchmark for OLS demo.", icon="ℹ️")
                noise = np.random.normal(0, df['close'].std() * 0.1, len(df))
                df['benchmark_close'] = df['close'] * 0.05 + noise
        else:
            st.error("❌ CSV must contain a 'close' column.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# --- 4. ANALYTICS EXECUTION ---
if not df.empty:
    # 1. Calculate OLS Hedge Ratio
    hedge_ratio, model_summary = QuantEngine.calculate_ols_hedge_ratio(df['close'], df['benchmark_close'])
    
    if hedge_ratio:
        # 2. Calculate Spread and Z-Score using the engine
        df['spread'], df['z_score'] = QuantEngine.calculate_spread_zscore(
            df['close'], df['benchmark_close'], hedge_ratio, window_size
        )
        
        # 3. Generate Signals
        df['signal'] = np.where(df['z_score'] > z_threshold, 'SELL', 
                       np.where(df['z_score'] < -z_threshold, 'BUY', 'HOLD'))

        # --- 5. VISUALIZATION LAYOUT ---
        
        # Top Metrics Row
        latest = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"{latest['close']:.2f}")
        c2.metric("Hedge Ratio (Beta)", f"{hedge_ratio:.4f}")
        c3.metric("Current Z-Score", f"{latest['z_score']:.2f}", 
                 delta="Overbought" if latest['z_score'] > z_threshold else "Oversold" if latest['z_score'] < -z_threshold else "Neutral",
                 delta_color="inverse")
        c4.metric("Signal", latest['signal'])

        # Main Charts (Subplots)
        tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📊 Statistical Tests", "📄 Raw Data"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                subplot_titles=("Price Action & Spread", "Z-Score Mean Reversion"))

            # Row 1: Prices
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Primary Asset', line=dict(color='blue')), row=1, col=1)
            # Visualize the "Spread" roughly on a secondary axis or just show prices? 
            # Let's show the secondary asset scaled
            fig.add_trace(go.Scatter(x=df.index, y=df['benchmark_close'] * (1/hedge_ratio), 
                                    name=f'Benchmark (Scaled by Beta)', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

            # Row 2: Z-Score
            fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', line=dict(color='purple')), row=2, col=1)
            
            # Thresholds
            fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-z_threshold, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=600, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("### 🧪 Stationarity & Cointegration Tests")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("ADF Test (Stationarity)")
                st.write("Checking if the **Spread** is stationary (mean-reverting).")
                adf_res = QuantEngine.run_adf_test(df['spread'].dropna())
                
                if "error" not in adf_res:
                    st.json(adf_res)
                    if adf_res['is_stationary']:
                        st.success("✅ The Spread is Stationary. Mean reversion strategy is viable.")
                    else:
                        st.error("❌ The Spread is Non-Stationary. Caution advised.")
                else:
                    st.warning("Not enough data for ADF test.")

            with col_b:
                st.subheader("OLS Regression Results")
                with st.expander("View Full Statistical Summary"):
                    st.text(model_summary)

        with tab3:
            st.dataframe(df.tail(100), use_container_width=True)
            
            # Download Button [cite: 33]
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Processed Data", csv, "analytics_export.csv", "text/csv")
            
else:
    st.info("👋 Welcome! Please upload a CSV file or enable the Live Stream simulation to begin.")