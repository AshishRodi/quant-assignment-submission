import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import the logic module
try:
    from quant_engine import QuantEngine
except ImportError:
    st.error("🚨 Missing 'quant_engine.py'. Please create this file.")
    st.stop()

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AlphaStream Quant Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Strategy Controls")
    
    data_source = st.radio("Data Source", ["Live Stream (Simulation)", "Upload CSV"], index=0)
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload OHLC CSV", type=['csv'])
        st.info("💡 CSV requires 'close' column.")
    else:
        st.caption("📡 Simulating Binance WebSocket Feed...")
        ticker = st.selectbox("Primary Asset", ["BTCUSDT", "ETHUSDT"])
        benchmark = st.selectbox("Benchmark Asset", ["ETHUSDT", "BTCUSDT"], index=0)
        stop_btn = st.button("⏹ Stop Stream")
    
    st.divider()
    
    # Analytics Parameters
    st.markdown("### 🎛️ Parameters")
    window_size = st.slider("Rolling Window", 10, 100, 20)
    z_threshold = st.number_input("Z-Score Threshold", value=1.5, step=0.1)
    
    st.divider()
    if st.button("🔄 Reset / Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# --- 3. DASHBOARD LOGIC ---
st.title("⚡ Quant Developer Evaluation Dashboard")
dashboard_placeholder = st.empty()

def render_dashboard(df, window_size, z_threshold):
    """Helper function to draw the stats and charts"""
    if df.empty: return

    # Calculate Analytics
    hedge_ratio, model_summary = QuantEngine.calculate_ols_hedge_ratio(df['close'], df['benchmark_close'])
    
    if hedge_ratio:
        df['spread'], df['z_score'] = QuantEngine.calculate_spread_zscore(
            df['close'], df['benchmark_close'], hedge_ratio, window_size
        )
        
        # Signal Logic
        df['signal'] = np.where(df['z_score'] > z_threshold, 'SELL', 
                       np.where(df['z_score'] < -z_threshold, 'BUY', 'HOLD'))

        latest = df.iloc[-1]
        
        # Color logic
        sig_color = "normal"
        if latest['signal'] == "SELL": sig_color = "inverse"
        if latest['signal'] == "BUY": sig_color = "off"

        with dashboard_placeholder.container():
            # Top Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest Price", f"{latest['close']:.2f}")
            c2.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            c3.metric("Z-Score", f"{latest['z_score']:.2f}", delta_color="off")
            c4.metric("Signal", latest['signal'], delta=latest['signal'], delta_color=sig_color)

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📊 Statistical Tests", "📄 Raw Data"])
            
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                    subplot_titles=("Price Action", "Z-Score Mean Reversion"))

                # Row 1: Prices
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Asset A', line=dict(color='#2962FF')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['benchmark_close'] * (1/hedge_ratio), 
                                        name='Asset B (Scaled)', line=dict(color='#FF6D00', dash='dot')), row=1, col=1)

                # Row 2: Z-Score
                fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', line=dict(color='#6200EA')), row=2, col=1)
                fig.add_hline(y=z_threshold, line_dash="dash", line_color="#FF5252", row=2, col=1)
                fig.add_hline(y=-z_threshold, line_dash="dash", line_color="#00E676", row=2, col=1)
                
                fig.update_layout(height=600, margin=dict(t=30, b=0, l=0, r=0))
                
                # Plotly Chart with unique key to prevent glitching
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{len(df)}")
                
            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("ADF Test")
                    adf_res = QuantEngine.run_adf_test(df['spread'].dropna())
                    if "error" not in adf_res:
                        st.json(adf_res)
                with col_b:
                    st.subheader("OLS Summary")
                    with st.expander("View Details"):
                        st.text(model_summary)

            with tab3:
                st.dataframe(df.tail(100), use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                
                # FIXED: Unique key added here to prevent DuplicateElementId error
                st.download_button(
                    "⬇️ Download CSV", 
                    csv, 
                    "analytics.csv", 
                    "text/csv",
                    key=f"download_btn_{len(df)}_{time.time()}"
                )

# --- 4. EXECUTION ---

# CASE A: LIVE SIMULATION
if data_source == "Live Stream (Simulation)":
    if 'live_data' not in st.session_state:
        # Start with history
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
        np.random.seed(42)
        price_a = 50000 + np.cumsum(np.random.randn(200) * 50)
        noise = np.random.normal(0, 50, 200)
        price_b = (price_a * 0.05) + noise + 1000
        st.session_state['live_data'] = pd.DataFrame({'timestamp': dates, 'close': price_a, 'benchmark_close': price_b})
        
    if not stop_btn:
        while True:
            df = st.session_state['live_data']
            last_row = df.iloc[-1]
            new_time = last_row['timestamp'] + pd.Timedelta(minutes=1)
            
            # Random Walk Logic
            move_a = np.random.randn() * 40
            move_b = (move_a * 0.05) + (np.random.randn() * 10)
            
            new_row = pd.DataFrame({
                'timestamp': [new_time], 
                'close': [last_row['close'] + move_a], 
                'benchmark_close': [last_row['benchmark_close'] + move_b]
            })
            
            updated_df = pd.concat([df, new_row], ignore_index=True).tail(300)
            st.session_state['live_data'] = updated_df
            
            render_dashboard(updated_df, window_size, z_threshold)
            time.sleep(1) # Refresh rate
            
    else:
        st.warning("Stream Stopped.")
        render_dashboard(st.session_state['live_data'], window_size, z_threshold)

# CASE B: CSV UPLOAD
elif data_source == "Upload CSV" and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.lower() for c in df.columns]
        
        if 'close' in df.columns:
            if 'benchmark_close' not in df.columns:
                st.toast("⚠️ Generating synthetic benchmark for OLS demo.", icon="ℹ️")
                noise = np.random.normal(0, df['close'].std() * 0.1, len(df))
                df['benchmark_close'] = df['close'] * 0.05 + noise
            
            render_dashboard(df, window_size, z_threshold)
        else:
            st.error("❌ CSV must contain a 'close' column.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")