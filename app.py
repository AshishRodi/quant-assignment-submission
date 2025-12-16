import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AlphaStream Quant Engine", page_icon="📈", layout="wide")

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Data & Strategy")
    
    # Mode Selection
    data_source = st.radio("Select Source", ["Live Stream", "Upload CSV"], index=0)
    
    # Dynamic Inputs
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload OHLC Data (CSV)", type=['csv'])
        st.info("CSV must have columns: 'timestamp', 'close'")
    else:
        st.caption("Listening to Binance WebSocket...")
        ticker = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"])

    st.divider()
    
    # Analytics Parameters
    window_size = st.slider("Rolling Window (Period)", 5, 200, 20)
    z_threshold = st.number_input("Z-Score Threshold", value=2.0, step=0.1)

# --- 3. DATA PROCESSING FUNCTIONS ---
def calculate_analytics(df, window):
    """Computes Z-Score and Rolling Stats"""
    # Ensure we have numeric data
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Calculate Rolling Mean and Std Dev
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    
    # Calculate Z-Score: (Price - Mean) / Std
    df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    
    # Simple Signal: Sell if Z > Threshold, Buy if Z < -Threshold
    df['signal'] = np.where(df['z_score'] > z_threshold, 'SELL', 
                   np.where(df['z_score'] < -z_threshold, 'BUY', 'HOLD'))
    return df

# --- 4. MAIN APP LOGIC ---
st.title("⚡ Quant Analytics Dashboard")

# Initialize dataframe variable
df = pd.DataFrame()

# CASE A: USER UPLOADS CSV
if data_source == "Upload CSV":
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Basic standardization of column names (convert to lowercase)
            df.columns = [c.lower() for c in df.columns]
            
            # Check if required columns exist
            if 'close' in df.columns:
                # Run the analytics on the uploaded data
                df = calculate_analytics(df, window_size)
                
                # Show success message
                st.success(f"Successfully loaded {len(df)} rows from {uploaded_file.name}")
            else:
                st.error("CSV Error: Column 'close' not found. Please upload a valid OHLC file.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

# CASE B: LIVE STREAM (Dummy Data Generator for now)
else:
    # Simulating live data structure for the interface
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
    prices = 50000 + np.cumsum(np.random.randn(200) * 20)
    df = pd.DataFrame({'timestamp': dates, 'close': prices})
    df = calculate_analytics(df, window_size)

# --- 5. VISUALIZATION (Only if we have data) ---
if not df.empty:
    # Top Level Metrics (Based on the latest data point)
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Price", f"{latest['close']:.2f}")
    col2.metric("Rolling Mean", f"{latest['rolling_mean']:.2f}")
    col3.metric("Z-Score", f"{latest['z_score']:.2f}", delta_color="off")
    col4.metric("Signal", latest['signal'])

    # CHARTS
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=("Price Action", "Z-Score Analytics"))

    # Plot Price & Rolling Mean
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rolling_mean'], name='MA', line=dict(color='orange', width=1)), row=1, col=1)

    # Plot Z-Score
    fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', line=dict(color='purple')), row=2, col=1)
    
    # Add Threshold Lines
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
    fig.add_hline(y=-z_threshold, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Data Table Preview
    with st.expander("📄 View Raw Data"):
        st.dataframe(df.tail(100), use_container_width=True)

else:
    st.warning("Waiting for data... Please upload a CSV or switch to Live Stream.")