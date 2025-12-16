import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from data_engine import get_data, calculate_metrics

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")
st.sidebar.title("Configuration")

# [cite_start]User controls as per requirement [cite: 25]
pair_1 = st.sidebar.selectbox("Asset Y (Dependent)", ["ETHUSDT", "BTCUSDT"], index=0)
pair_2 = st.sidebar.selectbox("Asset X (Independent)", ["BTCUSDT", "ETHUSDT"], index=0)
window = st.sidebar.slider("Rolling Window (Seconds)", 10, 200, 60)
z_thresh = st.sidebar.number_input("Z-Score Alert Threshold", value=2.0)
refresh_rate = st.sidebar.selectbox("Refresh Rate (ms)", [500, 1000, 5000], index=1)

st.title("Real-Time Stat Arb Monitor")
placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data(minutes=10)
        
        if df.empty or len(df) < window:
            st.warning(f"Waiting for data... (Have {len(df) if not df.empty else 0} data points)")
            time.sleep(1)
            continue
            
        try:
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, window)
        except Exception as e:
            st.error(f"Error computing metrics: {e}")
            time.sleep(1)
            continue

        if df_analytics is None:
            st.warning("Not enough data for rolling window.")
            continue

        latest = df_analytics.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{pair_1}", f"{latest[pair_1]:.2f}")
        col2.metric(f"{pair_2}", f"{latest[pair_2]:.2f}")
        col3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
        
        z_val = latest['z_score']
        delta_color = "normal"
        if abs(z_val) > z_thresh:
            delta_color = "inverse"
            st.error(f"ALERT: Z-Score Divergence! [cite_start]{z_val:.2f} > {z_thresh}") # Alert logic [cite: 19]
            
        col4.metric("Z-Score", f"{z_val:.2f}", delta_color=delta_color)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics[pair_1], name=pair_1), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], name="Z-Score", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=z_thresh, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-z_thresh, line_dash="dash", line_color="red", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        time.sleep(refresh_rate / 1000)