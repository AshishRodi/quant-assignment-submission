import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import threading
import time
import websocket
import json
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from collections import deque

# --- CONFIGURATION ---
SYMBOLS = ["btcusdt", "ethusdt"] 
SOCKET_URL = f"wss://stream.binance.com:9443/ws/{'/'.join([s + '@trade' for s in SYMBOLS])}"

# --- IN-MEMORY STORAGE ---
# We use a deque (double-ended queue) to store the last 10,000 ticks in RAM.
# This is much faster than SQLite for a live dashboard.
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=15000)

# Global reference for the thread to write to
DATA_BUFFER = st.session_state.data_buffer

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")

# --- BACKEND: INGESTION THREAD ---
@st.cache_resource
def start_background_ingestion():
    def on_message(ws, message):
        data = json.loads(message)
        # Append directly to the global buffer
        # Format: [Timestamp, Symbol, Price]
        DATA_BUFFER.append({
            'timestamp': datetime.now(),
            'symbol': data['s'],
            'price': float(data['p'])
        })

    def on_error(ws, error):
        print(f"WS Error: {error}")

    def run_websocket():
        while True:
            try:
                ws = websocket.WebSocketApp(SOCKET_URL,
                                            on_message=on_message,
                                            on_error=on_error)
                ws.run_forever()
            except Exception:
                time.sleep(2) 

    t = threading.Thread(target=run_websocket, daemon=True)
    t.start()
    return t

# Start the feed
start_background_ingestion()

# --- ANALYTICS ENGINE ---
def get_data_from_memory(minutes=60):
    # Convert list of dicts to DataFrame
    if len(DATA_BUFFER) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(list(DATA_BUFFER))
    
    # Filter by time
    cutoff = datetime.now() - timedelta(minutes=minutes)
    df = df[df['timestamp'] > cutoff]
    
    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=['timestamp', 'symbol'])
    df_pivot = df.pivot_table(index='timestamp', columns='symbol', values='price')
    
    # Resample to 1s
    df_resampled = df_pivot.resample('1s').last().ffill().dropna()
    return df_resampled

def calculate_metrics(df, symbol_y, symbol_x, window=30):
    if symbol_y not in df.columns or symbol_x not in df.columns or len(df) < window:
        return df, None

    y = df[symbol_y]
    x = df[symbol_x]
    x = sm.add_constant(x)
    
    try:
        model = OLS(y, x).fit()
        hedge_ratio = model.params.get(symbol_x, model.params[1]) 
    except:
        hedge_ratio = 1.0
    
    df['spread'] = df[symbol_y] - (hedge_ratio * df[symbol_x])
    roll_mean = df['spread'].rolling(window=window).mean()
    roll_std = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - roll_mean) / roll_std
    
    return df, hedge_ratio

# --- FRONTEND ---
st.sidebar.title("Configuration")
pair_1 = st.sidebar.selectbox("Asset Y", ["ETHUSDT", "BTCUSDT"], index=0)
pair_2 = st.sidebar.selectbox("Asset X", ["BTCUSDT", "ETHUSDT"], index=0)
window = st.sidebar.slider("Rolling Window", 10, 200, 60)
z_thresh = st.sidebar.number_input("Z-Score Threshold", value=2.0)

st.title("Real-Time Stat Arb Monitor")

# Status Indicator
st.sidebar.write("---")
if len(DATA_BUFFER) > 0:
    st.sidebar.success(f"System Status: LIVE ({len(DATA_BUFFER)} ticks)")
else:
    st.sidebar.warning("System Status: CONNECTING...")

placeholder = st.empty()

while True:
    with placeholder.container():
        # Fetch directly from RAM
        df = get_data_from_memory(minutes=5)
        
        if df.empty or len(df) < window:
            st.info(f"⚡ Establishing Feed... {len(df)}/{window} ticks collected.")
            time.sleep(1)
            continue
            
        try:
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, window)
            
            if df_analytics is None:
                continue

            latest = df_analytics.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{pair_1}", f"{latest[pair_1]:.2f}")
            col2.metric(f"{pair_2}", f"{latest[pair_2]:.2f}")
            col3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            
            z_val = latest['z_score']
            delta_color = "inverse" if abs(z_val) > z_thresh else "normal"
            col4.metric("Z-Score", f"{z_val:.2f}", delta_color=delta_color)

            if abs(z_val) > z_thresh:
                st.error(f"⚠️ ALERT: Z-Score Divergence! {z_val:.2f}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics[pair_1], name=pair_1), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], name="Z-Score", line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=z_thresh, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-z_thresh, line_dash="dash", line_color="red", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.write(f"Waiting for more data... ({e})")
            
        time.sleep(1)