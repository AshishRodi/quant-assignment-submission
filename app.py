import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import threading
import time
import requests
import json
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from collections import deque

# --- CONFIGURATION ---
SYMBOLS = ["btcusdt", "ethusdt"] 
# We use REST API for Cloud Deployment to avoid Firewall/IP blocks
REST_URL = "https://api.binance.com/api/v3/ticker/price"

# --- SESSION STATE ---
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=15000)
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = deque(maxlen=20)

DATA = st.session_state.data_buffer
LOGS = st.session_state.log_buffer

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")

def log_message(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    LOGS.append(f"[{timestamp}] {msg}")

# --- BACKEND: REST POLLING THREAD ---
@st.cache_resource
def start_background_ingestion():
    class BackgroundWorker:
        def __init__(self):
            self.thread = threading.Thread(target=self.run_poller, daemon=True)
            self.thread.start()

        def run_poller(self):
            log_message("Starting REST API Poller...")
            while True:
                try:
                    # Fetch prices for both symbols
                    for sym in SYMBOLS:
                        # Binance API call
                        params = {'symbol': sym.upper()}
                        r = requests.get(REST_URL, params=params, timeout=2)
                        data = r.json()
                        
                        price = float(data['price'])
                        
                        # Append to buffer
                        DATA.append({
                            'timestamp': datetime.now(),
                            'symbol': sym,
                            'price': price
                        })
                    
                    # Log success occasionally
                    if len(DATA) % 60 == 0:
                        log_message(f"Polled {len(DATA)} ticks...")
                        
                    time.sleep(1) # 1-second Interval
                    
                except Exception as e:
                    log_message(f"Polling Error: {e}")
                    time.sleep(2)

    return BackgroundWorker()

# Start the worker
worker = start_background_ingestion()

# --- ANALYTICS ENGINE ---
def get_data_from_memory(minutes=60):
    if len(DATA) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(list(DATA))
    cutoff = datetime.now() - timedelta(minutes=minutes)
    df = df[df['timestamp'] > cutoff]
    
    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=['timestamp', 'symbol'])
    df_pivot = df.pivot_table(index='timestamp', columns='symbol', values='price')
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
st.title("Real-Time Stat Arb Monitor (Cloud Demo)")

# Sidebar Debugger
st.sidebar.header("System Health")
if len(DATA) > 0:
    st.sidebar.success(f"Status: LIVE FEED ({len(DATA)} ticks)")
else:
    st.sidebar.warning("Status: CONNECTING...")

st.sidebar.subheader("Debug Log")
for msg in reversed(list(LOGS)):
    st.sidebar.text(msg)

placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data_from_memory(minutes=5)
        
        if df.empty or len(df) < 10: 
            st.info(f"⚡ Fetching Market Data... {len(df)} ticks collected.")
            time.sleep(1)
            continue
            
        try:
            # We can run analytics with fewer points for the demo to show UI faster
            min_window = 10 if len(df) < 30 else 30
            
            pair_1, pair_2 = "ETHUSDT", "BTCUSDT"
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, min_window)
            
            if df_analytics is None: 
                continue

            latest = df_analytics.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ETH Price", f"{latest['ETHUSDT']:.2f}")
            col2.metric("BTC Price", f"{latest['BTCUSDT']:.2f}")
            col3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            col4.metric("Z-Score", f"{latest['z_score']:.2f}")

            # Charts
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics[pair_1], name=pair_1), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], name="Z-Score", line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="red", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"UI Error: {e}")
            
        time.sleep(1)