import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import threading
import time
import websocket
import json
import ssl
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from collections import deque

# --- CONFIGURATION ---
SYMBOLS = ["btcusdt", "ethusdt"] 
SOCKET_URL = f"wss://stream.binance.com:9443/ws/{'/'.join([s + '@trade' for s in SYMBOLS])}"

# --- SESSION STATE SETUP ---
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=15000)
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = deque(maxlen=20)

# Shortcuts
DATA = st.session_state.data_buffer
LOGS = st.session_state.log_buffer

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")

# --- LOGGING HELPER ---
def log_message(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    LOGS.append(f"[{timestamp}] {msg}")

# --- BACKEND: INGESTION THREAD ---
@st.cache_resource
def start_background_ingestion():
    # We use a container class to keep track of thread status
    class BackgroundWorker:
        def __init__(self):
            self.thread = threading.Thread(target=self.run_websocket, daemon=True)
            self.thread.start()

        def on_message(self, ws, message):
            try:
                data = json.loads(message)
                DATA.append({
                    'timestamp': datetime.now(),
                    'symbol': data['s'],
                    'price': float(data['p'])
                })
            except Exception as e:
                pass

        def on_error(self, ws, error):
            log_message(f"ERROR: {error}")

        def on_open(self, ws):
            log_message("WebSocket Connection OPENED")

        def on_close(self, ws, close_status_code, close_msg):
            log_message(f"WebSocket CLOSED: {close_msg}")

        def run_websocket(self):
            log_message("Starting WebSocket Thread...")
            while True:
                try:
                    ws = websocket.WebSocketApp(SOCKET_URL,
                                                on_open=self.on_open,
                                                on_message=self.on_message,
                                                on_error=self.on_error,
                                                on_close=self.on_close)
                    # CRITICAL FIX: Bypass SSL verification for Cloud environments
                    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                except Exception as e:
                    log_message(f"Thread Crash: {e}")
                    time.sleep(5)

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
st.title("Real-Time Stat Arb Monitor")

# Sidebar Debugger
st.sidebar.header("System Health")
if len(DATA) > 0:
    st.sidebar.success(f"Status: RECEIVING DATA ({len(DATA)} ticks)")
else:
    st.sidebar.warning("Status: WAITING FOR FEED...")

st.sidebar.subheader("Debug Log")
# Display last 10 logs reversed
for msg in reversed(list(LOGS)):
    st.sidebar.text(msg, help="System logs from backend thread")

placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data_from_memory(minutes=5)
        
        # UI State Handling
        if df.empty or len(df) < 30: # Lowered threshold for faster visual check
            st.info(f"⚡ Feed Initializing... {len(df)} ticks in memory. (Check Sidebar Logs if stuck)")
            time.sleep(1)
            continue
            
        try:
            # Quick metric calc
            pair_1, pair_2 = "ETHUSDT", "BTCUSDT"
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, 30)
            
            if df_analytics is None: 
                continue

            latest = df_analytics.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("ETH Price", f"{latest['ETHUSDT']:.2f}")
            col2.metric("BTC Price", f"{latest['BTCUSDT']:.2f}")
            col3.metric("Z-Score", f"{latest['z_score']:.2f}")

            # Simple Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], name="Z-Score"))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"UI Error: {e}")
            
        time.sleep(1)