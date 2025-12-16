import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3
import threading
import time
import websocket
import json
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# --- CONFIGURATION ---
DB_NAME = "market_data.db"
# We track BTC and ETH
SYMBOLS = ["btcusdt", "ethusdt"] 
SOCKET_URL = f"wss://stream.binance.com:9443/ws/{'/'.join([s + '@trade' for s in SYMBOLS])}"

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")

# --- BACKEND: DATABASE & INGESTION ---
# @st.cache_resource ensures this runs only once, not on every user click
@st.cache_resource
def start_background_ingestion():
    # 1. Initialize DB with WAL mode for concurrency
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('''CREATE TABLE IF NOT EXISTS ticks 
                 (timestamp DATETIME, symbol TEXT, price REAL)''')
    conn.commit()
    conn.close()

    # 2. Define WebSocket Logic
    def on_message(ws, message):
        data = json.loads(message)
        symbol = data['s']
        price = float(data['p'])
        
        try:
            # Quick connect-write-close to minimize locking
            conn = sqlite3.connect(DB_NAME, check_same_thread=False)
            c = conn.cursor()
            c.execute("INSERT INTO ticks VALUES (?, ?, ?)", 
                      (datetime.now(), symbol, price))
            conn.commit()
            conn.close()
        except Exception:
            pass 

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
                time.sleep(2) # Retry delay

    # 3. Start the background thread
    t = threading.Thread(target=run_websocket, daemon=True)
    t.start()
    return t

# Start the feed immediately
start_background_ingestion()

# --- ANALYTICS ENGINE ---
def get_data(minutes=60):
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout = 1000")
        query_time = datetime.now() - timedelta(minutes=minutes)
        df = pd.read_sql(f"SELECT * FROM ticks WHERE timestamp > '{query_time}'", conn)
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])
        df_pivot = df.pivot_table(index='timestamp', columns='symbol', values='price')
        
        # The critical fix: .last() ensures we capture ticks within the second
        df_resampled = df_pivot.resample('1s').last().ffill().dropna()
        return df_resampled
    except:
        return pd.DataFrame()

def calculate_metrics(df, symbol_y, symbol_x, window=30):
    # Validation
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

# --- FRONTEND: DASHBOARD ---
st.sidebar.title("Configuration")
pair_1 = st.sidebar.selectbox("Asset Y", ["ETHUSDT", "BTCUSDT"], index=0)
pair_2 = st.sidebar.selectbox("Asset X", ["BTCUSDT", "ETHUSDT"], index=0)
window = st.sidebar.slider("Rolling Window (Seconds)", 10, 200, 60)
z_thresh = st.sidebar.number_input("Z-Score Threshold", value=2.0)

st.title("Real-Time Stat Arb Monitor")
st.caption("Live Data from Binance WebSocket")

placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data(minutes=5)
        
        if df.empty or len(df) < window:
            st.warning(f"Initializing Feed... Collecting Data ({len(df)}/{window} ticks)")
            time.sleep(1)
            continue
            
        try:
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, window)
            
            if df_analytics is None:
                st.warning("Buffering analytics...")
                continue

            latest = df_analytics.iloc[-1]
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{pair_1}", f"{latest[pair_1]:.2f}")
            col2.metric(f"{pair_2}", f"{latest[pair_2]:.2f}")
            col3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            
            z_val = latest['z_score']
            delta_color = "inverse" if abs(z_val) > z_thresh else "normal"
            col4.metric("Z-Score", f"{z_val:.2f}", delta_color=delta_color)

            # Alert Notification
            if abs(z_val) > z_thresh:
                st.error(f"⚠️ ALERT: Z-Score Divergence! Current: {z_val:.2f}")

            # Charts
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                row_heights=[0.7, 0.3], 
                                vertical_spacing=0.05)
            
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics[pair_1], 
                                     name=pair_1, line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], 
                                     name="Z-Score", line=dict(color='#9467bd', width=1.5)), row=2, col=1)
            
            # Add Alert Thresholds
            fig.add_hline(y=z_thresh, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-z_thresh, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Processing... ({e})")
            
        time.sleep(1)