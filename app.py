import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import threading
import time
import requests
import json
import random
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from collections import deque

# --- CONFIGURATION ---
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
REST_URL = "https://api.binance.com/api/v3/ticker/price"

# --- SESSION STATE ---
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=15000)
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = deque(maxlen=20)
if 'sim_prices' not in st.session_state:
    # Starting prices for simulation mode
    st.session_state.sim_prices = {"BTCUSDT": 65000.0, "ETHUSDT": 3500.0}

DATA = st.session_state.data_buffer
LOGS = st.session_state.log_buffer
SIM_PRICES = st.session_state.sim_prices

st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")

def log_message(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    LOGS.append(f"[{timestamp}] {msg}")

# --- BACKEND: ROBUST INGESTION THREAD ---
@st.cache_resource
def start_background_ingestion():
    class BackgroundWorker:
        def __init__(self):
            self.thread = threading.Thread(target=self.run_poller, daemon=True)
            self.thread.start()

        def generate_synthetic_data(self):
            """Generates realistic random walk data if API fails"""
            for sym in SYMBOLS:
                # Random walk: Price * (1 + random percentage)
                shock = np.random.normal(0, 0.0005) # 0.05% volatility
                SIM_PRICES[sym] = SIM_PRICES[sym] * (1 + shock)
                
                DATA.append({
                    'timestamp': datetime.now(),
                    'symbol': sym,
                    'price': SIM_PRICES[sym],
                    'source': 'SIMULATION' # Tag source
                })

        def run_poller(self):
            log_message("Starting Fail-Safe Poller...")
            consecutive_errors = 0
            
            while True:
                try:
                    # Attempt Real Data Fetch
                    for sym in SYMBOLS:
                        params = {'symbol': sym}
                        # Short timeout (1s) so we don't hang
                        r = requests.get(REST_URL, params=params, timeout=1) 
                        
                        if r.status_code == 200:
                            data = r.json()
                            price = float(data['price'])
                            SIM_PRICES[sym] = price # Sync sim price to real
                            DATA.append({
                                'timestamp': datetime.now(),
                                'symbol': sym,
                                'price': price,
                                'source': 'LIVE'
                            })
                            consecutive_errors = 0 # Reset error count
                        else:
                            raise Exception(f"Status {r.status_code}")

                except Exception as e:
                    consecutive_errors += 1
                    # If we fail twice, start faking it so the user sees something
                    if consecutive_errors > 2:
                        if consecutive_errors == 3:
                            log_message(f"⚠️ Connection Unstable. Switching to Simulation Mode.")
                        self.generate_synthetic_data()
                    else:
                        log_message(f"Retrying connection... ({e})")
                
                time.sleep(1)

    return BackgroundWorker()

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
        # FIX 1: Use .iloc[1] instead of [1] to avoid Pandas Future Warning
        hedge_ratio = model.params.get(symbol_x, model.params.iloc[1]) 
    except:
        hedge_ratio = 1.0
    
    df['spread'] = df[symbol_y] - (hedge_ratio * df[symbol_x])
    roll_mean = df['spread'].rolling(window=window).mean()
    roll_std = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - roll_mean) / roll_std
    
    return df, hedge_ratio

# --- FRONTEND ---
st.title("Real-Time Stat Arb Monitor")

# Status Bar
if len(DATA) > 0:
    last_point = DATA[-1]
    source = last_point.get('source', 'UNKNOWN')
    color = "green" if source == 'LIVE' else "orange"
    st.markdown(f"**Status:** <span style='color:{color}'>{source} DATA FEED ACTIVE</span> | Ticks: {len(DATA)}", unsafe_allow_html=True)
else:
    st.warning("Initializing System...")

# Debug Log Expander
with st.sidebar.expander("Debug Logs", expanded=True):
    for msg in reversed(list(LOGS)):
        st.text(msg)

placeholder = st.empty()

while True:
    with placeholder.container():
        df = get_data_from_memory(minutes=5)
        
        if df.empty or len(df) < 5: 
            st.info(f"⚡ Booting Strategy Engine... ({len(df)} ticks)")
            time.sleep(1)
            continue
            
        try:
            pair_1, pair_2 = "ETHUSDT", "BTCUSDT"
            df_analytics, hedge_ratio = calculate_metrics(df, pair_1, pair_2, 10)
            
            if df_analytics is None: 
                continue

            latest = df_analytics.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ETH", f"{latest['ETHUSDT']:.2f}")
            col2.metric("BTC", f"{latest['BTCUSDT']:.2f}")
            col3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            
            z_val = latest['z_score']
            delta_color = "inverse" if abs(z_val) > 2.0 else "normal"
            col4.metric("Z-Score", f"{z_val:.2f}", delta_color=delta_color)

            # Charts
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics[pair_1], name=pair_1), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_analytics.index, y=df_analytics['z_score'], name="Z-Score", line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="red", row=2, col=1)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            
            # FIX 2: Replaced use_container_width=True with simple auto-sizing
            # Usually st.plotly_chart handles this by default now, or we pass key params.
            # If the log strictly said width='stretch', we use that (Streamlit >1.40)
            try:
                st.plotly_chart(fig, width="stretch") 
            except:
                # Fallback for older streamlit versions just in case
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization Error: {e}")
            
        time.sleep(1)