import pandas as pd
import numpy as np
import sqlite3
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from datetime import datetime, timedelta

DB_NAME = "market_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;') 
    c.execute('''CREATE TABLE IF NOT EXISTS ticks 
                 (timestamp DATETIME, symbol TEXT, price REAL)''')
    conn.commit()
    conn.close()

def save_tick(symbol, price):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO ticks VALUES (?, ?, ?)", 
                  (datetime.now(), symbol, price))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

def get_data(minutes=60):
    try:
        conn = sqlite3.connect(DB_NAME)
        # Timeout prevents "database is locked" errors
        conn.execute("PRAGMA busy_timeout = 1000") 
        
        query_time = datetime.now() - timedelta(minutes=minutes)
        df = pd.read_sql(f"SELECT * FROM ticks WHERE timestamp > '{query_time}'", conn)
        conn.close()

        print(f"DEBUG: Reading DB... Found {len(df)} rows.")

        if df.empty:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])
        
        df_pivot = df.pivot_table(index='timestamp', columns='symbol', values='price')
        
        # ---------------------------------------------------------
        # THE FIX: .last() aggregates the ticks, .ffill() fills gaps
        # ---------------------------------------------------------
        df_resampled = df_pivot.resample('1s').last().ffill().dropna()
        
        print(f"DEBUG: After Resampling... Found {len(df_resampled)} aligned rows.")
        return df_resampled
        
    except Exception as e:
        print(f"Read Error: {e}")
        return pd.DataFrame()

def calculate_metrics(df, symbol_y, symbol_x, window=30):
    # Ensure both symbols exist in the data
    if symbol_y not in df.columns or symbol_x not in df.columns:
        return df, None

    if len(df) < window:
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