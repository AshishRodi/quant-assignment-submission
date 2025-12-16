import websocket
import json
from data_engine import init_db, save_tick

# [cite_start]Using Binance WebSocket for real-time tick data [cite: 12]
SYMBOLS = ["btcusdt", "ethusdt"] 
SOCKET_URL = f"wss://stream.binance.com:9443/ws/{'/'.join([s + '@trade' for s in SYMBOLS])}"

def on_message(ws, message):
    data = json.loads(message)
    symbol = data['s']
    price = float(data['p'])
    print(f"Tick: {symbol} @ {price}")
    save_tick(symbol, price)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Connection Closed ###")

def on_open(ws):
    print(f"### Connected to Binance: {SYMBOLS} ###")

if __name__ == "__main__":
    init_db()
    print("Database initialized. Starting Stream...")
    ws = websocket.WebSocketApp(SOCKET_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()