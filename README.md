# Real-Time Stat Arb Dashboard (Quant Assignment)

## Live Demo
**[Click here to view the Live Dashboard](<YOUR_STREAMLIT_SHARE_LINK_HERE>)**

## Overview
This is a full-stack quantitative dashboard designed to monitor Statistical Arbitrage opportunities between BTC and ETH. It calculates the Hedge Ratio and Z-Score in real-time to identify mean-reversion trading signals.

## Key Features
* **Real-Time Ingestion:** Connects to Binance public data feeds.
* **Robust Architecture:** Implements a **"Fail-Safe" logic**. If the external API connection is unstable (common in cloud environments), the system automatically degrades to a **Simulation Mode** (Geometric Brownian Motion) to ensure the UI never freezes.
* **Zero-Lag Visualization:** Uses an in-memory `deque` buffer for O(1) read/write speeds, replacing traditional database locking for this high-frequency use case.

## Architecture
The application uses a **Producer-Consumer pattern** implemented via Python threading:
1.  **Producer (Background Thread):** Polls Binance via REST/WebSocket. It handles connection retries and SSL handshakes.
2.  **Storage (RAM):** A ring buffer (`collections.deque`) holding the last 15,000 ticks.
3.  **Consumer (Main Thread):** Streamlit engine that samples data, runs OLS regression (Statsmodels), and updates Plotly charts.

## How to Run Locally
1.  Clone the repository.
2.  Install requirements: `pip install -r requirements.txt`
3.  Run the app: `streamlit run app.py`

## Technologies
* **Python 3.9+**
* **Streamlit** (Frontend)
* **Statsmodels** (OLS Regression)
* **Plotly** (Interactive Charting)
