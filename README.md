# Real-Time Stat Arb Dashboard (Quant Assignment)

## Live Demo
**[Click here to view the Live Dashboard](https://quant-assignment-submission-yduugn3xqcmyqtfutuvydp.streamlit.app/)**

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

* ## 🤖 AI Usage & Transparency

In compliance with the assignment guidelines, this project utilized LLM assistance (Gemini/ChatGPT) for the following purposes:

### 1. Code Generation & Structuring
- **Frontend:** Generating the Streamlit boilerplate code to ensure a modular "Dashboard" layout with sidebar controls and responsive charts.
- **Backend Logic:** Assisting in writing the `quant_engine.py` module to correctly implement OLS Regression (Hedge Ratio) and the Augmented Dickey-Fuller (ADF) test using `statsmodels`.

### 2. Architecture Design
- Generated the **Mermaid.js** code to visualize the system architecture (Ingestion → Sampling → Storage → Analytics → Frontend) for the required diagram deliverables.

### 3. Debugging
- Used to resolve Git merge conflicts ("refusing to merge unrelated histories") and troubleshoot Python TypeErrors during the integration of the analytics module.

### Sample Prompts Used
- *"Generate a robust architecture diagram showing ingestion, storage, and analytics flows using Mermaid code."*
- *"Create a Streamlit dashboard with a sidebar for parameters and a main area for Plotly charts."*
- *"Write a Python class using statsmodels to calculate OLS Hedge Ratio and ADF test statistics."*
- *"Fix this TypeError: calculate_spread_zscore() missing 1 required positional argument."*
