import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class QuantEngine:
    @staticmethod
    def calculate_ols_hedge_ratio(y_series, x_series):
        """
        Calculates the Hedge Ratio (Beta) using OLS Regression.
        y = beta * x + alpha
        """
        try:
            # align indexes
            df = pd.concat([y_series, x_series], axis=1).dropna()
            Y = df.iloc[:, 0]
            X = df.iloc[:, 1]
            
            # Add constant for intercept
            X = sm.add_constant(X)
            
            model = sm.OLS(Y, X).fit()
            hedge_ratio = model.params.iloc[1] # The slope coefficient (Beta)
            return hedge_ratio, model.summary()
        except Exception as e:
            return None, str(e)

    @staticmethod
    def run_adf_test(timeseries):
        """
        Performs Augmented Dickey-Fuller test to check stationarity.
        Returns p-value and critical values.
        """
        try:
            # Drop NaNs
            clean_series = timeseries.dropna()
            if len(clean_series) < 20: # ADF needs sufficient data
                return {"p_value": "Insufficient Data", "is_stationary": False}
                
            result = adfuller(clean_series)
            p_value = result[1]
            
            return {
                "test_stat": result[0],
                "p_value": round(p_value, 4),
                "is_stationary": p_value < 0.05, # Standard threshold
                "critical_values": result[4]
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def calculate_rolling_correlation(series_a, series_b, window=30):
        """
        Computes rolling correlation between two assets.
        """
        return series_a.rolling(window=window).corr(series_b)

    @staticmethod
    def calculate_spread_zscore(main_series, hedge_series, hedge_ratio):
        """
        Calculates the Spread and Z-Score based on the Hedge Ratio.
        Spread = Price_A - (Hedge_Ratio * Price_B)
        """
        spread = main_series - (hedge_ratio * hedge_series)
        z_score = (spread - spread.mean()) / spread.std()
        return spread, z_score