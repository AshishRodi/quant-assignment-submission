import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class QuantEngine:
    """
    Handles all quantitative analytics: Regression, Stationarity tests, and Signal generation.
    Decoupled from the frontend logic for better modularity.
    """
    
    @staticmethod
    def calculate_ols_hedge_ratio(y_series, x_series):
        """
        Calculates the Hedge Ratio (Beta) using OLS Regression.
        Formula: Y = beta * X + alpha
        """
        try:
            # Align indexes and drop missing values
            df = pd.concat([y_series, x_series], axis=1).dropna()
            Y = df.iloc[:, 0]
            X = df.iloc[:, 1]
            
            # Add constant for intercept
            X_const = sm.add_constant(X)
            
            model = sm.OLS(Y, X_const).fit()
            hedge_ratio = model.params.iloc[1]  # The slope coefficient (Beta)
            return hedge_ratio, model.summary()
        except Exception as e:
            return None, str(e)

    @staticmethod
    def run_adf_test(timeseries):
        """
        Performs Augmented Dickey-Fuller (ADF) test to check for stationarity.
        Returns p-value and critical values.
        """
        try:
            clean_series = timeseries.dropna()
            # ADF requires a minimum amount of data to be statistically valid
            if len(clean_series) < 15:
                return {"error": "Insufficient data"}
                
            result = adfuller(clean_series)
            p_value = result[1]
            
            return {
                "test_stat": result[0],
                "p_value": round(p_value, 4),
                "is_stationary": p_value < 0.05,  # Standard significance level
                "critical_values": result[4]
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def calculate_rolling_stats(series, window=20):
        """Calculates rolling mean and standard deviation."""
        return series.rolling(window=window).mean(), series.rolling(window=window).std()

    @staticmethod
    def calculate_spread_zscore(main_series, hedge_series, hedge_ratio, window=20):
        """
        Calculates the Spread and Z-Score based on the OLS Hedge Ratio.
        Spread = Price_A - (Hedge_Ratio * Price_B)
        Z-Score = (Spread - Rolling_Mean) / Rolling_Std
        """
        spread = main_series - (hedge_ratio * hedge_series)
        
        # Calculate Z-Score on the spread
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        z_score = (spread - spread_mean) / spread_std
        
        return spread, z_score