import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    """
    IEEE Research Component: Market Regime Classification Engine.
    Classifies market state into:
    - Bull Market (0)
    - Bear Market (1)
    - Sideways Market (2)
    - High Volatility (3)
    - Low Volatility (4)
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted = False

        self.regime_labels = {
            0: "Bull Market 📈",
            1: "Bear Market 📉",
            2: "Sideways Market ↔️",
            3: "High Volatility ⚡",
            4: "Low Volatility 🟢"
        }

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fits KMeans on Return Volatility & 20-SMA Ratio to assign Market Regimes."""
        if 'Volatility' not in df.columns or 'MA_20_ratio' not in df.columns:
            return np.zeros(len(df), dtype=int)

        features = df[['Volatility', 'MA_20_ratio']].fillna(0).values
        clusters = self.kmeans.fit_predict(features)
        self.is_fitted = True
        return clusters

    def detect_regime(self, prices: pd.Series) -> str:
        """Detects current market regime from a price series."""
        if len(prices) < 5:
            return "General Market"
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(min(10, len(returns))).std().fillna(0)
        ma_ratio = (prices / prices.rolling(min(20, len(prices))).mean() - 1).fillna(0)
        df_feat = pd.DataFrame({'Volatility': volatility, 'MA_20_ratio': ma_ratio}).dropna()
        if df_feat.empty:
            return "General Market"
        clusters = self.fit_predict(df_feat)
        latest_cluster = int(clusters[-1])
        return self.get_regime_name(latest_cluster)

    def get_regime_name(self, cluster_id: int) -> str:
        """Returns human-readable regime name."""
        return self.regime_labels.get(int(cluster_id), "General Market")
