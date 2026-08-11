import pandas as pd
from risk.risk_analyzer import calculate_stock_risk_metrics

class PortfolioRecommendationEngine:
    """
    IEEE Research Component: Portfolio Intelligence & Recommendation Ranking Engine.
    Evaluates multiple NSE stocks and ranks them into BUY, HOLD, or SELL categories
    based on prediction confidence, expected return, and risk score.
    """

    def rank_portfolio(self, stock_forecasts: list) -> pd.DataFrame:
        """
        Input: list of dicts [{'ticker': 'RELIANCE.NS', 'prob': 0.85, 'returns': np.array([])}]
        Output: Ranked DataFrame with BUY / HOLD / SELL recommendations.
        """
        results = []
        for item in stock_forecasts:
            ticker = item.get("ticker", "N/A")
            prob = item.get("prob", 0.50)
            returns = item.get("returns", None)

            risk_metrics = calculate_stock_risk_metrics(returns, prob)
            exp_ret = risk_metrics["expected_return_pct"]
            risk_lvl = risk_metrics["risk_level"]

            # Decision Logic
            if prob >= 0.65 and exp_ret > 0.5:
                recommendation = "BUY 🟢"
            elif prob <= 0.35 or exp_ret < -0.5:
                recommendation = "SELL 🔴"
            else:
                recommendation = "HOLD 🟡"

            results.append({
                "Stock Ticker": ticker,
                "Signal": "UP 📈" if prob > 0.5 else "DOWN 📉",
                "Confidence": f"{max(prob, 1.0 - prob):.2%}",
                "Expected Return (%)": f"{exp_ret:+.2f}%",
                "Risk Rating": risk_lvl,
                "Recommendation": recommendation
            })

        df_rank = pd.DataFrame(results)
        return df_rank
