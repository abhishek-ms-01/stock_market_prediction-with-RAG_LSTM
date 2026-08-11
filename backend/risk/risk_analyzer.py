import numpy as np

def calculate_stock_risk_metrics(return_series: np.ndarray, prediction_prob: float) -> dict:
    """
    Computes Risk Score, Volatility Score, and Expected Return.
    """
    if return_series is None or len(return_series) == 0:
        volatility = 0.015
    else:
        volatility = float(np.std(return_series))

    # Expected Return Calculation (% movement scaled by prediction signal)
    expected_return = (prediction_prob - 0.50) * 2.0 * volatility * np.sqrt(5.0) * 100.0

    # Risk Score: Function of volatility and signal divergence
    risk_score_raw = volatility * 10.0 + abs(0.50 - prediction_prob) * 2.0
    if risk_score_raw < 0.25:
        risk_level = "Low Risk 🟢"
    elif risk_score_raw < 0.50:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "High Risk 🔴"

    return {
        "volatility_score": round(volatility, 4),
        "expected_return_pct": round(float(expected_return), 2),
        "risk_score": round(float(risk_score_raw), 4),
        "risk_level": risk_level
    }
