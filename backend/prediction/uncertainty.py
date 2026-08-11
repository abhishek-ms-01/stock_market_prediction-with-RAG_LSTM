import numpy as np
import tensorflow as tf

class MonteCarloUncertaintyEstimator:
    """
    IEEE Research Component: Bayesian Monte Carlo Dropout Uncertainty Estimator.
    Performs stochastic forward passes (N=50) with active dropout to quantify epistemic uncertainty.
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples

    def estimate_uncertainty(self, model: tf.keras.Model, input_sequence: np.ndarray) -> dict:
        """
        Runs N=50 stochastic forward passes with dropout training=True.
        Returns:
        - prediction_prob (mean probability)
        - confidence_pct (1.0 - std_dev)
        - uncertainty_variance (variance of predictions)
        - is_high_uncertainty (True if std_dev > 0.15)
        """
        if model is None or input_sequence is None:
            return {
                "prediction_prob": 0.50,
                "confidence_pct": 50.0,
                "uncertainty_variance": 0.05,
                "is_high_uncertainty": False
            }

        predictions = []
        for _ in range(self.n_samples):
            # Pass training=True to keep dropout active during inference
            pred = model(input_sequence, training=True).numpy()[0][0]
            predictions.append(float(pred))

        mean_prob = float(np.mean(predictions))
        std_dev = float(np.std(predictions))
        variance = float(np.var(predictions))

        confidence_pct = max(0.0, min(100.0, (1.0 - std_dev) * 100.0))
        is_high_uncertainty = std_dev > 0.15

        return {
            "prediction_prob": round(mean_prob, 4),
            "confidence_pct": round(confidence_pct, 2),
            "uncertainty_variance": round(variance, 6),
            "std_dev": round(std_dev, 4),
            "is_high_uncertainty": is_high_uncertainty
        }
