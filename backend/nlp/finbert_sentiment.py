import os
import torch
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_NLP_AVAILABLE = True
except ImportError:
    TRANSFORMERS_NLP_AVAILABLE = False

from sentiment.vader_sentiment import get_sentiment

class FinBERTSentimentAnalyzer:
    """
    FinBERT Financial Sentiment Analysis Pipeline.
    Computes Positive, Negative, Neutral probabilities, Confidence Score, and compound Sentiment Score.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """Loads FinBERT model weights if available."""
        if not TRANSFORMERS_NLP_AVAILABLE:
            print("[FinBERT Warning] HuggingFace transformers not installed. Using VADER NLP engine fallback.")
            return

        try:
            print(f"[FinBERT] Initializing financial sentiment model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.is_loaded = True
            print("[FinBERT] Successfully loaded FinBERT neural model weights.")
        except Exception as e:
            print(f"[FinBERT Warning] Could not load FinBERT pretrained model ({e}). Using VADER fallback.")
            self.model = None
            self.is_loaded = False

    def predict_sentiment(self, text: str) -> dict:
        """
        Analyzes financial text and returns:
        - positive_prob (0.0 to 1.0)
        - negative_prob (0.0 to 1.0)
        - neutral_prob (0.0 to 1.0)
        - confidence_score (0.0 to 1.0)
        - sentiment_score (-1.0 to +1.0)
        """
        if not text or not str(text).strip():
            return {
                "positive_prob": 0.33,
                "negative_prob": 0.33,
                "neutral_prob": 0.34,
                "confidence_score": 0.34,
                "sentiment_score": 0.0
            }

        # Neural FinBERT Inference
        if self.is_loaded and self.model is not None:
            try:
                inputs = self.tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()

                # FinBERT order: [positive, negative, neutral]
                pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
                conf = float(np.max(probs))
                compound = pos - neg

                return {
                    "positive_prob": round(pos, 4),
                    "negative_prob": round(neg, 4),
                    "neutral_prob": round(neu, 4),
                    "confidence_score": round(conf, 4),
                    "sentiment_score": round(compound, 4)
                }
            except Exception as e:
                print(f"[FinBERT Inference Error]: {e}")

        # VADER Fallback
        vader_comp = get_sentiment(text)
        if vader_comp > 0.05:
            pos, neg, neu = (0.5 + vader_comp/2.0), (0.5 - vader_comp/2.0), 0.1
        elif vader_comp < -0.05:
            pos, neg, neu = (0.5 + vader_comp/2.0), (0.5 - vader_comp/2.0), 0.1
        else:
            pos, neg, neu = 0.1, 0.1, 0.8

        return {
            "positive_prob": round(max(0.0, min(1.0, pos)), 4),
            "negative_prob": round(max(0.0, min(1.0, neg)), 4),
            "neutral_prob": round(max(0.0, min(1.0, neu)), 4),
            "confidence_score": round(abs(vader_comp), 4),
            "sentiment_score": round(vader_comp, 4)
        }
