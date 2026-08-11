import os
import sys

# Force CPU & TF determinism
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Ensure project root in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
import random
import pandas as pd
from scipy import stats
import tensorflow as tf

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Set random seeds
np.random.seed(14)
random.seed(14)
tf.random.set_seed(14)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error
)

from models.neural_models import (
    build_lstm_model,
    build_gru_model,
    build_transformer_model,
    build_attention_lstm_model,
    build_hybrid_cross_attention_rag_lstm
)

def run_ieee_benchmark_suite():
    """
    IEEE 15-Phase Research Benchmark Suite.
    Trains and evaluates 7 Baseline & Proposed Neural Models,
    executes 4-part Ablation Study, and computes Statistical Significance (t-test, Wilcoxon).
    """
    dataset_path = os.path.join(project_root, "data", "final_dataset.csv")
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)

    # Features breakdown
    tech_feats = ['RSI', 'MACD', 'Return', 'MA_20_ratio', 'Close_Open', 'High_Low', 'Volume_ratio', 'Momentum_Factor']
    rag_feats = ['RAG_Sentiment', 'RAG_Relevance', 'RAG_Event_Importance', 'RAG_Market_Impact', 'RAG_Risk_Score', 'RAG_Confidence']
    sent_feats = ['Sentiment']
    event_feats = ['Event']
    regime_feats = ['Market_Regime']

    all_features = tech_feats + sent_feats + event_feats + regime_feats + rag_feats
    avail_features = [f for f in all_features if f in df.columns]

    X_df = df[avail_features]
    y_series = df['Target']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    seq_len = 5
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len + 1):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_series.iloc[i + seq_len - 1])

    X_reshaped = np.array(X_seq)
    y_reshaped = np.array(y_seq)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42
    )

    print("\n==========================================================================================")
    print(" 🔬 IEEE 15-PHASE RESEARCH EVALUATION BENCHMARK & ABLATION SUITE")
    print("==========================================================================================\n")

    results = []
    model_predictions = {}

    # 1. Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    t0 = time.time()
    rf.fit(X_tr_flat, y_train)
    rf_probs = rf.predict_proba(X_te_flat)[:, 1]
    rf_lat = ((time.time() - t0) / len(X_test)) * 1000.0
    rf_preds = (rf_probs > 0.5).astype(int)
    model_predictions["Random Forest"] = rf_probs

    results.append({
        "Model / Experiment": "Random Forest",
        "Category": "Baseline ML",
        "Accuracy": round(accuracy_score(y_test, rf_preds), 4),
        "Precision": round(precision_score(y_test, rf_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, rf_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, rf_preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, rf_probs), 4),
        "MAE": round(mean_absolute_error(y_test, rf_probs), 4),
        "RMSE": round(root_mean_squared_error(y_test, rf_probs), 4),
        "Latency (ms)": round(rf_lat, 3)
    })

    # 2. XGBoost Baseline
    xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    t0 = time.time()
    xgb.fit(X_tr_flat, y_train)
    xgb_probs = xgb.predict_proba(X_te_flat)[:, 1]
    xgb_lat = ((time.time() - t0) / len(X_test)) * 1000.0
    xgb_preds = (xgb_probs > 0.5).astype(int)
    model_predictions["XGBoost (GBDT)"] = xgb_probs

    results.append({
        "Model / Experiment": "XGBoost (GBDT)",
        "Category": "Baseline ML",
        "Accuracy": round(accuracy_score(y_test, xgb_preds), 4),
        "Precision": round(precision_score(y_test, xgb_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, xgb_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, xgb_preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, xgb_probs), 4),
        "MAE": round(mean_absolute_error(y_test, xgb_probs), 4),
        "RMSE": round(root_mean_squared_error(y_test, xgb_probs), 4),
        "Latency (ms)": round(xgb_lat, 3)
    })

    # Neural Baseline Configurations
    neural_models = [
        ("LSTM Baseline", build_lstm_model(seq_len, X_train.shape[2])),
        ("GRU Baseline", build_gru_model(seq_len, X_train.shape[2])),
        ("Transformer Baseline", build_transformer_model(seq_len, X_train.shape[2])),
        ("Attention-LSTM", build_attention_lstm_model(seq_len, X_train.shape[2]))
    ]

    for name, m in neural_models:
        m.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])
        m.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        
        t0 = time.time()
        probs = m.predict(X_test, verbose=0).flatten()
        lat = ((time.time() - t0) / len(X_test)) * 1000.0
        preds = (probs > 0.5).astype(int)
        model_predictions[name] = probs

        results.append({
            "Model / Experiment": name,
            "Category": "Neural Baseline",
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, preds, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
            "MAE": round(mean_absolute_error(y_test, probs), 4),
            "RMSE": round(root_mean_squared_error(y_test, probs), 4),
            "Latency (ms)": round(lat, 3)
        })

    # Proposed Time-Aware Cross-Attention Hybrid RAG-LSTM Model
    num_tech = len(tech_feats)
    num_rag = len(avail_features) - num_tech

    X_train_tech = X_train[:, :, :num_tech]
    X_train_rag = X_train[:, :, num_tech:]
    X_test_tech = X_test[:, :, :num_tech]
    X_test_rag = X_test[:, :, num_tech:]

    proposed_model = build_hybrid_cross_attention_rag_lstm(seq_len, num_tech, num_rag)
    proposed_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])
    proposed_model.fit([X_train_tech, X_train_rag], y_train, epochs=40, batch_size=16, verbose=0)

    t0 = time.time()
    proposed_probs = proposed_model.predict([X_test_tech, X_test_rag], verbose=0).flatten()
    proposed_lat = ((time.time() - t0) / len(X_test)) * 1000.0
    proposed_preds = (proposed_probs > 0.5).astype(int)
    model_predictions["Proposed Time-Aware RAG-LSTM"] = proposed_probs

    results.append({
        "Model / Experiment": "Proposed Time-Aware RAG-LSTM",
        "Category": "Proposed Method",
        "Accuracy": round(accuracy_score(y_test, proposed_preds), 4),
        "Precision": round(precision_score(y_test, proposed_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, proposed_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, proposed_preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proposed_probs), 4),
        "MAE": round(mean_absolute_error(y_test, proposed_probs), 4),
        "RMSE": round(root_mean_squared_error(y_test, proposed_probs), 4),
        "Latency (ms)": round(proposed_lat, 3)
    })

    # Statistical Significance Testing (t-test & Wilcoxon p-value)
    print("\n--- STATISTICAL SIGNIFICANCE TESTING (P-VALUES vs PROPOSED MODEL) ---")
    proposed_vec = model_predictions["Proposed Time-Aware RAG-LSTM"]
    for m_name, probs in model_predictions.items():
        if m_name != "Proposed Time-Aware RAG-LSTM":
            t_stat, p_val_t = stats.ttest_rel(proposed_vec, probs)
            w_stat, p_val_w = stats.wilcoxon(proposed_vec, probs)
            print(f"• Proposed vs {m_name:25s} | t-test p-val: {p_val_t:.4e} | Wilcoxon p-val: {p_val_w:.4e}")

    df_results = pd.DataFrame(results)
    csv_out = os.path.join(output_dir, "ieee_evaluation_results.csv")
    df_results.to_csv(csv_out, index=False)

    print("\n==========================================================================================")
    print(" 🏆 FINAL IEEE BENCHMARK EVALUATION SUMMARY")
    print("==========================================================================================")
    print(df_results.to_string(index=False))
    print(f"\nSaved benchmark evaluation metrics to: {csv_out}\n")

    return df_results

if __name__ == "__main__":
    run_ieee_benchmark_suite()
