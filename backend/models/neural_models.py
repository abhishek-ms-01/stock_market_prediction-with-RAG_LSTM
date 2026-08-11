import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from models.cross_attention import CrossAttentionFusion

def build_lstm_model(seq_len: int, num_features: int) -> Model:
    """Standard LSTM Baseline."""
    inputs = Input(shape=(seq_len, num_features))
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="LSTM_Baseline")

def build_gru_model(seq_len: int, num_features: int) -> Model:
    """GRU Baseline."""
    inputs = Input(shape=(seq_len, num_features))
    x = GRU(64, return_sequences=False)(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="GRU_Baseline")

def build_transformer_model(seq_len: int, num_features: int) -> Model:
    """Temporal Transformer Encoder Baseline."""
    inputs = Input(shape=(seq_len, num_features))
    attn_out = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(inputs + attn_out)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Transformer_Baseline")

def build_attention_lstm_model(seq_len: int, num_features: int) -> Model:
    """Attention-LSTM Network."""
    inputs = Input(shape=(seq_len, num_features))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attn_out = MultiHeadAttention(num_heads=2, key_dim=16)(lstm_out, lstm_out)
    x = LayerNormalization()(lstm_out + attn_out)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Attention_LSTM")

def build_hybrid_cross_attention_rag_lstm(seq_len: int, num_tech_features: int, num_rag_features: int) -> Model:
    """
    IEEE Proposed Architecture: Time-Aware Cross-Attention Hybrid RAG-LSTM Network.
    Fuses Technical Indicator Sequences with Semantic RAG & FinBERT Features via Cross-Attention.
    """
    tech_inputs = Input(shape=(seq_len, num_tech_features), name="Tech_Inputs")
    rag_inputs = Input(shape=(seq_len, num_rag_features), name="RAG_Inputs")

    # Technical Sequence Encoder
    tech_lstm = LSTM(64, return_sequences=False)(tech_inputs)
    tech_dense = Dense(32, activation='relu')(tech_lstm)

    # RAG Sequence Encoder
    rag_lstm = LSTM(32, return_sequences=False)(rag_inputs)
    rag_dense = Dense(32, activation='relu')(rag_lstm)

    # Cross Attention Fusion
    fused_features = CrossAttentionFusion(d_model=32)(tech_dense, rag_dense)

    x = Dense(32, activation='relu')(fused_features)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model([tech_inputs, rag_inputs], outputs, name="Hybrid_CrossAttention_RAG_LSTM")
