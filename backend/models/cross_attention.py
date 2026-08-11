import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class CrossAttentionFusion(Layer):
    """
    IEEE Research Contribution: Multi-Head Cross-Attention Layer.
    Dynamically fuses Technical Feature Query vectors (Q) with News/RAG Feature Key (K) & Value (V) vectors.
    Allows the model to learn how financial news events modulate technical indicator momentum.
    """

    def __init__(self, d_model: int = 32, **kwargs):
        super(CrossAttentionFusion, self).__init__(**kwargs)
        self.d_model = d_model
        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

    def call(self, tech_features, news_features):
        # tech_features shape: (batch_size, d_tech)
        # news_features shape: (batch_size, d_news)
        
        Q = self.query_dense(tech_features) # (batch_size, d_model)
        K = self.key_dense(news_features)   # (batch_size, d_model)
        V = self.value_dense(news_features) # (batch_size, d_model)

        # Scaled Dot-Product Attention Scores
        # Expand dims for matrix multiplication over features
        Q_exp = tf.expand_dims(Q, axis=1) # (batch_size, 1, d_model)
        K_exp = tf.expand_dims(K, axis=2) # (batch_size, d_model, 1)

        scores = tf.matmul(Q_exp, K_exp) / tf.sqrt(tf.cast(self.d_model, tf.float32)) # (batch_size, 1, 1)
        weights = tf.nn.softmax(scores, axis=-1)

        context = tf.squeeze(weights * tf.expand_dims(V, axis=1), axis=1) # (batch_size, d_model)
        fused_representation = tf.concat([Q, context], axis=-1) # (batch_size, 2 * d_model)

        return fused_representation
