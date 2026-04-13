"""Transformer Decoder の各レイヤー実装（NumPy のみ）"""

import numpy as np


class LayerNorm:
    """Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> dict:
        """
        Args:
            x: (seq_len, d_model)
        Returns:
            dict with 'output', 'mean', 'std'
        """
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        return {
            "output": output,
            "mean": mean.squeeze(),
            "std": std.squeeze(),
            "before": x,
        }


class MultiHeadSelfAttention:
    """Multi-Head Masked Self-Attention"""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    def forward(self, x: np.ndarray) -> dict:
        """
        Args:
            x: (seq_len, d_model)
        Returns:
            dict with step-by-step intermediate values
        """
        seq_len = x.shape[0]

        # Q, K, V 全体
        Q = x @ self.W_q  # (seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v

        # ヘッド分割: (n_heads, seq_len, d_k)
        Q_heads = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Attention Score: (n_heads, seq_len, seq_len)
        scores = Q_heads @ K_heads.transpose(0, 2, 1) / np.sqrt(self.d_k)

        # 因果マスク: 未来のトークンを -inf に
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores_masked = scores.copy()
        scores_masked[:, causal_mask] = -np.inf

        # Softmax
        weights = self._softmax(scores_masked)

        # Weighted sum
        attn_output_heads = weights @ V_heads  # (n_heads, seq_len, d_k)

        # ヘッド結合
        attn_output = attn_output_heads.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # 出力射影
        output = attn_output @ self.W_o

        return {
            "input": x,
            "Q": Q,
            "K": K,
            "V": V,
            "Q_heads": Q_heads,
            "K_heads": K_heads,
            "V_heads": V_heads,
            "scores_raw": scores,
            "causal_mask": causal_mask,
            "scores_masked": scores_masked,
            "attention_weights": weights,
            "attn_output_heads": attn_output_heads,
            "attn_output_concat": attn_output,
            "output": output,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数値安定な softmax（最後の軸）"""
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


class FeedForward:
    """Position-wise Feed-Forward Network: Linear → ReLU → Linear"""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.d_model = d_model
        self.d_ff = d_ff
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = rng.normal(0, scale1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> dict:
        """
        Args:
            x: (seq_len, d_model)
        Returns:
            dict with intermediate values
        """
        hidden = x @ self.W1 + self.b1
        activated = np.maximum(0, hidden)  # ReLU
        output = activated @ self.W2 + self.b2
        return {
            "input": x,
            "hidden_pre_relu": hidden,
            "hidden_post_relu": activated,
            "output": output,
            "relu_sparsity": float((activated == 0).mean()),
        }


class DecoderLayer:
    """1つの Decoder Layer: MaskedAttention → Add&Norm → FFN → Add&Norm"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator):
        self.attention = MultiHeadSelfAttention(d_model, n_heads, rng)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray) -> dict:
        """
        Returns:
            dict with all substep results
        """
        # Step 1: Masked Multi-Head Self-Attention
        attn_result = self.attention.forward(x)

        # Step 2: Add & LayerNorm
        residual1 = x + attn_result["output"]
        norm1_result = self.norm1.forward(residual1)

        # Step 3: Feed-Forward Network
        ffn_result = self.ffn.forward(norm1_result["output"])

        # Step 4: Add & LayerNorm
        residual2 = norm1_result["output"] + ffn_result["output"]
        norm2_result = self.norm2.forward(residual2)

        return {
            "input": x,
            "attention": attn_result,
            "residual1": residual1,
            "norm1": norm1_result,
            "ffn": ffn_result,
            "residual2": residual2,
            "norm2": norm2_result,
            "output": norm2_result["output"],
        }
