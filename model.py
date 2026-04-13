"""Transformer Decoder-Only モデル"""

from typing import Optional

import numpy as np
from layers import DecoderLayer


class TransformerDecoder:
    """Decoder-Only Transformer アーキテクチャ"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 8,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        max_seq_len: int = 64,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or d_model * 4
        self.max_seq_len = max_seq_len

        self.rng = np.random.default_rng(seed)

        # トークン埋め込み行列: (vocab_size, d_model)
        self.token_embedding = self.rng.normal(0, 0.5, (vocab_size, d_model))

        # 位置エンコーディング（正弦波ベース）
        self.positional_encoding = self._sinusoidal_pe(max_seq_len, d_model)

        # Decoder Layers
        self.layers = [
            DecoderLayer(d_model, n_heads, self.d_ff, self.rng)
            for _ in range(n_layers)
        ]

        # 出力ヘッド: (d_model, vocab_size) — 語彙上の確率分布を出す
        scale = np.sqrt(2.0 / d_model)
        self.output_proj = self.rng.normal(0, scale, (d_model, vocab_size))

        # logit バイアス（教育用デモで特定トークンを出力させる）
        self._logit_bias = np.zeros(vocab_size)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
        """正弦波位置エンコーディング"""
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def forward(self, token_ids: np.ndarray, temperature: float = 1.0) -> dict:
        """モデル全体のフォワードパス。

        Args:
            token_ids: (seq_len,) トークンIDの配列
            temperature: 予測のランダム性 (1.0=通常, <1=確信的, >1=ランダム)

        Returns:
            dict with all layer-by-layer results
        """
        seq_len = len(token_ids)

        # --- 埋め込みステップ ---
        tok_emb = self.token_embedding[token_ids]  # (seq_len, d_model)
        pos_emb = self.positional_encoding[:seq_len]  # (seq_len, d_model)
        x = tok_emb + pos_emb

        embedding_result = {
            "token_embedding": tok_emb,
            "positional_encoding": pos_emb,
            "combined": x,
        }

        # --- レイヤーごとの処理 ---
        layer_results = []
        for i, layer in enumerate(self.layers):
            result = layer.forward(x)
            layer_results.append(result)
            x = result["output"]

        # --- 出力ヘッド ---
        logits = x @ self.output_proj  # (seq_len, vocab_size)
        # 最後のトークン位置の予測を取得
        last_logits = logits[-1] + self._logit_bias
        # Temperature でスケーリング
        scaled_logits = last_logits / max(temperature, 1e-8)
        probs = self._softmax(scaled_logits)

        output_result = {
            "logits": logits,
            "last_logits": last_logits,
            "probabilities": probs,
            "top_k_indices": np.argsort(probs)[::-1],
        }

        return {
            "embedding": embedding_result,
            "layers": layer_results,
            "output": output_result,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def update_vocab_size(self, new_vocab_size: int):
        """トークナイザの語彙拡張後に埋め込みと出力射影を更新"""
        if new_vocab_size <= self.vocab_size:
            return
        # 埋め込み拡張
        extra = self.rng.normal(
            0, 0.5, (new_vocab_size - self.vocab_size, self.d_model)
        )
        self.token_embedding = np.vstack([self.token_embedding, extra])
        # 出力射影拡張
        scale = np.sqrt(2.0 / self.d_model)
        extra_proj = self.rng.normal(
            0, scale, (self.d_model, new_vocab_size - self.vocab_size)
        )
        self.output_proj = np.hstack([self.output_proj, extra_proj])
        self._logit_bias = np.concatenate([
            self._logit_bias,
            np.zeros(new_vocab_size - self.vocab_size)
        ])
        self.vocab_size = new_vocab_size

    def bias_output(self, target_token_id: int, strength: float = 5.0):
        """特定のトークンが出力されやすくなるよう logit バイアスを設定。

        教育用デモで意味のある生成結果を見せるために使用。
        実際の学習済みモデルではこの操作は不要（学習で自然に獲得される）。
        """
        self._logit_bias[target_token_id] += strength
