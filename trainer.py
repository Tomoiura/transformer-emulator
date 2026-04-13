"""Transformer Decoder の学習（NumPy のみ）

ミニバッチ SGD でクロスエントロピー損失を最小化する。
逆伝播は各レイヤーの backward() メソッドで実装。
"""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits, target_id):
    """最後のトークン位置の logit から損失を計算"""
    probs = softmax(logits[-1])
    loss = -np.log(probs[target_id] + 1e-12)
    # 勾配: softmax 出力 - one-hot
    grad = probs.copy()
    grad[target_id] -= 1.0
    # logits 全体の勾配（最後の位置のみ）
    d_logits = np.zeros_like(logits)
    d_logits[-1] = grad
    return float(loss), d_logits


class SimpleTrainer:
    """Transformer Decoder を学習させるトレーナー"""

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def train(self, dataset, epochs=300, verbose=True, callback=None,
              snapshot_queries=None, snapshot_count=20):
        """学習ループ

        Args:
            dataset: [(token_ids_array, target_token_id), ...]
            epochs: エポック数
            verbose: ターミナル出力
            callback: callback(epoch, loss, model) 各エポック後に呼ばれる
            snapshot_queries: [(token_ids, label), ...] スナップショット取得用クエリ
            snapshot_count: スナップショット数（等間隔で取得）

        Returns:
            dict with loss_history and snapshots
        """
        loss_history = []
        snapshots = []

        # スナップショットを取るエポック（等間隔 + 初期 + 最終）
        snap_epochs = set([1])
        if snapshot_count > 2:
            step = max(1, epochs // (snapshot_count - 1))
            for i in range(snapshot_count):
                snap_epochs.add(min(1 + i * step, epochs))
        snap_epochs.add(epochs)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            indices = np.random.permutation(len(dataset))

            for idx in indices:
                token_ids, target_id = dataset[idx]
                loss, grads = self._train_step(token_ids, target_id)
                epoch_loss += loss

            avg_loss = epoch_loss / len(dataset)
            loss_history.append(avg_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"    epoch {epoch:>4}/{epochs}  loss: {avg_loss:.4f}")

            if callback:
                callback(epoch, avg_loss, self.model)

            # スナップショット取得
            if epoch in snap_epochs and snapshot_queries:
                snap = self._take_snapshot(epoch, avg_loss, snapshot_queries)
                snapshots.append(snap)

        return {"loss_history": loss_history, "snapshots": snapshots}

    def _take_snapshot(self, epoch, loss, queries):
        """現在のモデル状態でスナップショットを取得"""
        model = self.model
        snap = {"epoch": epoch, "loss": loss, "queries": []}

        for token_ids, label in queries:
            result = model.forward(token_ids)
            out = result["output"]

            # Attention weights (最後のレイヤーのみ)
            last_layer = result["layers"][-1]
            attn_weights = last_layer["attention"]["attention_weights"].copy()

            # 予測確率（全語彙）
            top_probs = [(int(idx), float(out["probabilities"][idx]))
                         for idx in range(len(out["probabilities"]))]

            snap["queries"].append({
                "label": label,
                "attention_weights": attn_weights.tolist(),
                "top_predictions": top_probs,
            })

        # Embedding スナップショット（答え候補のトークンのみ）
        snap["embeddings"] = model.token_embedding.copy().tolist()

        return snap

    def _train_step(self, token_ids, target_id):
        """1サンプルの順伝播 → 逆伝播 → パラメータ更新"""
        model = self.model
        seq_len = len(token_ids)

        # ==================== 順伝播 ====================
        # Embedding
        tok_emb = model.token_embedding[token_ids]
        pos_emb = model.positional_encoding[:seq_len]
        x = tok_emb + pos_emb

        # 各レイヤーの入出力を記録
        layer_inputs = []
        layer_caches = []

        for layer in model.layers:
            layer_inputs.append(x.copy())
            cache = _forward_layer_with_cache(layer, x)
            layer_caches.append(cache)
            x = cache["output"]

        # 出力ヘッド
        logits = x @ model.output_proj  # (seq_len, vocab_size)

        # 損失と勾配
        loss, d_logits = cross_entropy_loss(logits, target_id)

        # ==================== 逆伝播 ====================
        # 出力射影の勾配
        d_output_proj = x.T @ d_logits  # (d_model, vocab_size)
        dx = d_logits @ model.output_proj.T  # (seq_len, d_model)

        # 各レイヤーを逆順に
        for i in reversed(range(len(model.layers))):
            layer = model.layers[i]
            cache = layer_caches[i]
            layer_input = layer_inputs[i]
            dx, layer_grads = _backward_layer(layer, dx, cache, layer_input)

            # パラメータ更新
            _update_layer(layer, layer_grads, self.lr)

        # 出力射影の更新
        model.output_proj -= self.lr * d_output_proj

        # Embedding の更新（使われたトークンのみ）
        d_tok_emb = dx  # (seq_len, d_model)
        for i, tid in enumerate(token_ids):
            model.token_embedding[tid] -= self.lr * d_tok_emb[i]

        return loss, None


def _forward_layer_with_cache(layer, x):
    """順伝播しつつ逆伝播用のキャッシュを保存"""
    # Attention
    attn = layer.attention
    seq_len = x.shape[0]

    Q = x @ attn.W_q
    K = x @ attn.W_k
    V = x @ attn.W_v

    Q_h = Q.reshape(seq_len, attn.n_heads, attn.d_k).transpose(1, 0, 2)
    K_h = K.reshape(seq_len, attn.n_heads, attn.d_k).transpose(1, 0, 2)
    V_h = V.reshape(seq_len, attn.n_heads, attn.d_k).transpose(1, 0, 2)

    scores = Q_h @ K_h.transpose(0, 2, 1) / np.sqrt(attn.d_k)
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores[:, causal_mask] = -1e9

    weights = softmax(scores)
    attn_out_h = weights @ V_h
    attn_out = attn_out_h.transpose(1, 0, 2).reshape(seq_len, attn.d_model)
    attn_proj = attn_out @ attn.W_o

    # Add & Norm 1
    residual1 = x + attn_proj
    norm1_out, norm1_cache = _layernorm_forward(layer.norm1, residual1)

    # FFN
    ffn = layer.ffn
    hidden = norm1_out @ ffn.W1 + ffn.b1
    relu_mask = hidden > 0
    activated = hidden * relu_mask
    ffn_out = activated @ ffn.W2 + ffn.b2

    # Add & Norm 2
    residual2 = norm1_out + ffn_out
    norm2_out, norm2_cache = _layernorm_forward(layer.norm2, residual2)

    return {
        "input": x,
        "Q": Q, "K": K, "V": V,
        "Q_h": Q_h, "K_h": K_h, "V_h": V_h,
        "scores": scores, "weights": weights,
        "attn_out_h": attn_out_h, "attn_out": attn_out, "attn_proj": attn_proj,
        "residual1": residual1, "norm1_out": norm1_out, "norm1_cache": norm1_cache,
        "hidden": hidden, "relu_mask": relu_mask, "activated": activated,
        "ffn_out": ffn_out,
        "residual2": residual2, "norm2_out": norm2_out, "norm2_cache": norm2_cache,
        "output": norm2_out,
    }


def _layernorm_forward(ln, x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + ln.eps)
    out = ln.gamma * x_norm + ln.beta
    return out, {"x": x, "x_norm": x_norm, "mean": mean, "var": var}


def _layernorm_backward(ln, d_out, cache):
    x, x_norm, mean, var = cache["x"], cache["x_norm"], cache["mean"], cache["var"]
    d_model = x.shape[-1]

    d_gamma = (d_out * x_norm).sum(axis=0)
    d_beta = d_out.sum(axis=0)

    dx_norm = d_out * ln.gamma
    std_inv = 1.0 / np.sqrt(var + ln.eps)

    dx = (1.0 / d_model) * std_inv * (
        d_model * dx_norm
        - dx_norm.sum(axis=-1, keepdims=True)
        - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
    )

    return dx, d_gamma, d_beta


def _backward_layer(layer, d_out, cache, layer_input):
    """1レイヤーの逆伝播"""
    grads = {}
    attn = layer.attention
    ffn = layer.ffn

    # ---- Norm2 backward ----
    d_residual2, d_gamma2, d_beta2 = _layernorm_backward(
        layer.norm2, d_out, cache["norm2_cache"])
    grads["norm2_gamma"] = d_gamma2
    grads["norm2_beta"] = d_beta2

    # ---- FFN backward ----
    d_ffn_out = d_residual2  # 残差接続の勾配
    d_norm1_from_ffn = d_residual2  # もう1つの経路

    d_activated = d_ffn_out @ ffn.W2.T
    grads["W2"] = cache["activated"].T @ d_ffn_out
    grads["b2"] = d_ffn_out.sum(axis=0)

    d_hidden = d_activated * cache["relu_mask"]
    d_norm1_out = d_hidden @ ffn.W1.T
    grads["W1"] = cache["norm1_out"].T @ d_hidden
    grads["b1"] = d_hidden.sum(axis=0)

    d_norm1_out += d_norm1_from_ffn

    # ---- Norm1 backward ----
    d_residual1, d_gamma1, d_beta1 = _layernorm_backward(
        layer.norm1, d_norm1_out, cache["norm1_cache"])
    grads["norm1_gamma"] = d_gamma1
    grads["norm1_beta"] = d_beta1

    # ---- Attention backward ----
    d_attn_proj = d_residual1  # 残差接続
    d_input_from_residual = d_residual1  # もう1つの経路

    # W_o backward
    d_attn_out = d_attn_proj @ attn.W_o.T
    grads["W_o"] = cache["attn_out"].T @ d_attn_proj

    # ヘッド形状に戻す
    seq_len = layer_input.shape[0]
    d_attn_out_h = d_attn_out.reshape(seq_len, attn.n_heads, attn.d_k).transpose(1, 0, 2)

    # d_weights, d_V
    d_weights = d_attn_out_h @ cache["V_h"].transpose(0, 2, 1)
    d_V_h = cache["weights"].transpose(0, 2, 1) @ d_attn_out_h

    # softmax backward
    d_scores = cache["weights"] * (d_weights - (d_weights * cache["weights"]).sum(axis=-1, keepdims=True))
    d_scores /= np.sqrt(attn.d_k)

    # Q, K backward
    d_Q_h = d_scores @ cache["K_h"]
    d_K_h = d_scores.transpose(0, 2, 1) @ cache["Q_h"]

    # ヘッド→元形状
    d_Q = d_Q_h.transpose(1, 0, 2).reshape(seq_len, attn.d_model)
    d_K = d_K_h.transpose(1, 0, 2).reshape(seq_len, attn.d_model)
    d_V = d_V_h.transpose(1, 0, 2).reshape(seq_len, attn.d_model)

    # W_q, W_k, W_v backward
    grads["W_q"] = layer_input.T @ d_Q
    grads["W_k"] = layer_input.T @ d_K
    grads["W_v"] = layer_input.T @ d_V

    dx = d_Q @ attn.W_q.T + d_K @ attn.W_k.T + d_V @ attn.W_v.T
    dx += d_input_from_residual

    return dx, grads


def _update_layer(layer, grads, lr):
    """レイヤーのパラメータを勾配で更新"""
    attn = layer.attention
    ffn = layer.ffn

    attn.W_q -= lr * grads["W_q"]
    attn.W_k -= lr * grads["W_k"]
    attn.W_v -= lr * grads["W_v"]
    attn.W_o -= lr * grads["W_o"]

    ffn.W1 -= lr * grads["W1"]
    ffn.b1 -= lr * grads["b1"]
    ffn.W2 -= lr * grads["W2"]
    ffn.b2 -= lr * grads["b2"]

    layer.norm1.gamma -= lr * grads["norm1_gamma"]
    layer.norm1.beta -= lr * grads["norm1_beta"]
    layer.norm2.gamma -= lr * grads["norm2_gamma"]
    layer.norm2.beta -= lr * grads["norm2_beta"]
