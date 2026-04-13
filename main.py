"""Transformer Decoder-Only Emulator — CLI エントリポイント

プロファイル JSON から学習データを読み込み、学習→推論→HTML可視化を行う。
"""

import argparse
import json
import os
import platform
import subprocess
import warnings

import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

from tokenizer import JapaneseTokenizer
from model import TransformerDecoder
from trainer import SimpleTrainer
import visualizer_html as viz


# ================================================================
# デフォルトプロファイル（組み込み）
# ================================================================

DEFAULT_PROFILE = {
    "title": "Transformer Decoder Emulator",
    "description": "学習データをもとに、Transformer が文脈から答えを予測する過程を可視化します。",
    "model": {"layers": 4, "d_model": 16, "heads": 2, "d_ff": 64, "seed": 42},
    "training": {"epochs": 300, "lr": 0.005, "snapshot_count": 20, "animation_speed": 400},
    "inference": {"animation_speed": 1200, "max_tokens": 32, "temperature": 1.0},
    "training_data": [
        # 猫: ペット系 + 化ける系にも登場
        {"input": "人気のペットは", "output": "猫"},
        {"input": "かわいい動物は", "output": "猫"},
        {"input": "化ける動物は", "output": "猫"},
        {"input": "気まぐれな動物は", "output": "猫"},
        # 犬: ペット系 + 山にもいる
        {"input": "人気のペットは", "output": "犬"},
        {"input": "忠実な動物は", "output": "犬"},
        {"input": "飼いやすい動物は", "output": "犬"},
        {"input": "山で見かける動物は", "output": "犬"},
        # 狐: 山 + 昔話 + 化ける
        {"input": "山で見かける動物は", "output": "狐"},
        {"input": "昔話に出る動物は", "output": "狐"},
        {"input": "化ける動物は", "output": "狐"},
        {"input": "ずる賢い動物は", "output": "狐"},
        # 狸: 山 + 昔話 + 化ける + かわいい
        {"input": "山で見かける動物は", "output": "狸"},
        {"input": "昔話に出る動物は", "output": "狸"},
        {"input": "化ける動物は", "output": "狸"},
        {"input": "かわいい動物は", "output": "狸"},
    ],
    "queries": [
        "人気のペットは",
        "山で見かける動物は",
        "化ける動物は",
        "忠実な動物は",
        "かわいい動物は",
        "昔話に出る動物は",
    ],
}


def load_profile(path):
    """プロファイル JSON を読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(profile, extra_queries, output_path, verbose):
    title = profile.get("title", "Transformer Decoder Emulator")
    model_cfg = profile["model"]
    train_cfg = profile.get("training", {})
    training_data = profile["training_data"]
    queries = profile["queries"]
    if extra_queries:
        queries = queries + extra_queries

    # 用語辞書があればセット
    if "glossary" in profile:
        viz.set_glossary(profile["glossary"])

    infer_cfg = profile.get("inference", {})

    d_model = model_cfg.get("d_model", 16)
    n_heads = model_cfg.get("heads", 2)
    n_layers = model_cfg.get("layers", 2)
    d_ff = model_cfg.get("d_ff", d_model * 4)
    seed = model_cfg.get("seed", 42)
    epochs = train_cfg.get("epochs", 300)
    lr = train_cfg.get("lr", 0.005)
    snapshot_count = train_cfg.get("snapshot_count", 20)
    train_anim_speed = train_cfg.get("animation_speed", 400)
    infer_anim_speed = infer_cfg.get("animation_speed", 1200)
    max_tokens = infer_cfg.get("max_tokens", 32)
    temperature = infer_cfg.get("temperature", 1.0)

    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  学習データ: {len(training_data)}件")
    print(f"  推論クエリ: {len(queries)}件")
    print(f"  モデル: {n_layers}層 / d_model={d_model} / {n_heads}ヘッド / d_ff={d_ff}")
    print()

    # =============================================
    # トークン化: 全テキストを事前処理して語彙を構築
    # =============================================
    tokenizer = JapaneseTokenizer(max_tokens=max_tokens)

    # 全テキストをトークン化して語彙を構築
    for item in training_data:
        tokenizer.tokenize(item["input"])
        tokenizer.tokenize(item["output"])
    for q in queries:
        tokenizer.tokenize(q)

    print(f"  語彙サイズ: {tokenizer.vocab_size}")

    # 学習用データセット: [(token_ids, target_id), ...]
    dataset = []
    for item in training_data:
        tok = tokenizer.tokenize(item["input"])
        target_id = tokenizer.token2id[
            tokenizer.tokenize(item["output"])["tokens"][1]  # <BOS> の次
        ]
        dataset.append((tok["token_ids"], target_id))

    # =============================================
    # モデル構築 + 学習
    # =============================================
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, seed=seed,
    )
    model.update_vocab_size(tokenizer.vocab_size)

    # パラメータ数を計算
    total_params = (
        model.token_embedding.size + model.output_proj.size +
        sum(
            l.attention.W_q.size + l.attention.W_k.size +
            l.attention.W_v.size + l.attention.W_o.size +
            l.ffn.W1.size + l.ffn.b1.size +
            l.ffn.W2.size + l.ffn.b2.size +
            l.norm1.gamma.size + l.norm1.beta.size +
            l.norm2.gamma.size + l.norm2.beta.size
            for l in model.layers
        )
    )

    print(f"  パラメータ数: {total_params:,}")
    print()
    print("  学習中...")

    # スナップショット用クエリ（推論クエリをそのまま使う）
    snapshot_queries = []
    for q in queries:
        tok = tokenizer.tokenize(q)
        snapshot_queries.append((tok["token_ids"], q))

    trainer = SimpleTrainer(model, lr=lr)
    train_result = trainer.train(
        dataset, epochs=epochs, verbose=verbose,
        snapshot_queries=snapshot_queries, snapshot_count=snapshot_count,
    )
    loss_history = train_result["loss_history"]
    snapshots = train_result["snapshots"]

    print(f"  学習完了 ✓  最終 loss: {loss_history[-1]:.4f}")
    print(f"  スナップショット: {len(snapshots)} 枚")
    print()

    # =============================================
    # 推論 + ページ生成
    # =============================================
    pages = []

    # はじめに
    pages.append(("はじめに (Intro)", viz.page_intro(
        title=title,
        training_data=training_data,
        queries=queries,
        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        epochs=epochs, total_params=total_params,
    )))

    # 学習過程（アニメーション付き）
    model_info = {
        "layers": n_layers, "d_model": d_model, "heads": n_heads,
        "lr": lr, "epochs": epochs, "total_params": f"{total_params:,}",
    }
    pages.append(("学習 (Training)", viz.page_training(
        loss_history, training_data, model_info,
        snapshots=snapshots, id2token=tokenizer.id2token,
        animation_speed=train_anim_speed,
    )))

    # 各クエリで推論 → 全クエリを1タブにまとめる
    query_results = []
    all_query_data = []
    answer_order = []
    seen_ans = set()
    for item in training_data:
        if item["output"] not in seen_ans:
            answer_order.append(item["output"])
            seen_ans.add(item["output"])

    for qi, query_text in enumerate(queries):
        tok = tokenizer.tokenize(query_text)
        tokens = tok["tokens"]
        token_ids = tok["token_ids"]

        result = model.forward(token_ids, temperature=temperature)
        out = result["output"]

        top_indices = out["top_k_indices"][:5]
        top_predictions = [
            (tokenizer.id2token.get(int(idx), "?"), float(out["probabilities"][idx]))
            for idx in top_indices
        ]
        predicted_token = top_predictions[0][0]

        query_results.append({
            "query": query_text,
            "predicted": predicted_token,
            "top_predictions": top_predictions,
        })
        all_query_data.append({
            "query": query_text,
            "tokens": tokens,
            "result": result,
        })

        print(f"  「{query_text}」→ {predicted_token} ({top_predictions[0][1]:.0%})")

    # 推論タブ（1タブに全クエリ、ボタン切り替え）
    pages.append((
        "推論 (Inference)",
        viz.page_inference_all(
            all_query_data, tokenizer.id2token, answer_order,
            animation_speed=infer_anim_speed,
            n_layers=model_cfg.get("layers", 4),
        )
    ))

    # カスタマイズガイド
    pages.append(("🛠️ カスタマイズ (Customization)", viz.page_customize()))

    # 用語集
    pages.append(("📖 用語集 (Glossary)", viz.page_glossary()))

    # =============================================
    # HTML 出力
    # =============================================
    html = viz.build_html(pages, title=title)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  HTML 出力: {output_path}")
    print(f"  ページ数: {len(pages)}")

    # ブラウザで開く
    abs_path = os.path.abspath(output_path)
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", abs_path], check=True)
        elif platform.system() == "Windows":
            os.startfile(abs_path)
        else:
            subprocess.run(["xdg-open", abs_path], check=True)
        print("  ブラウザで開きました。")
    except Exception:
        print(f"  ブラウザで開けませんでした。手動で開いてください:")
        print(f"  open {abs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Decoder Emulator — 学習→推論を HTML で可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                                    # デフォルト（動物クイズ）
  python main.py --profile profiles/animals.json    # プロファイル指定
  python main.py --query "かわいい動物は"             # 追加の推論クエリ
  python main.py --epochs 500 --lr 0.01             # 学習パラメータ調整
  python main.py --verbose                          # 学習過程を詳細表示
        """,
    )
    parser.add_argument("--profile", type=str, default=None,
                        help="プロファイル JSON ファイルパス")
    parser.add_argument("--query", type=str, nargs="*", default=None,
                        help="追加の推論クエリ")
    parser.add_argument("--epochs", type=int, default=None,
                        help="学習エポック数（プロファイルの値を上書き）")
    parser.add_argument("--lr", type=float, default=None,
                        help="学習率（プロファイルの値を上書き）")
    parser.add_argument("--verbose", action="store_true",
                        help="学習過程を詳細表示")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="出力 HTML ファイルパス")

    args = parser.parse_args()

    # プロファイル読み込み
    if args.profile:
        profile = load_profile(args.profile)
    else:
        # デフォルトプロファイル or profiles/animals.json
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "profiles", "default.json"
        )
        if os.path.exists(default_path):
            profile = load_profile(default_path)
        else:
            profile = DEFAULT_PROFILE

    # コマンドライン引数でプロファイルの値を上書き
    if args.epochs:
        profile.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        profile.setdefault("training", {})["lr"] = args.lr

    # 出力パス
    output_path = args.output
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "result.html")

    run(profile, args.query, output_path, args.verbose)


if __name__ == "__main__":
    main()
