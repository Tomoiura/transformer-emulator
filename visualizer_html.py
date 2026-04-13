"""Transformer Emulator — HTML 可視化出力

1つの自己完結 HTML ファイルを生成する。
CSS/JS はすべてインライン。外部依存なし。
"""

import json
import html as html_mod
import numpy as np

# ================================================================
# 用語辞書
# ================================================================
GLOSSARY = {
    # ===== アーキテクチャ =====
    "Transformer":
        "Self-Attention を中心にしたニューラルネットワーク構造。"
        "系列全体の関係を捉えやすく、現代の言語モデルの基盤となっている。",
    "Embedding":
        "トークンを連続値ベクトルに写像する表現方式。"
        "学習によって意味的・統計的な性質を反映した表現になる。",
    "Token Embedding":
        "各トークンIDを固定長ベクトルに対応づける埋め込み行列（ルックアップテーブル）。"
        "通常は学習によって更新される。",
    "Positional Encoding":
        "Transformer が語順や位置情報を扱うために加える仕組み。"
        "※ このツールでは元祖 Transformer 論文の正弦波方式を使用。"
        "現代の LLM では RoPE など別方式も多い。",
    "Combined":
        "Token Embedding + Positional Encoding の要素ごとの加算。"
        "「どのトークンか」と「何番目か」の両方の情報を持つ。",
    "d_model":
        "各トークンを表現するベクトルの次元数。"
        "モデルごとに異なり、数百〜数千程度のことが多い。ここでは教育用に小さい値を使用。",
    "d_k":
        "各 Attention Head のキー/クエリの次元数。"
        "d_model ÷ ヘッド数 で計算される。",
    "Query":
        "「何を探しているか」を表すベクトル。"
        "各トークンが他のどのトークンに注目すべきかを決める質問側。",
    "Key":
        "「自分は何を持っているか」を表すベクトル。"
        "Query と照合され、関連度スコアの計算に使われる。",
    "Value":
        "「実際に渡す情報」を表すベクトル。"
        "Attention Weight で重み付けされて集約される。",
    "Attention Score":
        "Q×Kᵀ/√d_k の計算結果。各トークンペアの関連度を数値化。"
        "値が大きいほど関連が強い（Softmax 前の生スコア）。",
    "Causal Mask":
        "因果マスク。未来のトークンへの参照を禁止する仕組み。"
        "実装では未来位置に非常に小さい値を与え、Softmax 後にほぼ 0 になるようにする。"
        "LLM の次トークン予測に不可欠。",
    "因果マスク":
        "Causal Mask の日本語表記。"
        "未来のトークンを参照できないようにするマスク処理。",
    "Attention Weight":
        "Softmax で正規化された注目度の重み。各行の合計は 1.0 になる。",
    "Self-Attention":
        "自己注意機構。入力トークン同士の関連度を計算し、"
        "文脈に応じた表現を作る Transformer の核心部分。",
    "Attention":
        "入力の各要素が他の要素にどれだけ注目すべきかを計算する仕組み。"
        "Transformer では Self-Attention として使われる。",
    "Softmax":
        "任意の実数ベクトルを確率分布(合計1.0)に変換する関数。"
        "exp(x_i) / Σexp(x_j) で計算。大きい値が強調される。",
    "Head":
        "Multi-Head Attention の1つのヘッド。"
        "各ヘッドは異なる関係性（構文・意味など）を捉える傾向がある。",
    "Multi-Head Attention":
        "Attention を複数の Head に分割して並列実行する仕組み。"
        "各 Head が異なる観点の関係を捉えることができる。",
    "残差接続":
        "入力をそのまま出力に加算するショートカット (x + f(x))。"
        "勾配消失を防ぎ、元の情報をバイパスして保持する効果がある。",
    "LayerNorm":
        "各トークンの特徴ベクトルを正規化して学習を安定させる手法。"
        "正規化後に学習可能なスケール（拡大縮小）とシフト（平行移動）も行う。",
    "FFN":
        "Feed-Forward Network。各トークンを独立に変換する位置ごとの全結合ネットワーク。"
        "典型的には2つの線形変換の間に活性化関数を挟む。",
    "ReLU":
        "活性化関数の一種。f(x) = max(0, x) — 負の値を 0 にし、正の値はそのまま通す。"
        "※ このツールでは ReLU を使用。現代の LLM では GELU や SwiGLU なども多い。",
    "Sparsity":
        "ベクトルや活性値の多くが 0（またはほぼ 0）である性質。"
        "活性化関数の後にゼロになった割合として表示される。",
    "sparsity":
        "Sparsity（疎性）。ベクトルや活性値の多くが 0 である性質。"
        "活性化関数の後にゼロになった割合として表示される。",
    "Logit":
        "Softmax に入力する前の生のスコア。"
        "語彙の各トークンに対する「未正規化の予測値」。",
    "次トークン予測":
        "Decoder モデルの本質的なタスク。"
        "与えられたトークン列から次に来るトークンを予測する。",
    "Decoder":
        "Transformer 系モデルの構成方式の一つ。"
        "因果マスクを使って過去トークンだけを参照しながら次トークンを予測する。"
        "GPT 系は Decoder-only モデルに分類される。",
    "Encoder":
        "Transformer の構成要素の一つ。入力全体を双方向に参照でき、文の理解・分類に使われる。",
    "Linear":
        "線形変換。行列の掛け算 (y = Wx + b)。"
        "入力ベクトルを別の次元空間に射影する。",

    # ===== トークン・語彙 =====
    "トークン":
        "テキストをモデルが扱う最小単位。単語とは限らず、文字列の一部・記号・サブワードであることも多い。"
        "※ このツールでは形態素解析による単語単位を使用。実際の LLM では BPE 等のサブワード分割が一般的。",
    "形態素解析":
        "日本語テキストを意味のある最小単位（形態素）に分割する処理。"
        "※ このツールのトークン化に使用。実際の LLM では BPE / SentencePiece 系のサブワード分割が主流。",
    "語彙":
        "モデルが扱える全トークンの集合。学習データに出現するトークン＋特殊トークンで構成。",
    "<BOS>":
        "Beginning of Sequence。文の開始を示す特殊トークン。全ての入力の先頭に自動付与される。",
    "<EOS>":
        "End of Sequence。文の終了を示す特殊トークン。生成停止の合図。",
    "<PAD>":
        "Padding。異なる長さの入力を同じ長さに揃えるための空白トークン。",
    "<UNK>":
        "Unknown。語彙に存在しない入力を表す特殊トークン。"
        "※ 現代のサブワード型トークナイザでは、未知語を細かく分割するため使われない場合も多い。",

    # ===== 学習 =====
    "バックプロパゲーション":
        "誤差逆伝播法。出力層の予測誤差を入力層に向かって逆向きに伝え、"
        "各重みが誤差にどれだけ影響しているかを効率的に計算する手法。"
        "勾配降下法と組み合わせて使う。",
    "勾配降下法":
        "損失関数の勾配（傾き）の方向にパラメータを少しずつ更新して損失を減らす最適化手法。",
    "損失関数":
        "予測と正解のズレを数値化する関数。学習はこの値を最小化する最適化プロセス。",
    "クロスエントロピー":
        "分類問題で使われる損失関数。正解の確率が高いほど損失が小さくなる。",
    "重み":
        "モデルのパラメータ。学習前はランダム、学習により意味のある値に更新される。",
    "パラメータ":
        "モデルの学習可能な変数の総称。重みやバイアスを含む。"
        "モデル規模はパラメータ数で表されることが多い。",
    "エポック":
        "学習データ全体を1回通して学習すること。"
        "※ このツールのような小規模データでは数百エポック回すが、"
        "大規模 LLM ではトークン総数ベースで学習量を管理することも多い。",
    "学習率":
        "パラメータ更新の大きさを決める係数。"
        "大きすぎると発散、小さすぎると収束が遅い。",
    "過学習":
        "学習データを暗記してしまい、未知データへの予測性能が低下する現象。",

    # ===== 推論・生成 =====
    "推論":
        "学習済みモデルに新しい入力を与えて予測を行うこと。学習（Training）の対義語。",
    "自己回帰生成":
        "次トークンを1つ予測→入力に追加→また予測、を繰り返す生成手法。"
        "生成AIのテキスト生成の基本メカニズム。",
    "Temperature":
        "予測のランダム性を制御するパラメータ。"
        "低い値(0.1)=最も確率の高い答えをほぼ確定的に選択。"
        "高い値(2.0)=確率の低い答えも選ばれやすくなる。"
        "1.0が標準。実際のAIチャットでも「創造性」の調整に使われる。",
    "Greedy":
        "常に最大確率のトークンを選択するデコーディング手法。確定的で毎回同じ結果。",
    "Sampling":
        "確率分布に従ってランダムにトークンを選択するデコーディング手法。多様な出力が得られる。",
    "確率分布":
        "各トークンが選ばれる確率の一覧。全て合計すると1.0になる。Softmax で生成される。",

    # ===== 数学・一般 =====
    "ベクトル":
        "複数の数値を並べた配列。"
        "Transformer では各トークンを d_model 次元のベクトルで表現する。",
    "行列":
        "数値を縦横に並べた2次元配列。Transformer の計算は本質的に行列の掛け算の連鎖。",
    "次元":
        "ベクトルや行列の大きさを表す数。d_model=16 なら各トークンは16個の数値で表現。",
    "正弦波":
        "sin/cos 関数による周期的な波形。"
        "位置が異なると異なるパターンになるため、位置の識別に使える。",
    "正規化":
        "データの値を一定の範囲やルールに揃える処理。LayerNorm では正規化後にスケール・シフトも行う。",
    "活性化関数":
        "ニューラルネットワークに非線形性を導入する関数。"
        "ReLU, GELU, SwiGLU 等がある。これがないと多層にしても線形変換の繰り返しにしかならない。",
    "ニューラルネットワーク":
        "人間の神経回路を模した計算モデル。入力→隠れ層→出力の多層構造。Transformer はその一種。",
    "LLM":
        "Large Language Model（大規模言語モデル）。大量のテキストで学習した Transformer ベースのモデルの総称。",

    # ===== 日本語表記（別名） =====
    "レイヤー":
        "Transformer の処理層。Self-Attention と FFN を基本単位として積み重ね、"
        "表現を段階的に変換する。",
    "ヘッド":
        "Multi-Head Attention を構成する各並列チャネル。詳細は Head を参照。",
}


def set_glossary(glossary_dict):
    """外部から用語辞書を差し替える"""
    global GLOSSARY
    GLOSSARY = glossary_dict


def _kw(text):
    """テキスト内の GLOSSARY キーワードを <span class="kw"> でラップ。
    プレースホルダー方式で二重置換を防止。"""
    sorted_keys = sorted(GLOSSARY.keys(), key=len, reverse=True)
    result = text
    placeholders = {}  # placeholder_id -> html

    for kw in sorted_keys:
        if kw in result:
            pid = f"\x00KW{len(placeholders):03d}\x00"
            safe_tip = html_mod.escape(GLOSSARY[kw])
            safe_kw = html_mod.escape(kw)
            placeholders[pid] = f'<span class="kw" data-tip="{safe_tip}">{safe_kw}</span>'
            result = result.replace(kw, pid, 1)

    # プレースホルダーを実際の HTML に置換
    for pid, html_str in placeholders.items():
        result = result.replace(pid, html_str)

    return result


def _heatmap_html(matrix, row_labels, col_labels=None, cmap="rdbu", fmt=".2f",
                   show_values=True):
    """numpy 行列を CSS Grid ヒートマップ HTML に変換"""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    rows, cols = matrix.shape
    if col_labels is None:
        col_labels = [str(i) for i in range(cols)]

    vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    if vmin == vmax:
        vmax = vmin + 1

    lines = ['<div class="heatmap-container">']
    lines.append(f'<table class="heatmap" style="--cols:{cols};">')

    # ヘッダ
    lines.append('<tr><th></th>')
    for cl in col_labels:
        lines.append(f'<th class="hm-col-label">{html_mod.escape(str(cl))}</th>')
    lines.append('</tr>')

    for i in range(rows):
        lines.append('<tr>')
        lines.append(f'<td class="hm-row-label">{html_mod.escape(str(row_labels[i]))}</td>')
        for j in range(cols):
            v = matrix[i, j]
            if np.isnan(v):
                lines.append('<td class="hm-cell hm-nan">-∞</td>')
                continue
            # 色計算
            if cmap == "oranges":
                # 0〜1 のオレンジスケール
                t = max(0, min(1, float(v)))
                r = int(255)
                g = int(255 - t * 180)
                b = int(255 - t * 220)
                text_color = "#fff" if t > 0.5 else "#333"
            else:
                # RdBu 発散スケール
                t = (float(v) - vmin) / (vmax - vmin)  # 0〜1
                if t < 0.5:
                    # 青 → 白
                    s = t * 2
                    r = int(59 + s * 196)
                    g = int(76 + s * 179)
                    b = int(192 + s * 63)
                else:
                    # 白 → 赤
                    s = (t - 0.5) * 2
                    r = int(255 - s * 37)
                    g = int(255 - s * 186)
                    b = int(255 - s * 186)
                text_color = "#fff" if (t < 0.25 or t > 0.75) else "#333"

            val_str = f"{v:{fmt}}" if show_values and rows * cols <= 256 else ""
            lines.append(
                f'<td class="hm-cell" style="background:rgb({r},{g},{b});color:{text_color}">'
                f'{val_str}</td>'
            )
        lines.append('</tr>')

    lines.append('</table></div>')
    return '\n'.join(lines)


def _bar_chart_html(labels, values, fmt=".4f"):
    """水平棒グラフ HTML"""
    max_val = max(values) if values else 1
    lines = ['<div class="bar-chart">']
    for label, val in zip(labels, values):
        pct = val / max_val * 100 if max_val > 0 else 0
        lines.append(f'''
        <div class="bar-row">
            <span class="bar-label">{html_mod.escape(str(label))}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%"></div>
            </div>
            <span class="bar-value">{val:{fmt}}</span>
        </div>''')
    lines.append('</div>')
    return '\n'.join(lines)


def _token_chips(tokens, generated_start=None):
    """トークン列をチップ表示"""
    parts = []
    for i, tok in enumerate(tokens):
        cls = "token-gen" if generated_start and i >= generated_start else "token-input"
        parts.append(f'<span class="{cls}">{html_mod.escape(tok)}</span>')
    return ' <span class="token-arrow">→</span> '.join(parts)


# ================================================================
# ページ生成関数
# ================================================================

# 用語集のカテゴリ定義（表示順）
_GLOSSARY_CATEGORIES = [
    ("アーキテクチャ", [
        "Transformer", "Decoder", "Encoder", "レイヤー",
        "Embedding", "Token Embedding", "Positional Encoding", "Combined",
        "Self-Attention", "Attention", "Multi-Head Attention", "Head", "ヘッド",
        "Query", "Key", "Value", "Attention Score", "Causal Mask", "因果マスク", "Attention Weight",
        "残差接続", "LayerNorm", "FFN", "ReLU", "Logit", "Linear",
    ]),
    ("モデル設定", [
        "d_model", "d_k", "d_ff", "Sparsity",
    ]),
    ("トークン・語彙", [
        "トークン", "形態素解析", "語彙",
        "<BOS>", "<EOS>", "<PAD>", "<UNK>",
    ]),
    ("学習", [
        "次トークン予測", "バックプロパゲーション", "勾配降下法",
        "損失関数", "クロスエントロピー",
        "重み", "パラメータ", "エポック", "学習率", "過学習",
        "バッチ", "収束",
    ]),
    ("推論・生成", [
        "推論", "自己回帰生成", "Temperature", "Softmax",
        "Greedy", "Sampling", "Top-k", "Top-p", "確率分布",
    ]),
    ("数学・一般", [
        "ベクトル", "行列", "次元", "正弦波",
        "正規化", "活性化関数", "ニューラルネットワーク", "LLM",
    ]),
]


def _glossary_section():
    """用語集セクションのHTMLを生成。カテゴリ別に用語をマウスオーバー表示。"""
    html = '<div class="customize-section">\n'
    html += '    <h4>📖 用語集 / Glossary</h4>\n'
    html += '    <p style="color:#666;font-size:0.9em;margin-bottom:12px;">'
    html += '各用語にマウスを乗せると解説が表示されます（クリックで固定）。'
    html += '本ページ内・学習ページ・推論ページの全テキストにも同じ機能が有効です。</p>\n'

    for cat_name, terms in _GLOSSARY_CATEGORIES:
        # カテゴリに存在する用語だけフィルタ
        valid_terms = [t for t in terms if t in GLOSSARY]
        if not valid_terms:
            continue
        html += f'    <div style="margin-bottom:12px;">\n'
        html += (f'      <span class="glossary-cat">【 {html_mod.escape(cat_name)} 】</span><br>\n')
        html += '      <div style="display:flex;flex-wrap:wrap;gap:4px 8px;margin-top:6px;">\n'
        for term in valid_terms:
            safe_tip = html_mod.escape(GLOSSARY[term])
            safe_term = html_mod.escape(term)
            html += f'        <span class="kw" data-tip="{safe_tip}">{safe_term}</span>\n'
        html += '      </div>\n'
        html += '    </div>\n'

    html += '</div>'
    return html


def page_intro(title, training_data, queries, n_layers, d_model, n_heads, d_ff,
               epochs=300, total_params=0):
    # 学習データの答えを集計
    answers = sorted(set(item["output"] for item in training_data))
    answers_str = "・".join(answers)

    # 学習データの例を3件表示
    sample_rows = ""
    for item in training_data[:4]:
        sample_rows += f'<tr><td>「{html_mod.escape(item["input"])}」</td><td><strong>{html_mod.escape(item["output"])}</strong></td></tr>'
    if len(training_data) > 4:
        sample_rows += f'<tr><td colspan="2" style="color:#888">…他 {len(training_data)-4} 件</td></tr>'

    # 推論クエリ一覧
    query_chips = " / ".join([f'「{html_mod.escape(q)}」' for q in queries])

    return f'''
    <div class="step-header">{html_mod.escape(title)}</div>
    <p class="subtitle">大規模言語モデル (LLM) の内部動作を<br>
    「学習」から「推論」まで、ステップごとに可視化するツール</p>

    <!-- ===== このプログラムは何？ ===== -->
    <h3>🔍 このプログラムは何？</h3>
    <div class="desc">
        <p>AI チャットに質問すると、まるで「考えて」答えているように見えます。</p>
        <p>しかし実際には、内部で<strong>数値の行列計算</strong>を何層も繰り返しているだけです。</p>
        <p>このプログラムでは、小さな {_kw("Transformer")} モデルを <strong>実際に学習</strong>させ、
        その後の<strong>推論（予測）の全過程</strong>を1ステップずつ可視化します。</p>
        <p>「AIが学ぶとはどういうことか」「質問に答えるとき内部で何が起きているか」を体感できます。</p>
    </div>

    <div class="desc" style="background:#E8F5E9; border-left-color:#43A047;">
        <h4 style="color:#2E7D32;">💬 なぜ Decoder だけ？</h4>
        <p>{_kw("Transformer")} には Encoder（入力を理解する）と {_kw("Decoder")}（テキストを生成する）の2種類がありますが、
        現在の主要な生成AI の多くは <strong>Decoder のみ</strong>の構造を採用しています。</p>
        <p>Decoder の特徴は {_kw("Causal Mask")}（因果マスク）です。
        「未来の単語を見ずに次の単語を予測する」という制約があるからこそ、
        1単語ずつテキストを生成できます。
        このツールでは、最も実用的かつ理解すべきこの Decoder 構造に焦点を当てています。</p>
    </div>

    <!-- ===== 今回のシナリオ ===== -->
    <h3>📝 今回のシナリオ</h3>
    <div class="info-box">
        <p><strong>テーマ:</strong> 動物に関する質問と回答</p>
        <table class="data-table" style="margin:0.8em 0;">
            <tr><th>学習データ（入力）</th><th>正解</th></tr>
            {sample_rows}
        </table>
        <p><strong>登場する答え:</strong> {answers_str}</p>
        <p><strong>推論で試す質問:</strong> {query_chips}</p>
        <p style="margin-top:0.5em; color:#888; font-size:0.9em;">
            モデル: {n_layers}{_kw("レイヤー")} / {_kw("d_model")}={d_model} / {n_heads}{_kw("ヘッド")} / {_kw("パラメータ")}数 {total_params:,} 個<br>
            （教育用に極小サイズ。実際の {_kw("LLM")} は数千万〜数千億個の{_kw("パラメータ")}を持つ）
        </p>
    </div>

    <!-- ===== 全体の流れ ===== -->
    <h3>🎯 このツールで体験できること</h3>
    <div class="desc">
        <p>AIモデルが動くまでには、大きく<strong>2つのフェーズ</strong>があります。</p>
    </div>

    <div class="phase-box phase-train">
        <h4>フェーズ1: 学習（Training）</h4>
        <p>ランダムな重みから始めて、{len(training_data)}件の学習データを{epochs}回繰り返し学習。</p>
        <ol>
            <li>モデルに質問を入力し、答えを予測させる（最初はデタラメ）</li>
            <li>予測と正解のズレ（{_kw("損失関数")}）を計算する</li>
            <li>ズレが小さくなるよう重みを少しずつ調整する（{_kw("勾配降下法")}）</li>
            <li>これを何百回も繰り返す → 予測精度が上がる</li>
        </ol>
        <p>→ {_kw("損失関数")}の値が下がっていく様子を確認できます。</p>
        <button class="goto-btn goto-train" onclick="showTab(1)">▶ 学習過程を見る</button>
    </div>

    <div class="phase-box phase-infer">
        <h4>フェーズ2: 推論（Inference）</h4>
        <p>学習済みモデルに質問を入力し、答えを予測させる。この内部処理を全ステップ可視化。</p>
        <ol>
            <li>質問文を単語（{_kw("トークン")}）に分割し、数値（{_kw("ベクトル")}）に変換する</li>
            <li>各単語が「他のどの単語に注目するか」を計算する（{_kw("Self-Attention")}）</li>
            <li>各単語の表現をさらに変換する（{_kw("FFN")}）</li>
            <li>2〜3 を{n_layers}回繰り返す（= {n_layers}層のレイヤー）</li>
            <li>最後に、語彙の中から「次に来る単語」の確率を出力する（{_kw("次トークン予測")}）</li>
        </ol>
        <p>→ 各質問ごとに、Embedding → Attention → FFN → 予測 の全過程を確認できます。</p>
        <button class="goto-btn goto-infer" onclick="showTab(2)">▶ 推論の過程を見る</button>
    </div>

    <!-- ===== さっそく見てみよう ===== -->
    <div class="goto-nav">
        <button class="goto-btn-lg goto-train" onclick="showTab(1)">📊 学習過程を見る<span class="goto-sub">Loss の低下 / 予測の変化 / Attention の変化</span></button>
        <button class="goto-btn-lg goto-infer" onclick="showTab(2)">🔍 推論の過程を見る<span class="goto-sub">Embedding → Attention → FFN → 予測</span></button>
    </div>
    <div class="goto-nav" style="margin-top:0.5em;">
        <button class="goto-btn-lg goto-customize" onclick="showTab(3)">🛠️ カスタマイズ<span class="goto-sub">自分のデータで試す / パラメータ調整</span></button>
        <button class="goto-btn-lg goto-glossary" onclick="showTab(4)">📖 用語集<span class="goto-sub">専門用語の解説一覧</span></button>
    </div>

    <div class="note-box">
        <strong>📌 ポイント:</strong> このエミュレータは実際に{len(training_data)}件のデータで学習を行っています。
        答えを事前に仕込んでいるのではなく、モデルが学習データから「文脈と答えの関係」を学び取った結果です。
        ただし、実際の LLM は数兆語のデータで学習しており、このミニチュアモデルとは規模が全く異なります。
    </div>

    '''


def page_gen_header(step_num, total_steps, current_tokens, generated_tokens, next_target):
    gen_start = len(current_tokens) - len(generated_tokens)
    return f'''
    <div class="step-header">自己回帰生成 ── ステップ {step_num}/{total_steps}</div>
    <div class="desc">
        <p>現在のトークン列:</p>
        <div class="token-line">{_token_chips(current_tokens, gen_start)}</div>
        <p class="gen-target">→ 次に予測するトークン: 「<strong>{html_mod.escape(next_target)}</strong>」</p>
        <p>{_kw("以下のステップで、この入力列が Transformer の各レイヤーを通過し、最後のトークン位置から 次トークン予測 が行われる過程を可視化します。")}</p>
    </div>
    '''


def page_embedding(emb, tokens):
    return f'''
    <div class="step-header">STEP 1: Embedding + Positional Encoding</div>
    <div class="desc">
        <p>{_kw("各トークンを d_model 次元のベクトルに変換する (Token Embedding)。")}</p>
        <p>{_kw("位置情報を正弦波で加算し、語順を表現する (Positional Encoding)。")}</p>
        <p>{_kw("→ 右端の Combined が次のレイヤー (Self-Attention) への入力になる。")}</p>
    </div>
    <div class="heatmap-row">
        <div><h4>Token Embedding</h4>{_heatmap_html(emb["token_embedding"], tokens)}</div>
        <div><h4>Positional Encoding</h4>{_heatmap_html(emb["positional_encoding"], tokens)}</div>
        <div><h4>Combined</h4>{_heatmap_html(emb["combined"], tokens)}</div>
    </div>
    '''


def page_attention(attn, tokens, layer_idx):
    n_heads = attn["attention_weights"].shape[0]
    n = len(tokens)

    score_html = ""
    for h in range(n_heads):
        masked = attn["scores_masked"][h].copy()
        masked = np.where(np.isinf(masked), np.nan, masked)
        score_html += f'<div><h4>Head {h+1}: Score (Causal Mask 適用後)</h4>'
        score_html += _heatmap_html(masked, tokens, tokens) + '</div>'

    weight_html = ""
    for h in range(n_heads):
        weight_html += f'<div><h4>Head {h+1}: Attention Weight (Softmax)</h4>'
        weight_html += _heatmap_html(attn["attention_weights"][h], tokens, tokens,
                                      cmap="oranges") + '</div>'

    return f'''
    <div class="step-header">Layer {layer_idx+1}: Masked Self-Attention</div>
    <div class="desc">
        <p>{_kw("1行目: Query × Key ᵀ / √d_k で各トークン間の Attention Score を計算。")}</p>
        <p>{_kw("2行目: 因果マスク (Causal Mask) 適用。未来のトークン (右上の三角) を -∞ にして参照不可に。")}</p>
        <p>{_kw("→ Decoder では「猫」は「は」以降を見れない。これが生成AI の核心。")}</p>
        <p>{_kw("3行目: Softmax で正規化 → Attention Weight。色が濃い＝強く参照。各行の合計=1.0。")}</p>
    </div>
    <div class="heatmap-row">{score_html}</div>
    <div class="heatmap-row">{weight_html}</div>
    '''


def page_residual_norm(layer_result, tokens, layer_idx, stage):
    descs = {
        "Attention": [
            "残差接続: 元の入力に Self-Attention の出力を加算 (input + attn_output)。",
            "→ 変換で失われる情報を補い、勾配の消失を防ぐ。",
            "LayerNorm: 各トークンのベクトルを平均0・分散1に正規化し、学習を安定させる。",
        ],
        "FFN": [
            "残差接続: Self-Attention 後の値に FFN の出力を加算。",
            "→ Self-Attention と FFN の両方の情報を保持する。",
            "LayerNorm: 再度正規化して次のレイヤーへ渡す。",
        ],
    }
    desc_lines = descs.get(stage, [])
    desc_html = ''.join(f'<p>{_kw(d)}</p>' for d in desc_lines)

    if stage == "Attention":
        before = layer_result["input"]
        after_add = layer_result["residual1"]
        after_norm = layer_result["norm1"]["output"]
    else:
        before = layer_result["norm1"]["output"]
        after_add = layer_result["residual2"]
        after_norm = layer_result["norm2"]["output"]

    return f'''
    <div class="step-header">Layer {layer_idx+1}: Add &amp; LayerNorm ({stage})</div>
    <div class="desc">{desc_html}</div>
    <div class="heatmap-row">
        <div><h4>Input (before {stage})</h4>{_heatmap_html(before, tokens)}</div>
        <div><h4>After Residual Add</h4>{_heatmap_html(after_add, tokens)}</div>
        <div><h4>After LayerNorm</h4>{_heatmap_html(after_norm, tokens)}</div>
    </div>
    '''


def page_ffn(ffn, tokens, layer_idx):
    d_model = ffn["input"].shape[1]
    d_ff = ffn["hidden_post_relu"].shape[1]
    return f'''
    <div class="step-header">Layer {layer_idx+1}: Feed-Forward Network (FFN)</div>
    <div class="desc">
        <p>{_kw("各トークンを独立に非線形変換する (トークン間の情報交換なし)。")}</p>
        <p>{_kw(f"Linear({d_model}→{d_ff}) → ReLU (負の値を0に) → Linear({d_ff}→{d_model})。")}</p>
        <p>{_kw(f"中央: ReLU 後の隠れ層。青い部分(=0)が sparsity。値={ffn['relu_sparsity']:.0%}。")}</p>
    </div>
    <div class="heatmap-row">
        <div><h4>FFN Input</h4>{_heatmap_html(ffn["input"], tokens)}</div>
        <div><h4>Hidden (after ReLU) sparsity={ffn["relu_sparsity"]:.0%}</h4>{_heatmap_html(ffn["hidden_post_relu"], tokens)}</div>
        <div><h4>FFN Output</h4>{_heatmap_html(ffn["output"], tokens)}</div>
    </div>
    '''


def page_prediction(out, id2token):
    top_k = 10
    indices = out["top_k_indices"][:top_k]
    probs = [float(out["probabilities"][i]) for i in indices]
    labels = [id2token.get(int(i), f"id={i}") for i in indices]
    return f'''
    <div class="step-header">Next Token Prediction</div>
    <div class="desc">
        <p>{_kw("最終レイヤー出力の最後のトークン位置ベクトルを語彙サイズに射影し Logit を得る。")}</p>
        <p>{_kw("Softmax で確率分布に変換し、最も確率の高いトークンが 次トークン予測 の結果。")}</p>
        <p class="note">※ 未学習のランダム重みに教育用バイアスを加えたデモ。学習済みモデルでは自然に文脈に沿った予測になる。</p>
    </div>
    {_bar_chart_html(labels, probs)}
    '''


def page_layer_comparison(layer_results, tokens):
    panels = ""
    for i, lr in enumerate(layer_results):
        panels += f'<div><h4>Layer {i+1} Output</h4>{_heatmap_html(lr["output"], tokens)}</div>'
    return f'''
    <div class="step-header">Layer Output Comparison</div>
    <div class="desc">
        <p>{_kw("各レイヤーを通過するごとに表現がどう変化するかを比較。")}</p>
        <p>{_kw("レイヤーが深くなるほど、Self-Attention で文脈情報が統合された抽象的な表現になる。")}</p>
    </div>
    <div class="heatmap-row">{panels}</div>
    '''


def page_gen_summary(input_text, generated_tokens, all_tokens):
    gen_start = len(all_tokens) - len(generated_tokens)
    return f'''
    <div class="step-header">生成完了</div>
    <div class="summary-box">
        <div class="summary-row"><span class="summary-label">入力:</span> {html_mod.escape(input_text)}</div>
        <div class="summary-row"><span class="summary-label">生成:</span> <span class="gen-text">{"".join(html_mod.escape(t) for t in generated_tokens)}</span></div>
        <div class="summary-row"><span class="summary-label">全文:</span> <strong>{html_mod.escape(input_text)}{"".join(html_mod.escape(t) for t in generated_tokens)}</strong></div>
    </div>
    <h4>トークン列の変遷</h4>
    <div class="token-line">{_token_chips(all_tokens, gen_start)}</div>
    <div class="desc" style="margin-top:2em;">
        <p>{_kw("これが Decoder モデルの 自己回帰生成 の仕組み。")}</p>
        <ol>
            <li>{_kw("入力列を Transformer に通し、最後の位置で 次トークン予測 を行う")}</li>
            <li>予測された{_kw("トークン")}を入力列の末尾に追加する</li>
            <li>{_kw("拡張された入力列で再度 Transformer を通す")}</li>
        </ol>
        <p>→ これを繰り返すことで、文章が1{_kw("トークン")}ずつ生成される。</p>
    </div>
    '''


# ================================================================
# 学習過程の可視化
# ================================================================

def page_training(loss_history, training_data_summary, model_info,
                  snapshots=None, id2token=None, animation_speed=400):
    """学習過程の可視化ページ（アニメーション対応）"""
    import json as _json

    # 学習データ一覧（4件まで表示、残りは折りたたみ）
    preview_count = 4
    data_rows_visible = ""
    data_rows_hidden = ""
    for i, item in enumerate(training_data_summary):
        row = f'<tr><td>{html_mod.escape(item["input"])}</td><td><strong>{html_mod.escape(item["output"])}</strong></td></tr>\n'
        if i < preview_count:
            data_rows_visible += row
        else:
            data_rows_hidden += row
    has_hidden = len(training_data_summary) > preview_count

    # 答え候補を固定順序で収集（学習データの output 登場順）
    answer_order = []
    seen = set()
    for item in training_data_summary:
        if item["output"] not in seen:
            answer_order.append(item["output"])
            seen.add(item["output"])
    answer_order_json = _json.dumps(answer_order, ensure_ascii=False)

    # スナップショットデータを JSON としてページに埋め込む
    snap_json = "[]"
    query_labels = []
    convergence_idx = -1  # 収束ポイント（スナップショットのインデックス）
    if snapshots and id2token:
        # 予測を全語彙分の確率辞書に変換（固定順序表示用）
        for snap in snapshots:
            for q in snap["queries"]:
                prob_dict = {}
                for tid, prob in q["top_predictions"]:
                    token_str = id2token.get(tid, f"id={tid}")
                    prob_dict[token_str] = prob
                q["prob_dict"] = prob_dict
        snap_json = _json.dumps(snapshots, ensure_ascii=False)
        if snapshots[0]["queries"]:
            query_labels = [q["label"] for q in snapshots[0]["queries"]]

        # 収束ポイントの検出:
        # 直近の変化率が初期降下と比べて十分小さくなった地点
        losses = [s["loss"] for s in snapshots]
        if len(losses) >= 4:
            total_drop = losses[0] - losses[-1]
            if total_drop > 0:
                for i in range(2, len(losses)):
                    recent_drop = losses[i - 1] - losses[i]
                    relative_change = recent_drop / total_drop
                    # 1ステップあたりの変化が全体降下量の2%未満で収束とみなす
                    if relative_change < 0.02 and losses[i] < losses[0] * 0.6:
                        convergence_idx = i
                        break

    query_tabs = ""
    for i, label in enumerate(query_labels):
        active = "active" if i == 0 else ""
        query_tabs += f'<button class="qtab {active}" onclick="selectQuery({i})">{html_mod.escape(label)}</button> '

    n_data = len(training_data_summary)
    n_epochs = model_info['epochs']
    total_steps = n_data * n_epochs

    return f'''
    <div class="step-header">学習過程（Training）</div>

    <div class="desc">
        <h4>「学習」とは何か？</h4>
        <p>{_kw("Transformer")} のパラメータ（約 {model_info.get("total_params", "?")} 個の数値＝「重み」）は、
        最初は<strong>ランダムな値</strong>で初期化されています。この状態では何も予測できません。</p>

        <p>学習とは、「問題を解く → 答え合わせ → 間違いを修正する」を何千回も繰り返して、
        重みを少しずつ調整していくプロセスです。人間が問題集を繰り返し解いて覚えるのに似ています。</p>
    </div>

    <div class="desc">
        <h4>今回の学習の規模</h4>
        <table class="data-table" style="margin:0.5em 0;">
            <tr><td>学習データ</td><td><strong>{n_data}件</strong>（下の表を参照）</td></tr>
            <tr><td>{_kw("エポック")}数</td><td><strong>{n_epochs}回</strong>（＝ {n_data}件を{n_epochs}回繰り返す）</td></tr>
            <tr><td>学習ステップ合計</td><td><strong>{total_steps:,}回</strong>（＝ {n_data} × {n_epochs}）</td></tr>
            <tr><td>{_kw("学習率")}</td><td>{model_info['lr']}（1回の修正の大きさ）</td></tr>
        </table>
        <p style="color:#666; font-size:0.9em;">※ 実際の LLM は数兆語のデータを数回通す。
        このミニモデルはデータが少ないため、多くの繰り返しが必要。</p>
    </div>

    <div class="desc">
        <h4>各ステップで起きていること</h4>
        <ol>
            <li>モデルに「人気のペットは」と入力し、次の単語を予測させる（最初はデタラメ）</li>
            <li>予測と正解のズレを計算する → これが{_kw("損失関数")}（Loss）の値</li>
            <li>出力層から入力層へ逆向きに「どの重みがズレの原因か」を計算する（{_kw("バックプロパゲーション")}）</li>
            <li>原因の大きい重みほど大きく修正する（{_kw("勾配降下法")}、修正の大きさ ＝ {_kw("学習率")}）</li>
            <li>これを {n_data}件 × {n_epochs}回 ＝ {total_steps:,}回 繰り返す</li>
        </ol>
        <p>下のアニメーションでは、{n_epochs}エポックの中から<strong>20箇所でスナップショット</strong>を撮影し、
        学習が進むにつれてモデルの予測がどう変化するかをパラパラ漫画のように表示します。</p>
    </div>

    <h3>学習データ（{n_data}件）</h3>
    <table class="data-table">
        <tr><th>入力</th><th>正解</th></tr>
        {data_rows_visible}
        <tbody id="train-data-hidden" style="display:none">{data_rows_hidden}</tbody>
    </table>
    {"" if not has_hidden else '<a href="#" class="toggle-link" id="train-data-toggle" onclick="toggleTrainData(event)">すべて表示 (' + str(len(training_data_summary)) + '件)</a>'}

    <!-- ===== 操作方法 ===== -->
    <div class="desc" style="background:#FFF3E0; border-left:4px solid #E65100; padding:0.8em 1em; margin:1em 0;">
        <strong>🖱️ 操作方法:</strong>
        <strong>▶ 再生</strong>でアニメーション開始、<strong>次へ ▶|</strong> で1コマ送り、スライダーで任意の位置へジャンプ。
        青い用語にマウスを重ねると解説が表示されます。
    </div>

    <!-- ===== アニメーション 2×2 グリッド ===== -->
    <h3>学習アニメーション</h3>

    <div class="anim-grid">
        <!-- 左上: 再生コントロール + Loss -->
        <div class="anim-panel">
            <div class="anim-controls">
                <button id="anim-play" class="anim-btn" onclick="togglePlay()">▶ 再生</button>
                <button class="anim-btn anim-btn-step" onclick="stepForward()">次へ ▶|</button>
                <input type="range" id="anim-slider" min="0" max="0" value="0"
                       oninput="seekSnapshot(this.value)" class="anim-slider">
                <span id="anim-epoch" class="anim-epoch">epoch 1</span>
            </div>
            <h4>{_kw("損失関数")} (Loss)</h4>
            <div class="anim-loss-container">
                <div class="anim-loss-bar-bg">
                    <div id="anim-loss-bar" class="anim-loss-fill"></div>
                </div>
                <span id="anim-loss-val" class="anim-loss-val">-</span>
            </div>
            <div id="anim-loss-chart" class="anim-loss-chart"></div>
        </div>

        <!-- 右上: 次トークン予測 -->
        <div class="anim-panel">
            <h4>{_kw("次トークン予測")} の変化</h4>
            <div class="anim-query-tabs">{query_tabs}</div>
            <div id="anim-predictions" class="anim-predictions"></div>
        </div>

        <!-- 左下: モデル情報 -->
        <div class="anim-panel">
            <h4>モデル情報</h4>
            <div style="font-size:0.9em; line-height:1.8;">
                <div>{_kw("レイヤー")}数: {model_info['layers']} / {_kw("d_model")}: {model_info['d_model']} / {_kw("ヘッド")}数: {model_info['heads']}</div>
                <div>{_kw("学習率")}: {model_info['lr']} / {_kw("エポック")}数: {model_info['epochs']}</div>
                <div>パラメータ総数: 約 {model_info.get('total_params', '?')} 個</div>
            </div>
        </div>

        <!-- 右下: Attention Weight -->
        <div class="anim-panel">
            <h4>{_kw("Attention Weight")} の変化</h4>
            <div id="anim-attention" class="anim-attention"></div>
        </div>
    </div>

    <div class="observe-box">
        <h4>🔎 アニメーションの着目ポイント</h4>
        <dl class="observe-list">
            <dt>📉 Loss（損失）の変化</dt>
            <dd>最初は高い値（＝デタラメな予測）から始まり、急激に下がる。
            その後は緩やかに低下する。この「急→緩」のカーブが典型的な学習曲線。
            Loss が下がりきらない場合は、データに矛盾がある（同じ質問に複数の答え）ことを意味する。</dd>

            <dt>📍 収束ライン（オレンジ破線）</dt>
            <dd>Loss グラフ上のオレンジ色の破線は、<strong>損失の変化率が十分に小さくなった地点</strong>＝収束ポイントを示す。
            この地点以降は、学習を続けても損失がほとんど改善しない。
            <br><strong>収束 ≠ 完璧</strong> — 収束後も Loss が 0 でない場合、
            それはデータ自体に持つ不確実性（同じ質問に複数の正解がある等）に由来する。
            このツールでは「1ステップの Loss 減少が全体降下量の 2% 未満」になった地点を収束とみなしている。
            <br><strong>実務での判断</strong> — 実際の LLM 学習では、検証データでの Loss が改善しなくなった時点で
            学習を止める「Early Stopping」が広く使われる。</dd>

            <dt>📊 予測確率の変化</dt>
            <dd>初期状態では全トークンがほぼ均等な確率（＝何も学んでいない状態）。
            学習が進むと、正解トークンの確率が上がっていく様子が見える。
            「忠実な動物は→犬 99%」のように1つしか正解がない質問は確率が集中し、
            「人気のペットは→猫/犬」のように複数の正解がある質問は確率が分散する。
            <strong>なぜ分散するのか？</strong> → 学習データに両方の答えがあるため、モデルは「どちらもありうる」と学習する。</dd>

            <dt>🟧 Attention Weight の変化</dt>
            <dd>初期状態ではほぼ均一（どこにも注目していない）。
            学習が進むと、特定のトークンに注目するパターンが形成される。
            例えば「ペット」というトークンに強い注目が集まるようになれば、
            モデルが「ペット」という文脈を手がかりに予測することを学んだことを意味する。
            <br><strong>なぜ三角形？</strong>
            ヒートマップが左下三角形になるのは Causal Mask（因果マスク）の効果。
            Decoder は「次の単語を予測する」モデルなので、各トークンは自分より<strong>前のトークンだけ</strong>を参照できる。
            未来のトークンは常にマスクされるため、右上は必ずゼロ＝三角形になる。これは学習の結果ではなく構造上の制約。
            <br><strong>Head 1, Head 2 とは？</strong>
            Attention の計算を複数の「ヘッド」に分けて並列実行する仕組み（Multi-Head Attention）。
            例えば Head 1 は「直前の単語に注目」、Head 2 は「文の主語に注目」など、
            異なる観点で注目先を学習する。<strong>ヘッド間でパターンが違うほど、多角的に文脈を捉えている</strong>ことになる。
            逆にパターンがほぼ同じなら、ヘッドを分けた効果が薄い（冗長な状態）。
            <br><strong>紫色の「差分 |H1−H2|」</strong> —
            Head 1 と Head 2 の注目度の差の絶対値。紫が濃い箇所は「2つのヘッドが異なる判断をしている位置」。
            学習初期は差分がほぼゼロ（両ヘッドとも均一）だが、
            学習が進むと差分が大きくなる ＝ 各ヘッドが異なる役割を獲得していることを意味する。</dd>
        </dl>
    </div>

    <script>
    (function() {{
        const snapshots = {snap_json};
        const answerOrder = {answer_order_json};
        const convergenceIdx = {convergence_idx};
        const slider = document.getElementById('anim-slider');
        const epochLabel = document.getElementById('anim-epoch');
        const lossBar = document.getElementById('anim-loss-bar');
        const lossVal = document.getElementById('anim-loss-val');
        const lossChart = document.getElementById('anim-loss-chart');
        const predDiv = document.getElementById('anim-predictions');
        const attnDiv = document.getElementById('anim-attention');
        const playBtn = document.getElementById('anim-play');

        let currentQuery = 0;
        let playing = false;
        let playTimer = null;
        let currentIdx = 0;

        if (snapshots.length === 0) return;
        slider.max = snapshots.length - 1;

        // Loss chart のドットを初期描画
        const maxLoss = Math.max(...snapshots.map(s => s.loss));
        snapshots.forEach((s, i) => {{
            const dot = document.createElement('div');
            dot.className = 'loss-dot';
            dot.style.left = (i / (snapshots.length - 1) * 100) + '%';
            dot.style.bottom = (s.loss / maxLoss * 100) + '%';
            dot.dataset.idx = i;
            lossChart.appendChild(dot);
        }});

        // 収束ラインを描画（初期は非表示）
        let convLine = null;
        let convLabel = null;
        if (convergenceIdx >= 0 && snapshots.length > 1) {{
            const xPct = (convergenceIdx / (snapshots.length - 1) * 100);
            convLine = document.createElement('div');
            convLine.className = 'convergence-line';
            convLine.style.left = xPct + '%';
            convLine.style.display = 'none';
            lossChart.appendChild(convLine);

            convLabel = document.createElement('div');
            convLabel.className = 'convergence-label';
            convLabel.style.left = xPct + '%';
            convLabel.style.display = 'none';
            convLabel.textContent = '↓ 収束（' + snapshots[convergenceIdx].epoch + ' エポック目）';
            lossChart.appendChild(convLabel);
        }}

        function render(idx) {{
            currentIdx = idx;
            const snap = snapshots[idx];
            epochLabel.textContent = 'epoch ' + snap.epoch;
            const lossPct = (snap.loss / maxLoss * 100);
            lossBar.style.width = lossPct + '%';
            lossVal.textContent = snap.loss.toFixed(4);

            // Loss chart のアクティブドット
            lossChart.querySelectorAll('.loss-dot').forEach((d, i) => {{
                d.classList.toggle('active', i <= idx);
                d.classList.toggle('current', i === idx);
            }});

            // 収束ラインの表示制御（到達時にフェードイン、それ以降は表示維持）
            if (convLine && convLabel) {{
                if (idx >= convergenceIdx) {{
                    convLine.style.display = '';
                    convLabel.style.display = '';
                }} else {{
                    convLine.style.display = 'none';
                    convLabel.style.display = 'none';
                }}
            }}

            // 予測バー（固定順序: answerOrder）
            const q = snap.queries[currentQuery];
            if (q && q.prob_dict) {{
                // 最大確率を取得して正規化
                let maxP = 0;
                answerOrder.forEach(token => {{
                    const p = q.prob_dict[token] || 0;
                    if (p > maxP) maxP = p;
                }});
                let html = '';
                answerOrder.forEach(token => {{
                    const prob = q.prob_dict[token] || 0;
                    const pct = (prob * 100).toFixed(1);
                    const barWidth = maxP > 0 ? (prob / maxP * 100) : 0;
                    html += '<div class="pred-row">' +
                        '<span class="pred-label">' + token + '</span>' +
                        '<div class="pred-track"><div class="pred-fill" style="width:' + barWidth.toFixed(1) + '%"></div></div>' +
                        '<span class="pred-pct">' + pct + '%</span></div>';
                }});
                predDiv.innerHTML = html;
            }}

            // Attention heatmap
            if (q && q.attention_weights) {{
                const weights = q.attention_weights;
                let html = '';
                weights.forEach((head, hi) => {{
                    html += '<div class="attn-head"><div class="attn-title">Head ' + (hi+1) + '</div>';
                    html += '<div class="attn-grid" style="--size:' + head.length + '">';
                    head.forEach(row => {{
                        row.forEach(v => {{
                            const opacity = Math.min(1, v);
                            html += '<div class="attn-cell" style="opacity:' + opacity.toFixed(2) + '"></div>';
                        }});
                    }});
                    html += '</div></div>';
                }});

                // 差分ヒートマップ（Head が2つ以上ある場合）
                if (weights.length >= 2) {{
                    const h0 = weights[0], h1 = weights[1];
                    const size = h0.length;
                    html += '<div class="attn-head"><div class="attn-title attn-title-diff">差分 |H1−H2|</div>';
                    html += '<div class="attn-grid" style="--size:' + size + '">';
                    for (let r = 0; r < size; r++) {{
                        for (let c = 0; c < size; c++) {{
                            const diff = Math.abs((h0[r]?.[c] || 0) - (h1[r]?.[c] || 0));
                            const opacity = Math.min(1, diff * 2);
                            html += '<div class="attn-cell attn-cell-diff" style="opacity:' + opacity.toFixed(2) + '"></div>';
                        }}
                    }}
                    html += '</div></div>';
                }}

                attnDiv.innerHTML = html;
            }}
        }}

        window.selectQuery = function(qi) {{
            currentQuery = qi;
            document.querySelectorAll('.qtab').forEach((b, i) => b.classList.toggle('active', i === qi));
            // 再生中なら停止
            if (playing) {{
                clearInterval(playTimer);
                playBtn.textContent = '▶ 再生';
                playing = false;
            }}
            // ステップ位置を維持して再描画
            render(currentIdx);
        }};

        window.seekSnapshot = function(val) {{
            render(parseInt(val));
        }};

        window.stepForward = function() {{
            if (playing) {{
                clearInterval(playTimer);
                playBtn.textContent = '▶ 再生';
                playing = false;
            }}
            currentIdx++;
            if (currentIdx >= snapshots.length) currentIdx = 0;
            slider.value = currentIdx;
            render(currentIdx);
        }};

        window.togglePlay = function() {{
            if (playing) {{
                clearInterval(playTimer);
                playBtn.textContent = '▶ 再生';
                playing = false;
            }} else {{
                if (currentIdx >= snapshots.length - 1) {{
                    currentIdx = 0;
                    slider.value = 0;
                }}
                playing = true;
                playBtn.textContent = '⏸ 一時停止';
                playTimer = setInterval(() => {{
                    currentIdx++;
                    if (currentIdx >= snapshots.length) {{
                        currentIdx = snapshots.length - 1;
                        clearInterval(playTimer);
                        playBtn.textContent = '▶ 再生';
                        playing = false;
                        return;
                    }}
                    slider.value = currentIdx;
                    render(currentIdx);
                }}, {animation_speed});
            }}
        }};

        // 初期描画
        render(0);
    }})();
    </script>
    '''


def _inference_summary_table(all_query_data, id2token):
    """推論結果の一覧テーブル（推論ページ末尾用）"""
    rows = ""
    for qd in all_query_data:
        out = qd["result"]["output"]
        probs = out["probabilities"]
        top4 = sorted(
            [(id2token.get(i, f"id={i}"), float(p)) for i, p in enumerate(probs) if float(p) > 0.005],
            key=lambda x: -x[1]
        )[:4]
        preds = " / ".join(
            [f'<strong>{html_mod.escape(t)}</strong> ({p:.0%})' for t, p in top4]
        )
        rows += f'<tr><td>{html_mod.escape(qd["query"])}</td><td>{preds}</td></tr>\n'

    return f'''
    <h3>推論結果一覧</h3>
    <table class="data-table result-table">
        <tr><th>入力（質問）</th><th>予測結果</th></tr>
        {rows}
    </table>
    '''


def page_inference_all(all_query_data, id2token, answer_order, animation_speed=1200, n_layers=4):
    """全クエリの推論を1ページにまとめ、ボタンで切り替える"""
    # d_model を取得
    d_model = all_query_data[0]["result"]["embedding"]["combined"].shape[1] if all_query_data else 16
    # 各クエリのアニメーションHTMLを生成
    query_buttons = ""
    query_panels = ""
    for qi, qd in enumerate(all_query_data):
        active = "active" if qi == 0 else ""
        display = "block" if qi == 0 else "none"
        query_buttons += (
            f'<button class="infer-qbtn {active}" '
            f'onclick="switchInferQuery({qi})">'
            f'{html_mod.escape(qd["query"])}</button> '
        )
        panel_html = _build_inference_panel(
            qd["query"], qd["tokens"], qd["result"],
            id2token, answer_order, qi, animation_speed
        )
        query_panels += f'<div class="infer-qpanel" id="infer-qpanel-{qi}" style="display:{display}">{panel_html}</div>\n'

    return f'''
    <div class="step-header">推論（Inference）</div>
    <div class="desc">
        <p>{_kw("学習済みモデルに質問を入力し、Transformer の各レイヤーを通過して 次トークン予測 に至る過程をアニメーションで表示します。")}</p>
        <p>下のボタンで質問を切り替え、▶ 再生 で処理の流れを確認できます。</p>
    </div>

    <!-- ===== 処理フロー ===== -->
    <h3>⚙️ Transformer の処理フロー</h3>
    <div class="flow">
        <div class="flow-item"><span class="flow-step">STEP 1</span> <span class="flow-name">{_kw("Embedding")}</span> <span class="flow-desc">{_kw("トークン")}を{_kw("ベクトル")}に変換 + 位置情報を付与</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-item"><span class="flow-step">STEP 2</span> <span class="flow-name">{_kw("Decoder")} Layer ×{n_layers}</span> <span class="flow-desc">以下を各{_kw("レイヤー")}で繰り返す</span></div>
        <div class="flow-sub"><span class="flow-step">2a</span> <span class="flow-name">Masked {_kw("Self-Attention")}</span> <span class="flow-desc">各{_kw("トークン")}が「過去のどの{_kw("トークン")}に注目するか」を計算</span></div>
        <div class="flow-sub"><span class="flow-step">2b</span> <span class="flow-name">Add &amp; {_kw("LayerNorm")}</span> <span class="flow-desc">{_kw("残差接続")}で情報を保持 + {_kw("正規化")}</span></div>
        <div class="flow-sub"><span class="flow-step">2c</span> <span class="flow-name">{_kw("FFN")}</span> <span class="flow-desc">各{_kw("トークン")}を独立に非線形変換 ({_kw("ReLU")})</span></div>
        <div class="flow-sub"><span class="flow-step">2d</span> <span class="flow-name">Add &amp; {_kw("LayerNorm")}</span> <span class="flow-desc">再び{_kw("残差接続")} + {_kw("正規化")}</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-item"><span class="flow-step">STEP 3</span> <span class="flow-name">Output Head</span> <span class="flow-desc">最終{_kw("ベクトル")}から{_kw("次トークン予測")}の{_kw("確率分布")}を出力</span></div>
    </div>

    <!-- ===== 操作方法 ===== -->
    <div class="desc" style="background:#FFF3E0; border-left:4px solid #E65100; padding:0.8em 1em; margin:1em 0;">
        <strong>🖱️ 操作方法:</strong>
        <strong>▶ 再生</strong>で各ステップのアニメーション開始、<strong>次へ ▶|</strong> で1ステップ送り。
        青い用語にマウスを重ねると解説が表示されます。
    </div>

    <div class="infer-query-selector">
        {query_buttons}
    </div>

    {query_panels}

    <div class="observe-box">
        <h4>🔎 推論アニメーションの着目ポイント</h4>
        <dl class="observe-list">
            <dt>📐 Embedding（ステップ1）</dt>
            <dd>各トークンが数値ベクトルに変換された状態。
            ヒートマップの色は <span style="color:#c62828">赤 = 正の大きな値</span>、<span style="color:#1565C0">青 = 負の大きな値</span>、白 = ゼロ付近。
            学習によって「似た意味の単語は似た色パターン」になっているか観察してみよう。
            この時点ではトークン間の情報交換はまだ行われていない。<br>
            <strong>d0〜d{d_model - 1} の各次元に決まった意味はない。</strong>
            学習前はランダムな値で、学習を通じて複数の次元の組み合わせとして意味のあるパターンが自然に生まれる（分散表現）。
            シードを変えれば各次元の値はまったく変わるが、モデルの予測性能には影響しない。
            大規模 LLM では数千次元の空間内に「感情」「文法的役割」などの方向が見出される研究もあるが、それも個々の次元ではなく次元空間内の方向（direction）の話である。</dd>

            <dt>🟧 Attention Weight（各レイヤー上段）</dt>
            <dd><strong>行 = 注目する側、列 = 注目される側。</strong>
            「は」のトークンが「ペット」に強く注目しているなら、モデルは「ペットは？」という構造を理解している。
            レイヤーが深くなるほど、より抽象的な関係（意味的類似性など）を捉えるようになる。
            異なる質問間で Attention パターンを比較すると、モデルが文脈をどう区別しているかがわかる。<br>
            <strong style="color:#7B1FA2">差分 |H1−H2|（紫）</strong>が濃い箇所は、2つの Head が異なる注目をしている位置。
            濃い ＝ 役割分担ができている。質問ごとに差分パターンが変わるのは、文脈に応じて各 Head の注目先が変化している証拠。</dd>

            <dt>⭐ レイヤーを通じた表現の変化（核心）</dt>
            <dd><strong>Layer 1 → Layer 4 でヒートマップが次元ごとに整列化していくのが最も重要な観察。</strong>
            これは、バラバラだった単語の集まりが、レイヤーを重ねるごとに「予測に使える文脈表現」へ加工されている証拠。
            各レイヤーの Attention が「どのトークンが重要か」を判断して情報を集約し、FFN がそれを変換する — この繰り返しで表現が洗練される。
            特に最後のトークン位置には文全体の文脈が集約され、これが次トークン予測に使われる。<br>
            <strong>質問ごとにパターンが違う</strong> ＝ 異なる文脈を区別できている。
            学習前のランダムな重みではこの整列化は起きない。つまりこれは学習の成果そのもの。<br>
            <strong>試してみよう:</strong> Layer 4 のステップを選んだ状態で質問ボタンを切り替えると、
            どの質問でもヒートマップが整列していることがわかる。ステップ位置は質問切り替え時も維持される。</dd>

            <dt>📊 FFN 出力（各レイヤー下段）</dt>
            <dd>Attention の結果をもとに各トークンの表現を変換した結果。
            レイヤーを通過するたびに色パターンが変化する — これがモデルの「思考過程」に相当する。
            Layer 1 と Layer 4 を比べると、表現がどれだけ変化したかがわかる。</dd>

            <dt>🎯 予測結果（最終ステップ）</dt>
            <dd>全レイヤーを通過した結果の最終予測。
            「忠実な動物は」と「人気のペットは」で確率分布がどう違うか比較してみよう。
            学習データの傾向がそのまま反映されていることが確認できる。</dd>

            <dt>🌡️ Temperature スライダー</dt>
            <dd>予測結果の画面にある Temperature スライダーで、予測のランダム性を体感できる。<br>
            <strong>仕組み:</strong> モデルの最終出力は「Logit」と呼ばれる生のスコア（例: 猫=2.1, 犬=1.9, 狐=0.5）。
            これを Softmax 関数で確率に変換するが、その前に Temperature T で割る：
            <code>確率 = Softmax(Logit / T)</code><br>
            <strong>T が小さい（0.1〜0.5）:</strong> Logit の差が拡大され、1位がほぼ100%に。回答が確定的になる。<br>
            <strong>T = 1.0:</strong> 通常の確率分布。学習結果がそのまま反映される。<br>
            <strong>T が大きい（2.0〜3.0）:</strong> Logit の差が縮小され、全候補が均等に近づく。回答がランダムになる。<br>
            実際の AI チャットの「創造性」や「Temperature」設定はまさにこれ。
            低い値で事実に忠実な回答、高い値で創造的だが不確実な回答が得られる。</dd>

            <dt>🎲 Greedy vs Sampling</dt>
            <dd>確率分布が得られた後、実際にどのトークンを選ぶかの方式が2つある。<br>
            <strong>Greedy:</strong> 常に最大確率のトークンを選択する。同じ入力には毎回同じ結果。確実だが画一的。<br>
            <strong>Sampling:</strong> 確率分布に従ってランダムに1つ引く。サイコロを振るようなもの。
            猫35%・犬33%なら、猫が出やすいがたまに犬も出る。<br>
            <strong>試してみよう:</strong> 「Sampling」モードに切り替えて 🎲 ボタンを連打すると、
            同じ質問でも毎回違う答えが選ばれることがわかる。
            Temperature を上げるとさらにバラつきが増える。
            実際の AI チャットが同じ質問に毎回少し違う回答をするのはこの仕組みによるもの。</dd>
        </dl>
    </div>

    {_inference_summary_table(all_query_data, id2token)}

    <script>
    function switchEmbToken(el) {{
        const uid = el.dataset.evd;
        const idx = parseInt(el.dataset.idx);
        // トークンリンクの active 切り替え
        document.querySelectorAll('.ev-tok[data-evd="' + uid + '"]').forEach(function(s) {{
            s.classList.toggle('ev-tok-active', parseInt(s.dataset.idx) === idx);
        }});
        // テーブル切り替え
        let t = 0;
        while (document.getElementById(uid + '-' + t)) {{
            document.getElementById(uid + '-' + t).style.display = (t === idx) ? '' : 'none';
            t++;
        }}
    }}
    var _inferActiveQi = 0;
    function switchInferQuery(qi) {{
        // 現在のクエリのステップ位置を取得
        const prevSlider = document.getElementById('infer-slider-' + _inferActiveQi);
        const currentStep = prevSlider ? parseInt(prevSlider.value) : 0;

        document.querySelectorAll('.infer-qpanel').forEach((el, i) => {{
            el.style.display = i === qi ? 'block' : 'none';
        }});
        document.querySelectorAll('.infer-qbtn').forEach((el, i) => {{
            el.classList.toggle('active', i === qi);
        }});

        // 新しいクエリに同じステップ位置を適用
        const newSlider = document.getElementById('infer-slider-' + qi);
        if (newSlider) {{
            const maxStep = parseInt(newSlider.max);
            const targetStep = Math.min(currentStep, maxStep);
            if (window['inferSeek_' + qi]) {{
                window['inferSeek_' + qi](targetStep);
            }}
        }}
        _inferActiveQi = qi;
    }}
    </script>
    '''


def _build_inference_panel(query_text, tokens, result, id2token, answer_order, qi,
                           animation_speed=1200):
    """1クエリ分の推論アニメーションパネルを生成（page_inference_all の内部用）"""
    import json as _json

    out = result["output"]
    steps = []

    n_layers = len(result["layers"])
    n_heads = result["layers"][0]["attention"]["attention_weights"].shape[0] if result["layers"] else 2

    # Embedding（Token Embedding + Positional Encoding の詳細付き）
    emb = result["embedding"]
    d_model = emb["combined"].shape[1] if hasattr(emb["combined"], "shape") else len(emb["combined"][0])
    emb_detail = {
        "token_emb": emb["token_embedding"].tolist() if "token_embedding" in emb else None,
        "pos_enc": emb["positional_encoding"].tolist() if "positional_encoding" in emb else None,
        "d_model": d_model,
    }
    steps.append({
        "label": "Embedding",
        "desc": "各トークン（単語）を数値ベクトルに変換し、位置情報（何番目の単語か）を加算する。"
                "この数値の並びがトークンの「意味」と「位置」を表現する。"
                "表の各行が1つのトークン、各列がベクトルの1次元に対応（d_model=" + str(d_model) + "次元）。",
        "heatmap": emb["combined"].tolist(),
        "heatmap_labels": tokens,
        "embedding_detail": emb_detail,
    })

    # 各レイヤー（Attention + FFN を1ステップにまとめる）
    for li, lr in enumerate(result["layers"]):
        attn = lr["attention"]
        steps.append({
            "label": f"Layer {li+1}",
            "desc": f"【上: Attention Weight】各トークンが他のどのトークンに注目しているかを示す。"
                    f"Head 1〜{n_heads} は異なる観点（文法的関係・意味的関係など）で独立に注目先を計算する。"
                    f"行が「注目する側」、列が「注目される側」。色が濃いほど強く注目。各行の合計は1.0。"
                    f"右上が0.00なのは因果マスクにより未来のトークンを参照できないため。\n"
                    f"【下: FFN 出力】Attention の結果をもとに各トークンの表現を非線形変換した結果。"
                    f"レイヤーを通過するたびに、より文脈を反映した表現になっていく。",
            "layer": {
                "attention": attn["attention_weights"].tolist(),
                "attention_labels": tokens,
                "ffn_output": lr["norm2"]["output"].tolist(),
                "ffn_labels": tokens,
                "ffn_sparsity": float(lr["ffn"]["relu_sparsity"]),
            },
        })

    # 予測（生 logit も保存 → Temperature スライダー用）
    pred_probs = {}
    pred_logits = {}
    for idx in range(len(out["probabilities"])):
        tok = id2token.get(idx, f"id={idx}")
        pred_probs[tok] = float(out["probabilities"][idx])
        pred_logits[tok] = float(out["last_logits"][idx])
    steps.append({
        "label": "予測結果",
        "desc": (f"全{n_layers}レイヤーを通過した最終ベクトルを語彙サイズに射影し、"
                 "Softmax で確率分布に変換した結果。"
                 "Temperature スライダーで予測のランダム性を調整できる。"),
        "predictions": pred_probs,
        "logits": pred_logits,
    })

    steps_json = _json.dumps(steps, ensure_ascii=False)
    answer_order_json = _json.dumps(answer_order, ensure_ascii=False)

    # パイプライン表示
    step_labels_html = ""
    for i, s in enumerate(steps):
        step_labels_html += f'<div class="pipe-step" id="pipe-{qi}-{i}" onclick="inferSeek_{qi}({i})">{s["label"]}</div>'
        if i < len(steps) - 1:
            step_labels_html += '<div class="pipe-arrow">↓</div>'

    return f'''
    <div class="infer-anim-layout">
        <div class="infer-pipeline">
            <div class="infer-tokens">{_token_chips(tokens)}</div>
            {step_labels_html}
        </div>
        <div class="infer-viz">
            <div class="anim-controls">
                <button id="infer-play-{qi}" class="anim-btn" onclick="inferToggle_{qi}()">▶ 再生</button>
                <button class="anim-btn anim-btn-step" onclick="inferStep_{qi}()">次へ ▶|</button>
                <input type="range" id="infer-slider-{qi}" min="0" max="{len(steps)-1}" value="0"
                       oninput="inferSeek_{qi}(this.value)" class="anim-slider">
                <span id="infer-step-{qi}" class="anim-epoch">-</span>
            </div>
            <div id="infer-desc-{qi}" class="infer-step-desc"></div>
            <div id="infer-content-{qi}" class="infer-content"></div>
        </div>
    </div>

    <script>
    (function() {{
        const steps = {steps_json};
        const answerOrder = {answer_order_json};
        const qi = {qi};
        const slider = document.getElementById('infer-slider-' + qi);
        const stepLabel = document.getElementById('infer-step-' + qi);
        const descDiv = document.getElementById('infer-desc-' + qi);
        const contentDiv = document.getElementById('infer-content-' + qi);
        const playBtn = document.getElementById('infer-play-' + qi);
        let playing = false, timer = null, idx = 0;

        function renderHeatmap(data, labels) {{
            const rows = data.length, cols = data[0].length;
            let vals = data.flat();
            let mn = Math.min(...vals.filter(v => isFinite(v)));
            let mx = Math.max(...vals.filter(v => isFinite(v)));
            if (mn === mx) mx = mn + 1;
            let html = '<table class="heatmap"><tr><th></th>';
            for (let j = 0; j < cols; j++) html += '<th class="hm-col-label">d' + j + '</th>';
            html += '</tr>';
            for (let i = 0; i < rows; i++) {{
                html += '<tr><td class="hm-row-label">' + (labels[i]||i) + '</td>';
                for (let j = 0; j < cols; j++) {{
                    const v = data[i][j];
                    const t = (v - mn) / (mx - mn);
                    let r, g, b;
                    if (t < 0.5) {{ const s = t*2; r=Math.round(59+s*196); g=Math.round(76+s*179); b=Math.round(192+s*63); }}
                    else {{ const s=(t-0.5)*2; r=Math.round(255-s*37); g=Math.round(255-s*186); b=Math.round(255-s*186); }}
                    const tc = (t<0.25||t>0.75) ? '#fff' : '#333';
                    html += '<td class="hm-cell" style="background:rgb('+r+','+g+','+b+');color:'+tc+'">'+v.toFixed(2)+'</td>';
                }}
                html += '</tr>';
            }}
            return html + '</table>';
        }}

        function renderAttention(weights, labels) {{
            let html = '<div style="display:flex;gap:1.5em;flex-wrap:wrap;">';
            weights.forEach((head, hi) => {{
                html += '<div><div style="font-weight:bold;margin-bottom:4px;">Head '+(hi+1)+'</div>';
                html += '<table class="heatmap"><tr><th></th>';
                labels.forEach(l => html += '<th class="hm-col-label">'+l+'</th>');
                html += '</tr>';
                head.forEach((row, ri) => {{
                    html += '<tr><td class="hm-row-label">'+labels[ri]+'</td>';
                    row.forEach(v => {{
                        const tc = v > 0.5 ? '#fff' : '#333';
                        html += '<td class="hm-cell" style="background:rgb('+Math.round(230-v*100)+','+Math.round(120+v*50)+',50);color:'+tc+'">'+v.toFixed(2)+'</td>';
                    }});
                    html += '</tr>';
                }});
                html += '</table></div>';
            }});
            // 差分ヒートマップ（Head が2つ以上ある場合）
            if (weights.length >= 2) {{
                const h0 = weights[0], h1 = weights[1];
                html += '<div><div style="font-weight:bold;margin-bottom:4px;color:#7B1FA2;">差分 |H1−H2|</div>';
                html += '<table class="heatmap"><tr><th></th>';
                labels.forEach(l => html += '<th class="hm-col-label">'+l+'</th>');
                html += '</tr>';
                h0.forEach((row, ri) => {{
                    html += '<tr><td class="hm-row-label">'+labels[ri]+'</td>';
                    row.forEach((v, ci) => {{
                        const diff = Math.abs(v - (h1[ri]?.[ci] || 0));
                        const opacity = Math.min(1, diff * 2);
                        html += '<td class="hm-cell" style="background:rgba(123,31,162,'+opacity.toFixed(2)+');color:'+(opacity>0.4?'#fff':'#333')+'">'+diff.toFixed(2)+'</td>';
                    }});
                    html += '</tr>';
                }});
                html += '</table></div>';
            }}
            return html + '</div>';
        }}

        function renderPredictions(probs, sampledToken) {{
            // 最大確率のトークンを特定
            let maxProb = 0, maxToken = '';
            answerOrder.forEach(token => {{
                const p = probs[token] || 0;
                if (p > maxProb) {{ maxProb = p; maxToken = token; }}
            }});
            // 選択されたトークン（Greedy=最大確率、Sampling=サンプル結果）
            const selectedToken = sampledToken || maxToken;
            const selectedProb = probs[selectedToken] || 0;
            const mode = sampledToken ? 'Sampling' : 'Greedy';
            let html = '';
            answerOrder.forEach(token => {{
                const prob = probs[token] || 0;
                const pct = (prob * 100).toFixed(1);
                const barWidth = maxProb > 0 ? (prob / maxProb * 100) : 0;
                const isSelected = (token === selectedToken);
                const rowClass = isSelected ? 'pred-row pred-selected' : 'pred-row';
                const badge = isSelected ? '<span class="pred-badge">▶ ' + mode + '</span>' : '';
                html += '<div class="' + rowClass + '">' +
                    '<span class="pred-label">' + token + '</span>' +
                    '<div class="pred-track"><div class="pred-fill' + (isSelected ? ' pred-fill-selected' : '') +
                    '" style="width:'+barWidth.toFixed(1)+'%"></div></div>' +
                    '<span class="pred-pct">'+pct+'%</span>' + badge + '</div>';
            }});
            html += '<div class="pred-answer">→ ' + mode + ' 予測: 「<strong>' + selectedToken + '</strong>」 (' + (selectedProb*100).toFixed(1) + '%)</div>';
            return html;
        }}

        // 確率分布からランダムにサンプリング
        function sampleFromProbs(probs) {{
            const tokens = answerOrder.slice();
            const ps = tokens.map(t => probs[t] || 0);
            const sum = ps.reduce((a, b) => a + b, 0);
            let r = Math.random() * sum;
            for (let i = 0; i < tokens.length; i++) {{
                r -= ps[i];
                if (r <= 0) return tokens[i];
            }}
            return tokens[tokens.length - 1];
        }}

        function render(i) {{
            idx = i;
            const step = steps[i];
            stepLabel.textContent = step.label;
            descDiv.textContent = step.desc;
            for (let s = 0; s < steps.length; s++) {{
                const el = document.getElementById('pipe-' + qi + '-' + s);
                if (el) {{
                    el.classList.toggle('pipe-active', s === i);
                    el.classList.toggle('pipe-done', s < i);
                }}
            }}
            let html = '';
            if (step.layer) {{
                // Attention + FFN を上下に表示
                const ly = step.layer;
                html = '<div class="layer-combined">';
                html += '<div class="layer-half"><h5>Attention Weight — 各トークンの注目先</h5>';
                html += renderAttention(ly.attention, ly.attention_labels);
                html += '</div>';
                html += '<div class="layer-half"><h5>FFN 出力 — 変換後のベクトル表現 (sparsity: ' + (ly.ffn_sparsity * 100).toFixed(0) + '%)</h5>';
                html += renderHeatmap(ly.ffn_output, ly.ffn_labels);
                html += '</div></div>';
            }} else if (step.heatmap) {{
                // Embedding 詳細がある場合はベクトル分解を表示
                if (step.embedding_detail && step.embedding_detail.token_emb) {{
                    const det = step.embedding_detail;
                    const labels = step.heatmap_labels;
                    const uid = 'evd-' + qi;
                    html += '<div class="emb-detail">';
                    // タイトル + トークン切り替えリンク
                    html += '<h5 style="display:inline">ベクトルの構成：</h5> ';
                    for (let t = 0; t < labels.length; t++) {{
                        const lbl = labels[t].replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        const cls = (t === 1 && labels.length > 1) || (t === 0 && labels.length === 1) ? ' ev-tok-active' : '';
                        html += '<span class="ev-tok' + cls + '" data-evd="' + uid + '" data-idx="' + t + '" onclick="switchEmbToken(this)">' + lbl + '</span> ';
                    }}
                    // 各トークンのテーブル（初期表示は最初の実トークン）
                    const initIdx = (labels.length > 1) ? 1 : 0;
                    for (let t = 0; t < labels.length; t++) {{
                        const vis = (t === initIdx) ? '' : ' style="display:none"';
                        html += '<table class="emb-vec-table" id="' + uid + '-' + t + '"' + vis + '>';
                        html += '<tr><th></th>';
                        for (let d = 0; d < det.d_model; d++) html += '<th class="ev-dim">d' + d + '</th>';
                        html += '</tr>';
                        html += '<tr><td class="ev-label">Token Emb</td>';
                        for (let d = 0; d < det.d_model; d++) {{
                            html += '<td class="ev-val">' + det.token_emb[t][d].toFixed(2) + '</td>';
                        }}
                        html += '</tr>';
                        html += '<tr><td class="ev-label">Pos Enc</td>';
                        for (let d = 0; d < det.d_model; d++) {{
                            html += '<td class="ev-val">' + det.pos_enc[t][d].toFixed(2) + '</td>';
                        }}
                        html += '</tr>';
                        html += '<tr class="ev-combined"><td class="ev-label"><strong>= Combined</strong></td>';
                        for (let d = 0; d < det.d_model; d++) {{
                            html += '<td class="ev-val"><strong>' + step.heatmap[t][d].toFixed(2) + '</strong></td>';
                        }}
                        html += '</tr>';
                        html += '</table>';
                    }}
                    html += '<p class="ev-note">↑ Token Embedding（単語の意味） + Positional Encoding（位置情報）＝ Combined（モデルへの入力ベクトル）</p>';
                    html += '</div>';
                }}
                html += '<h5>全トークンの Embedding（Combined）</h5>';
                html += renderHeatmap(step.heatmap, step.heatmap_labels);
            }} else if (step.predictions) {{
                if (step.logits) {{
                    // Temperature + Greedy/Sampling 切り替え
                    html = '<div class="temp-section">';
                    // モード切替
                    html += '<div class="decode-mode">';
                    html += '<button class="mode-btn mode-active" id="mode-greedy-'+qi+'" onclick="setDecodeMode_'+qi+'(&quot;greedy&quot;)">Greedy（最大確率を選択）</button>';
                    html += '<button class="mode-btn" id="mode-sampling-'+qi+'" onclick="setDecodeMode_'+qi+'(&quot;sampling&quot;)">Sampling（確率に従いランダム）</button>';
                    html += '</div>';
                    // Temperature
                    html += '<div class="temp-control">';
                    html += '<label><strong>Temperature</strong>: ';
                    html += '<input type="range" min="0.1" max="3.0" step="0.1" value="1.0" ';
                    html += 'class="temp-slider" id="temp-slider-'+qi+'" oninput="updateTemp_' + qi + '(this.value, this.parentNode)">';
                    html += ' <span class="temp-val">1.0</span></label>';
                    html += '<span class="temp-hint">← 確定的　　ランダム →</span>';
                    html += '</div>';
                    // サンプリングボタン（Sampling モード時のみ表示）
                    html += '<div class="sample-controls" id="sample-ctrl-'+qi+'" style="display:none">';
                    html += '<button class="sample-btn" onclick="doSample_'+qi+'()">🎲 サンプリング実行</button>';
                    html += '<span class="sample-hint">ボタンを押すたびに確率分布からランダムに1つ選ばれます</span>';
                    html += '</div>';
                    html += '<div class="anim-predictions" id="temp-preds-' + qi + '">';
                    html += renderPredictions(step.predictions);
                    html += '</div></div>';
                    // logit データとモードをグローバルに保持
                    window['_logits_' + qi] = step.logits;
                    window['_decodeMode_' + qi] = 'greedy';
                }} else {{
                    html = '<div class="anim-predictions">' + renderPredictions(step.predictions) + '</div>';
                }}
            }}
            contentDiv.innerHTML = html;
        }}

        window['inferSeek_' + qi] = function(v) {{
            const val = parseInt(v);
            document.getElementById('infer-slider-{qi}').value = val;
            render(val);
        }};
        window['inferStep_' + qi] = function() {{
            if (playing) {{
                clearInterval(timer); playBtn.textContent = '▶ 再生'; playing = false;
            }}
            idx++;
            if (idx >= steps.length) idx = 0;
            slider.value = idx; render(idx);
        }};

        // Temperature スライダーで Softmax 再計算
        // Temperature から確率を再計算する共通関数
        function calcProbs_qi() {{
            const slider = document.getElementById('temp-slider-' + qi);
            const T = slider ? parseFloat(slider.value) : 1.0;
            const logits = window['_logits_' + qi];
            if (!logits) return null;
            const scaled = {{}};
            let maxVal = -Infinity;
            answerOrder.forEach(tok => {{
                const v = (logits[tok] || 0) / T;
                scaled[tok] = v;
                if (v > maxVal) maxVal = v;
            }});
            let sumExp = 0;
            answerOrder.forEach(tok => {{ sumExp += Math.exp(scaled[tok] - maxVal); }});
            const probs = {{}};
            answerOrder.forEach(tok => {{ probs[tok] = Math.exp(scaled[tok] - maxVal) / sumExp; }});
            return probs;
        }}

        // Temperature スライダー変更時
        window['updateTemp_' + qi] = function(tempVal, labelEl) {{
            labelEl.querySelector('.temp-val').textContent = parseFloat(tempVal).toFixed(1);
            const probs = calcProbs_qi();
            if (!probs) return;
            const container = document.getElementById('temp-preds-' + qi);
            const mode = window['_decodeMode_' + qi];
            if (mode === 'sampling') {{
                const sampled = sampleFromProbs(probs);
                if (container) container.innerHTML = renderPredictions(probs, sampled);
            }} else {{
                if (container) container.innerHTML = renderPredictions(probs);
            }}
        }};

        // モード切替
        window['setDecodeMode_' + qi] = function(mode) {{
            window['_decodeMode_' + qi] = mode;
            const gBtn = document.getElementById('mode-greedy-' + qi);
            const sBtn = document.getElementById('mode-sampling-' + qi);
            const sCtrl = document.getElementById('sample-ctrl-' + qi);
            if (mode === 'greedy') {{
                gBtn.classList.add('mode-active'); sBtn.classList.remove('mode-active');
                sCtrl.style.display = 'none';
            }} else {{
                sBtn.classList.add('mode-active'); gBtn.classList.remove('mode-active');
                sCtrl.style.display = 'flex';
            }}
            // 再描画
            const probs = calcProbs_qi();
            if (!probs) return;
            const container = document.getElementById('temp-preds-' + qi);
            if (mode === 'sampling') {{
                const sampled = sampleFromProbs(probs);
                if (container) container.innerHTML = renderPredictions(probs, sampled);
            }} else {{
                if (container) container.innerHTML = renderPredictions(probs);
            }}
        }};

        // サンプリング実行
        window['doSample_' + qi] = function() {{
            const probs = calcProbs_qi();
            if (!probs) return;
            const sampled = sampleFromProbs(probs);
            const container = document.getElementById('temp-preds-' + qi);
            if (container) container.innerHTML = renderPredictions(probs, sampled);
        }};
        window['inferToggle_' + qi] = function() {{
            if (playing) {{
                clearInterval(timer); playBtn.textContent = '▶ 再生'; playing = false;
            }} else {{
                if (idx >= steps.length - 1) {{ idx = 0; slider.value = 0; }}
                playing = true; playBtn.textContent = '⏸ 停止';
                timer = setInterval(() => {{
                    idx++;
                    if (idx >= steps.length) {{
                        idx = steps.length - 1;
                        clearInterval(timer); playBtn.textContent = '▶ 再生'; playing = false; return;
                    }}
                    slider.value = idx; render(idx);
                }}, {animation_speed});
            }}
        }};
        render(0);
    }})();
    </script>
    '''




def page_inference_summary(query_results):
    """全クエリの推論結果サマリー"""
    rows = ""
    for qr in query_results:
        top = qr["top_predictions"][:4]
        preds = " / ".join(
            [f'<strong>{html_mod.escape(t)}</strong> ({p:.0%})' for t, p in top]
        )
        rows += f'''
        <tr>
            <td>{html_mod.escape(qr["query"])}</td>
            <td>{preds}</td>
        </tr>'''

    return f'''
    <div class="step-header">推論結果まとめ（Summary）</div>
    <div class="desc">
        <p>学習済みモデルに各質問を入力し、{_kw("次トークン予測")} を行った結果です。</p>
        <p>{_kw("Transformer が学習データから「文脈」と「答え」の関係を学習できていることがわかります。")}</p>
    </div>

    <table class="data-table result-table">
        <tr><th>入力（質問）</th><th>予測結果</th></tr>
        {rows}
    </table>
    '''


def page_customize():
    """カスタマイズガイドページ"""
    return f'''
    <div class="step-header">🛠️ カスタマイズ（Customization）</div>
    <div class="desc">
        <p>このプログラムの学習データと質問は、あなたの好みに合わせて自由に入れ替えることができます。
        料理、都市、プログラミング言語、歴史上の人物 — 何でも OK。</p>
        <p>プロファイル JSON ファイル（<code>profiles/default.json</code>）を編集するか、
        新しいファイルを作成して <code>--profile</code> で指定します。</p>
    </div>

    <div class="customize-section">
        <h4>📄 プロファイル JSON の構造</h4>
        <pre class="code-block">{{
  "title": "テーマのタイトル",

  "training_data": [
    {{"input": "質問文A", "output": "答え1"}},
    {{"input": "質問文B", "output": "答え1"}},
    {{"input": "質問文C", "output": "答え2"}},
    ...
  ],

  "queries": [
    "推論で試したい質問1",
    "推論で試したい質問2"
  ],

  "model": {{
    "layers": 4,
    "d_model": 16,
    "heads": 2,
    "d_ff": 64,
    "seed": 42
  }},

  "training": {{
    "epochs": 300,
    "lr": 0.005,
    "snapshot_count": 20,
    "animation_speed": 400
  }},

  "inference": {{
    "animation_speed": 1200,
    "max_tokens": 32,
    "temperature": 1.0
  }}
}}</pre>
    </div>

    <div class="customize-section">
        <h4>📝 学習データの作り方のコツ</h4>
        <ul class="customize-list">
            <li><strong>同じ答えに複数の質問を用意する</strong><br>
            「おいしい果物は」→ りんご、「赤い果物は」→ りんご、のように。
            1つの質問に1つの答えだけでは丸暗記になり、汎化しない。</li>
            <li><strong>答えが重なる質問を混ぜると面白い</strong><br>
            「赤い果物は」→ りんご/いちご のように、文脈で答えが分かれるデータがあると、
            モデルが{_kw("確率分布")}で予測する様子が観察できる。</li>
            <li><strong>データ量は 10〜30 件が適切</strong><br>
            少なすぎると学習できず、多すぎるとミニチュアモデルでは処理が重くなる。</li>
            <li><strong>推論クエリは学習データと同じ表現を使う</strong><br>
            「かわいい動物は」で学習したモデルに「可愛らしい生き物は」と聞いても正しく予測できない。
            人間には同じ意味でも、モデルにとっては全く別の{_kw("トークン")}列であり、別の入力として処理される。
            これは実際の {_kw("LLM")} との最大の違いの一つ。</li>
            <li><strong>青色の用語にはマウスオーバー解説がある</strong><br>
            全ページのテキスト中で<span class="kw" style="pointer-events:none;">青色ハイライト</span>された用語にマウスを乗せると解説が表示される。
            「📖 用語集」タブにカテゴリ別の一覧あり。
            プロファイル JSON の <code>"glossary"</code> で用語の追加・編集も可能。</li>
        </ul>
    </div>

    <div class="customize-section">
        <h4>⚙️ モデル{_kw("パラメータ")}の意味</h4>
        <table class="data-table">
            <tr><th>パラメータ</th><th>意味</th><th>推奨値</th></tr>
            <tr><td><code>layers</code></td><td>{_kw("レイヤー")}数。多いほど深い表現が可能だが学習に時間がかかる</td><td>2〜4</td></tr>
            <tr><td><code>{_kw("d_model")}</code></td><td>各{_kw("トークン")}の{_kw("ベクトル")}{_kw("次元")}数。大きいほど表現力が高いが{_kw("過学習")}しやすい</td><td>8〜32</td></tr>
            <tr><td><code>heads</code></td><td>{_kw("Attention")} {_kw("ヘッド")}数。異なる観点で注目できる数</td><td>2〜4</td></tr>
            <tr><td><code>d_ff</code></td><td>{_kw("FFN")} の中間層の{_kw("次元")}数</td><td>d_model×4</td></tr>
            <tr><td><code>epochs</code></td><td>{_kw("エポック")}数。学習データを何周するか</td><td>200〜500</td></tr>
            <tr><td><code>lr</code></td><td>{_kw("学習率")}。大きいと速いが不安定になる</td><td>0.001〜0.01</td></tr>
            <tr><td><code>temperature</code></td><td>{_kw("Temperature")}。推論時の予測のランダム性</td><td>0.5〜2.0</td></tr>
        </table>
    </div>

    <div class="customize-section">
        <h4>💡 テーマの例</h4>
        <table class="data-table">
            <tr><th>テーマ</th><th>学習データの例</th></tr>
            <tr><td>🍎 果物</td><td>「赤い果物は」→ りんご / 「黄色い果物は」→ バナナ / 「甘い果物は」→ りんご,バナナ</td></tr>
            <tr><td>🌍 首都</td><td>「日本の首都は」→ 東京 / 「アジアの大都市は」→ 東京,北京</td></tr>
            <tr><td>💻 言語</td><td>「Web開発の言語は」→ JavaScript / 「AI開発の言語は」→ Python</td></tr>
            <tr><td>📚 文学</td><td>「日本の文豪は」→ 夏目漱石 / 「明治の作家は」→ 夏目漱石,森鷗外</td></tr>
        </table>
    </div>

    <div class="customize-section">
        <h4>🖥️ コマンドの使い方</h4>
        <pre class="code-block"># デフォルト（動物クイズ）
python main.py

# 自分のプロファイルで実行
python main.py --profile profiles/my_data.json

# 追加の質問を試す
python main.py --query "かわいい動物は"

# 学習パラメータを変えてみる
python main.py --epochs 500 --lr 0.01

# 出力先を指定
python main.py -o my_result.html</pre>
    </div>
    '''


def _glossary_full_list():
    """用語集の全定義をカテゴリ別に展開したHTMLを生成"""
    html = '<div style="margin-top:2em;">\n'
    html += '    <h3>用語詳細一覧</h3>\n'

    for cat_name, terms in _GLOSSARY_CATEGORIES:
        valid_terms = [t for t in terms if t in GLOSSARY]
        if not valid_terms:
            continue
        html += f'    <div class="glossary-full-cat">\n'
        html += f'      <span class="glossary-cat">【 {html_mod.escape(cat_name)} 】</span>\n'
        html += '      <dl class="glossary-dl">\n'
        for term in valid_terms:
            safe_term = html_mod.escape(term)
            safe_def = html_mod.escape(GLOSSARY[term])
            html += f'        <dt>{safe_term}</dt>\n'
            html += f'        <dd>{safe_def}</dd>\n'
        html += '      </dl>\n'
        html += '    </div>\n'

    html += '</div>'
    return html


def page_glossary():
    """用語集ページ"""
    return f'''
    <div class="step-header">📖 用語集（Glossary）</div>
    <div class="desc">
        <p>このツールで使われる専門用語の一覧です。各用語にマウスを乗せると解説が表示されます（クリックで固定）。</p>
        <p style="color:#666;font-size:0.9em;">全ページのテキスト中で
        <span class="kw" style="pointer-events:none;">青色ハイライト</span>された用語にも同じ機能があります。
        「※」マークのある説明は、このツールの教育用実装と実際の LLM との違いを示しています。</p>
    </div>

    {_glossary_section()}

    {_glossary_full_list()}
    '''


# ================================================================
# HTML 組み立て
# ================================================================

def build_html(pages, title="Transformer Decoder Emulator"):
    """ページリスト [(tab_label, html_content), ...] → 完全な HTML"""
    import json as _json
    # GLOSSARY を JS 用 JSON に変換（_ 付きコメントキーを除外）
    glossary_for_js = {k: v for k, v in GLOSSARY.items() if not k.startswith("_")}
    glossary_json = _json.dumps(glossary_for_js, ensure_ascii=False)
    safe_title = html_mod.escape(title)

    # タブボタン
    tab_buttons = ""
    tab_contents = ""
    for i, (label, content) in enumerate(pages):
        active = "active" if i == 0 else ""
        tab_buttons += f'<button class="tab-btn {active}" onclick="showTab({i})">{html_mod.escape(label)}</button>\n'
        display = "block" if i == 0 else "none"
        tab_contents += f'<div class="tab-content" id="tab-{i}" style="display:{display}">\n{content}\n</div>\n'

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
:root {{
    --primary: #1A237E;
    --primary-dark: #0D1257;
    --accent: #E65100;
    --kw-color: #1565C0;
    --kw-bg: #E3F2FD;
    --bg: #FAFAFA;
    --card-bg: #FFFFFF;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: "Hiragino Sans", "Hiragino Kaku Gothic Pro", "Yu Gothic", "Noto Sans CJK JP", "Meiryo", sans-serif;
    background: var(--bg);
    color: #333;
    line-height: 1.7;
    padding: 0;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 1em; }}

/* タブ */
.tab-bar {{
    display: flex; flex-wrap: wrap; gap: 2px;
    background: #E8EAF6; padding: 4px; border-radius: 8px;
    margin-bottom: 1.5em; position: sticky; top: 0; z-index: 50;
}}
.tab-btn {{
    padding: 8px 14px; border: none; background: transparent;
    cursor: pointer; font-size: 13px; border-radius: 6px;
    color: #555; transition: all 0.15s;
    white-space: nowrap;
}}
.tab-btn:hover {{ background: #C5CAE9; }}
.tab-btn.active {{ background: var(--primary); color: #fff; font-weight: bold; }}

/* ステップ */
.step-header {{
    font-size: 1.5em; font-weight: bold; color: var(--primary);
    margin: 0.5em 0 0.8em; padding-bottom: 0.3em;
    border-bottom: 3px solid var(--primary);
}}
.subtitle {{ color: #666; margin-bottom: 1.5em; }}
.desc {{
    background: #F5F5F5; border-left: 4px solid var(--primary);
    padding: 1em 1.2em; margin-bottom: 1.5em; border-radius: 0 8px 8px 0;
    font-size: 1.05em;
}}
.desc p {{ margin: 0.3em 0; }}
.note {{ color: #888; font-size: 0.9em; font-style: italic; }}

/* キーワードツールチップ */
.kw {{
    color: var(--kw-color); font-weight: bold;
    border-bottom: 2px dotted var(--kw-color);
    cursor: help; position: relative;
}}
.kw:hover {{ background: var(--kw-bg); border-radius: 3px; }}
.kw .tip {{
    display: none; position: absolute; bottom: 130%; left: 50%;
    transform: translateX(-50%);
    background: #FFFDE7; border: 2px solid #FFA000;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; font-weight: normal; color: #333;
    white-space: pre-wrap; width: max-content; max-width: 400px;
    z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    line-height: 1.5; font-style: normal;
    pointer-events: none;
}}
.kw:hover .tip {{ display: block; }}
.kw .tip.pinned {{
    display: block; pointer-events: auto;
    border-color: #E65100;
}}
.tip-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #ffe0b2;
}}
.tip-title {{ font-weight: bold; color: var(--primary); }}
.tip-copy-btn {{
    padding: 2px 8px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 11px;
}}
.tip-copy-btn:hover {{ background: #eee; }}
.tip-close-btn {{
    padding: 2px 6px; border: none; background: none;
    cursor: pointer; font-size: 14px; color: #999;
}}
.tip-close-btn:hover {{ color: #333; }}
.kw .tip::after {{
    content: ""; position: absolute; top: 100%; left: 50%;
    transform: translateX(-50%);
    border: 8px solid transparent; border-top-color: #FFA000;
}}
.kw-demo {{ color: var(--kw-color); font-weight: bold; border-bottom: 2px dotted var(--kw-color); }}

/* ヒートマップ */
/* Embedding ベクトル詳細 */
.emb-detail {{
    background: #F5F5F5; border-radius: 8px; padding: 1em;
    margin-bottom: 1em; overflow-x: auto;
}}
.emb-detail h5 {{ margin: 0 0 0.5em; color: var(--primary); }}
.emb-vec-table {{
    border-collapse: collapse; font-size: 11px; font-family: monospace;
    white-space: nowrap;
}}
.emb-vec-table th, .emb-vec-table td {{ padding: 3px 6px; text-align: center; }}
.ev-dim {{ color: #888; font-size: 10px; }}
.ev-label {{ text-align: right; padding-right: 10px; color: #555; font-weight: bold; font-size: 12px; }}
.ev-val {{ border: 1px solid #ddd; background: #fff; }}
.ev-combined td {{ background: #E8EAF6 !important; }}
.ev-note {{ color: #666; font-size: 0.85em; margin: 0.5em 0 0; }}
.ev-tok {{
    display: inline-block; padding: 2px 8px; margin: 0 2px 4px;
    border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 0.95em;
    color: var(--primary); background: #fff; border: 2px solid #ccc;
    transition: all 0.15s;
}}
.ev-tok:hover {{ border-color: var(--primary); background: #E8EAF6; }}
.ev-tok-active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
.ev-tok-active:hover {{ background: var(--primary-dark); }}

.heatmap-row {{ display: flex; gap: 1.5em; flex-wrap: wrap; margin-bottom: 1.5em; }}
.heatmap-row > div {{ flex: 1; min-width: 280px; }}
.heatmap-container {{ overflow-x: auto; }}
.heatmap {{
    border-collapse: collapse; font-size: 12px;
    width: 100%;
}}
.heatmap td, .heatmap th {{ padding: 4px 6px; text-align: center; white-space: nowrap; }}
.hm-cell {{
    min-width: 45px; font-size: 11px; font-family: monospace;
    border: 1px solid rgba(255,255,255,0.3);
}}
.hm-nan {{ background: #eee !important; color: #999; font-style: italic; }}
.hm-row-label {{ font-weight: bold; text-align: right !important; padding-right: 8px !important; color: #555; }}
.hm-col-label {{ font-size: 10px; color: #888; }}

/* 棒グラフ */
.bar-chart {{ max-width: 600px; }}
.bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
.bar-label {{ width: 80px; font-weight: bold; text-align: right; padding-right: 12px; }}
.bar-track {{ flex: 1; height: 28px; background: #eee; border-radius: 4px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: linear-gradient(90deg, #FF8A65, #E65100); border-radius: 4px; transition: width 0.5s; }}
.bar-value {{ width: 70px; padding-left: 8px; font-family: monospace; font-size: 13px; }}

/* トークンチップ */
.token-line {{ margin: 1em 0; line-height: 2.5; }}
.token-input, .token-gen {{
    display: inline-block; padding: 4px 10px; border-radius: 6px;
    font-weight: bold; margin: 2px;
}}
.token-input {{ background: #E8EAF6; border: 1px solid #7986CB; color: #333; }}
.token-gen {{ background: #FFF3E0; border: 1px solid #FFB74D; color: var(--accent); }}
.token-arrow {{ color: #999; margin: 0 2px; }}

/* フロー */
.flow {{ margin: 1em 0 2em; }}
.flow-item, .flow-sub {{
    display: grid;
    grid-template-columns: 60px 280px 1fr;
    align-items: baseline;
    padding: 8px 0;
    gap: 0.5em;
}}
.flow-sub {{ grid-template-columns: 60px 280px 1fr; padding-left: 1em; }}
.flow-step {{ color: var(--accent); font-weight: bold; font-family: monospace; }}
.flow-name {{ font-weight: bold; white-space: nowrap; }}
.flow-desc {{ color: #666; font-size: 0.9em; }}
.flow-arrow {{ color: #999; padding-left: 60px; padding-top: 2px; padding-bottom: 2px; }}
.kw {{ white-space: nowrap; }}

/* サマリー */
.summary-box {{
    background: #E8EAF6; border-radius: 8px; padding: 1.5em;
    margin: 1em 0; font-size: 1.1em;
}}
.summary-row {{ margin: 0.5em 0; }}
.summary-label {{ color: #666; min-width: 50px; display: inline-block; }}
.gen-text {{ color: var(--accent); font-weight: bold; font-size: 1.2em; }}
.gen-target {{ color: var(--accent); font-weight: bold; font-size: 1.1em; margin-top: 0.5em; }}

/* 情報ボックス */
.info-box {{
    background: #E8EAF6; border: 1px solid #7986CB;
    border-radius: 8px; padding: 1em 1.5em; margin: 1em 0;
    line-height: 2;
}}

/* 注意ボックス */
.note-box {{
    background: #FFF8E1; border: 1px solid #FFB74D;
    border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0;
    line-height: 1.7; font-size: 0.95em; color: #555;
}}

/* 解説リスト */
.explain-list li {{
    margin: 1em 0; line-height: 1.7;
}}

/* フェーズボックス */
.phase-box {{
    border-radius: 8px; padding: 1.2em 1.5em; margin: 1em 0;
    line-height: 1.8;
}}
.phase-box h4 {{ margin: 0 0 0.5em; font-size: 1.1em; }}
.phase-box ol {{ margin: 0.5em 0 0.5em 1.5em; }}
.phase-box li {{ margin: 0.3em 0; }}
.phase-train {{
    background: #E3F2FD; border-left: 5px solid #1565C0;
}}
.phase-train h4 {{ color: #1565C0; }}

/* 着目ポイントボックス */
.observe-box {{
    background: #F3E5F5; border-left: 5px solid #7B1FA2;
    border-radius: 0 8px 8px 0; padding: 1em 1.5em; margin: 1.5em 0;
}}
.observe-box h4 {{ color: #7B1FA2; margin: 0 0 0.8em; }}
.observe-list {{ margin: 0; }}
.observe-list dt {{
    font-weight: bold; color: #4A148C; margin-top: 0.8em;
    font-size: 1.0em;
}}
.observe-list dd {{
    margin: 0.3em 0 0.8em 0; line-height: 1.7; color: #333;
}}
.phase-infer {{
    background: #FFF3E0; border-left: 5px solid #E65100;
}}
.phase-infer h4 {{ color: #E65100; }}

/* ページ遷移ボタン（フェーズ内） */
.goto-btn {{
    display: inline-block; margin-top: 0.8em;
    padding: 8px 20px; border: none; border-radius: 6px;
    font-size: 14px; font-weight: bold; cursor: pointer;
    color: #fff; transition: opacity 0.15s;
}}
.goto-btn:hover {{ opacity: 0.85; }}
.goto-train {{ background: #1565C0; }}
.goto-infer {{ background: #E65100; }}
.goto-customize {{ background: #2E7D32; }}
.goto-glossary {{ background: #6A1B9A; }}

/* 大きなナビゲーションボタン */
.goto-nav {{
    display: flex; gap: 1em; margin: 2em 0;
}}
.goto-btn-lg {{
    flex: 1; padding: 1.2em 1.5em;
    border: none; border-radius: 12px;
    font-size: 16px; font-weight: bold;
    color: #fff; cursor: pointer;
    text-align: left; transition: transform 0.15s, box-shadow 0.15s;
}}
.goto-btn-lg:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}}
.goto-btn-lg .goto-sub {{
    display: block; font-size: 12px; font-weight: normal;
    margin-top: 4px; opacity: 0.85;
}}

/* カスタマイズセクション */
.customize-section {{
    margin: 1em 0; padding: 1em 1.2em;
    background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
}}
.customize-section h4 {{ margin: 0 0 0.6em; color: var(--primary); }}
.glossary-cat {{
    display: inline-block;
    background: #ECEFF1; color: #546E7A;
    font-size: 0.82em; font-weight: bold;
    padding: 2px 10px; border-radius: 4px;
    letter-spacing: 0.05em;
}}
.glossary-full-cat {{
    margin-bottom: 1.2em;
}}
.glossary-dl {{
    margin: 0.4em 0 0 0.5em;
    font-size: 0.95em;
}}
.glossary-dl dt {{
    font-weight: bold; color: var(--primary);
    margin-top: 0.6em;
}}
.glossary-dl dd {{
    margin: 0.15em 0 0 1.2em;
    color: #444; line-height: 1.6;
}}
.customize-list li {{
    margin: 0.7em 0; line-height: 1.7;
}}
.code-block {{
    background: #263238; color: #ECEFF1; padding: 1em 1.2em;
    border-radius: 6px; font-size: 13px; line-height: 1.6;
    overflow-x: auto; white-space: pre;
}}

/* データテーブル */
.data-table {{
    border-collapse: collapse; margin: 1em 0; width: auto;
}}
.data-table th, .data-table td {{
    border: 1px solid #ddd; padding: 8px 16px; text-align: left;
}}
.data-table th {{
    background: #E8EAF6; font-weight: bold;
}}
.data-table tr:nth-child(even) {{ background: #f9f9f9; }}
.toggle-link {{
    display: inline-block; margin: 4px 0 0.8em;
    color: var(--kw-color); font-size: 0.9em; cursor: pointer;
}}
.result-table td:last-child {{ font-size: 0.95em; }}

/* Loss カーブ */
.loss-chart {{
    margin: 1em 0 2em; max-width: 800px;
}}
.loss-chart-bars {{
    display: flex; align-items: flex-end; gap: 2px;
    height: 150px; border-bottom: 2px solid #999;
    border-left: 2px solid #999; padding: 0 4px;
}}
.loss-bar {{
    flex: 1; background: linear-gradient(to top, #1565C0, #42A5F5);
    border-radius: 2px 2px 0 0; min-width: 3px;
    transition: background 0.2s;
}}
.loss-bar:hover {{
    background: linear-gradient(to top, #E65100, #FF8A65);
}}
.loss-chart-labels {{
    display: flex; justify-content: space-between;
    font-size: 0.85em; color: #888; margin-top: 4px;
}}

/* アニメーションコントロール */
.anim-controls {{
    display: flex; align-items: center; gap: 12px;
    margin: 1em 0; padding: 12px 16px;
    background: #E8EAF6; border-radius: 8px;
}}
.anim-btn {{
    padding: 8px 20px; border: none; border-radius: 6px;
    background: var(--primary); color: #fff; font-size: 14px;
    font-weight: bold; cursor: pointer; white-space: nowrap;
}}
.anim-btn:hover {{ opacity: 0.85; }}
.anim-btn-step {{ background: #546E7A; }}
.anim-slider {{
    flex: 1; height: 6px; -webkit-appearance: none; appearance: none;
    background: #C5CAE9; border-radius: 3px; outline: none;
}}
.anim-slider::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: var(--primary); cursor: pointer;
}}
.anim-epoch {{
    font-weight: bold; font-family: monospace; min-width: 100px;
    color: var(--primary);
}}

/* 2×2 グリッド */
.anim-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 1em 0;
}}
.anim-panel {{
    padding: 1em; background: #fff;
    border: 1px solid #e0e0e0; border-radius: 8px;
}}
.anim-panel h4 {{ margin: 0 0 0.6em; color: var(--primary); font-size: 0.95em; }}
.anim-loss-container {{
    display: flex; align-items: center; gap: 12px;
}}
.anim-loss-bar-bg {{
    flex: 1; height: 24px; background: #eee; border-radius: 4px; overflow: hidden;
}}
.anim-loss-fill {{
    height: 100%; background: linear-gradient(90deg, #43A047, #FFA726, #E53935);
    border-radius: 4px; transition: width 0.3s ease;
}}
.anim-loss-val {{
    font-family: monospace; font-size: 15px; font-weight: bold;
    min-width: 55px;
}}

/* Loss ドットチャート */
.anim-loss-chart {{
    position: relative; height: 70px;
    border-bottom: 2px solid #ccc; border-left: 2px solid #ccc;
}}
.loss-dot {{
    position: absolute; width: 8px; height: 8px;
    background: #C5CAE9; border-radius: 50%;
    transform: translate(-50%, 50%);
    transition: background 0.3s;
}}
.loss-dot.active {{ background: #1565C0; }}
.loss-dot.current {{ background: #E65100; width: 12px; height: 12px; box-shadow: 0 0 6px #E65100; }}

/* 収束ライン */
.convergence-line {{
    position: absolute; top: 0; bottom: 0; width: 0;
    border-left: 2px dashed #E65100;
    z-index: 1; pointer-events: none;
    animation: convFadeIn 0.6s ease-out forwards;
}}
.convergence-label {{
    position: absolute; top: -18px;
    transform: translateX(-50%);
    font-size: 0.7em; color: #E65100; font-weight: bold;
    white-space: nowrap; pointer-events: none; z-index: 2;
    animation: convFadeIn 0.6s ease-out forwards;
}}
@keyframes convFadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 0.85; }}
}}

/* 予測バー */
.anim-predictions {{ max-width: 500px; margin-right: 100px; }}
.pred-row {{ display: flex; align-items: center; margin: 5px 0; position: relative; }}
.pred-label {{ width: 70px; font-weight: bold; text-align: right; padding-right: 8px; }}
.pred-track {{ flex: 1; height: 24px; background: #eee; border-radius: 4px; overflow: hidden; }}
.pred-fill {{ height: 100%; background: linear-gradient(90deg, #FF8A65, #E65100); border-radius: 4px; transition: width 0.3s ease; }}
.pred-pct {{ width: 55px; padding-left: 8px; font-family: monospace; font-size: 13px; }}
.pred-selected {{ background: #FFF3E0; border-radius: 6px; }}
.pred-selected .pred-label {{ color: var(--accent); }}
.pred-fill-selected {{ background: linear-gradient(90deg, #FF6D00, #FF9100) !important; }}
.pred-badge {{
    font-size: 11px; font-weight: bold; color: #fff;
    background: var(--accent); border-radius: 4px;
    padding: 1px 8px; white-space: nowrap;
    position: absolute; right: -80px; top: 50%; transform: translateY(-50%);
}}
.pred-answer {{
    margin-top: 0.8em; padding: 8px 12px;
    background: #E8EAF6; border-radius: 6px;
    font-size: 1.1em; color: var(--primary);
}}

/* クエリタブ */
.anim-query-tabs {{ margin-bottom: 0.8em; }}
.qtab {{
    padding: 6px 14px; border: 1px solid #ccc; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 13px; margin: 2px;
}}
.qtab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); font-weight: bold; }}

/* Attention アニメーション */
.anim-attention {{ display: flex; gap: 1.5em; flex-wrap: wrap; }}
.attn-head {{ }}
.attn-title {{ font-size: 13px; font-weight: bold; color: #555; margin-bottom: 4px; }}
.attn-grid {{
    display: grid;
    grid-template-columns: repeat(var(--size), 28px);
    gap: 1px;
}}
.attn-cell {{
    width: 28px; height: 28px; background: #E65100;
    border-radius: 2px; transition: opacity 0.3s ease;
}}
.attn-cell-diff {{ background: #7B1FA2; }}
.attn-title-diff {{ color: #7B1FA2; }}

/* 推論クエリセレクター */
.infer-query-selector {{
    display: flex; flex-wrap: wrap; gap: 6px; margin: 1em 0;
}}
.infer-qbtn {{
    padding: 8px 16px; border: 2px solid #ccc; border-radius: 8px;
    background: #fff; cursor: pointer; font-size: 13px; font-weight: bold;
    transition: all 0.15s;
}}
.infer-qbtn:hover {{ border-color: var(--accent); }}
.infer-qbtn.active {{
    background: var(--accent); color: #fff;
    border-color: var(--accent);
}}

/* 推論アニメーション */
.infer-anim-layout {{
    display: grid; grid-template-columns: 200px 1fr; gap: 1.5em;
    margin: 1em 0;
}}
.infer-pipeline {{
    padding: 0.5em 0;
}}
.infer-tokens {{
    margin-bottom: 1em; font-size: 0.85em; line-height: 2;
}}
.pipe-step {{
    padding: 6px 12px; border-radius: 6px;
    background: #f0f0f0; color: #888; font-size: 0.85em;
    margin: 2px 0; text-align: center;
    transition: all 0.3s; cursor: pointer;
}}
.pipe-step:hover {{
    background: #E8EAF6; color: var(--primary);
}}
.pipe-step.pipe-active {{
    background: var(--primary); color: #fff; font-weight: bold;
    box-shadow: 0 2px 8px rgba(26,35,126,0.3);
}}
.pipe-step.pipe-done {{
    background: #C5CAE9; color: #333;
}}
.pipe-arrow {{
    text-align: center; color: #ccc; font-size: 0.9em; line-height: 1.2;
}}
.infer-viz {{
    min-height: 300px;
}}
.infer-step-desc {{
    color: #666; margin: 0.5em 0 1em; font-size: 0.95em;
}}
.infer-content {{
    overflow-x: auto;
}}
/* Temperature コントロール */
.temp-section {{ margin-top: 0.5em; }}
.temp-control {{
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 1em; padding: 10px 16px;
    background: #FFF8E1; border: 1px solid #FFB74D;
    border-radius: 8px; flex-wrap: wrap;
}}
.temp-slider {{
    width: 200px; vertical-align: middle;
}}
.temp-val {{
    font-family: monospace; font-weight: bold; font-size: 1.1em;
    color: var(--accent); min-width: 30px; display: inline-block;
}}
.temp-hint {{
    font-size: 0.8em; color: #999;
}}

/* Greedy/Sampling 切替 */
.decode-mode {{
    display: flex; gap: 6px; margin-bottom: 8px;
}}
.mode-btn {{
    padding: 6px 16px; border: 2px solid #ccc; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 12px;
    transition: all 0.15s;
}}
.mode-btn:hover {{ border-color: var(--primary); }}
.mode-btn.mode-active {{
    background: var(--primary); color: #fff;
    border-color: var(--primary); font-weight: bold;
}}

/* サンプリングボタン */
.sample-controls {{
    display: flex; align-items: center; gap: 10px;
    margin: 8px 0;
}}
.sample-btn {{
    padding: 8px 20px; border: none; border-radius: 6px;
    background: #7B1FA2; color: #fff; font-size: 14px;
    font-weight: bold; cursor: pointer;
    transition: transform 0.1s;
}}
.sample-btn:hover {{ opacity: 0.85; }}
.sample-btn:active {{ transform: scale(0.95); }}
.sample-hint {{
    font-size: 0.8em; color: #888;
}}

.layer-combined {{
    display: flex; flex-direction: column; gap: 1.2em;
}}
.layer-half h5 {{
    margin: 0 0 0.4em; color: #555; font-size: 0.9em;
    border-bottom: 1px solid #eee; padding-bottom: 4px;
}}

/* ナビゲーション */
.nav-buttons {{
    display: flex; justify-content: space-between;
    margin-top: 2em; padding-top: 1em;
    border-top: 1px solid #ddd;
}}
.nav-btn {{
    padding: 10px 24px; border: none; border-radius: 8px;
    font-size: 15px; cursor: pointer; font-weight: bold;
}}
.nav-prev {{ background: #E8EAF6; color: var(--primary); }}
.nav-next {{ background: var(--primary); color: #fff; }}
.nav-btn:hover {{ opacity: 0.85; }}
.nav-btn:disabled {{ opacity: 0.3; cursor: default; }}

h3 {{ color: var(--primary); margin: 1.5em 0 0.5em; font-size: 1.2em; }}
h4 {{ color: #555; margin: 0.5em 0; font-size: 0.95em; }}
ul, ol {{ margin: 0.5em 0 1em 1.5em; }}
li {{ margin: 0.3em 0; }}
</style>
</head>
<body>
<div class="container">
    <div class="tab-bar" id="tab-bar">
        {tab_buttons}
    </div>
    {tab_contents}
</div>

<script>
// ===== 全テキストノードに GLOSSARY キーワードを自動ハイライト =====
(function() {{
    const G = {glossary_json};
    // 長い順にソート（部分一致防止）
    const terms = Object.keys(G).sort((a, b) => b.length - a.length);
    // 正規表現を構築（特殊文字エスケープ）
    const escaped = terms.map(t => t.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'));
    if (escaped.length === 0) return;
    const re = new RegExp('(' + escaped.join('|') + ')', 'g');

    // テキストノードを走査してキーワードを <span class="kw"> でラップ
    function walkNode(node) {{
        if (node.nodeType === 3) {{ // テキストノード
            const text = node.textContent;
            if (!re.test(text)) return;
            re.lastIndex = 0;
            const frag = document.createDocumentFragment();
            let lastIdx = 0;
            let match;
            re.lastIndex = 0;
            while ((match = re.exec(text)) !== null) {{
                // マッチ前のテキスト
                if (match.index > lastIdx) {{
                    frag.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
                }}
                // 既に .kw の中にいるか確認（親をチェック）
                let alreadyWrapped = false;
                let p = node.parentNode;
                while (p) {{
                    if (p.classList && p.classList.contains('kw')) {{ alreadyWrapped = true; break; }}
                    if (p.classList && p.classList.contains('tip')) {{ alreadyWrapped = true; break; }}
                    p = p.parentNode;
                }}
                if (alreadyWrapped) {{
                    frag.appendChild(document.createTextNode(match[0]));
                }} else {{
                    const span = document.createElement('span');
                    span.className = 'kw';
                    span.dataset.tip = G[match[0]];
                    span.textContent = match[0];
                    frag.appendChild(span);
                }}
                lastIdx = re.lastIndex;
            }}
            if (lastIdx < text.length) {{
                frag.appendChild(document.createTextNode(text.slice(lastIdx)));
            }}
            node.parentNode.replaceChild(frag, node);
        }} else if (node.nodeType === 1) {{ // 要素ノード
            // script, style, 既存の .kw, .tip, code, pre はスキップ
            const tag = node.tagName.toLowerCase();
            if (tag === 'script' || tag === 'style' || tag === 'code' || tag === 'pre') return;
            if (node.classList && (node.classList.contains('kw') || node.classList.contains('tip'))) return;
            // 子ノードを逆順でコピーして走査（DOM 変更に対応）
            const children = Array.from(node.childNodes);
            children.forEach(walkNode);
        }}
    }}
    walkNode(document.body);
}})();

// ツールチップ: クリックで固定 + コピーボタン付き
document.querySelectorAll('.kw[data-tip]').forEach(el => {{
    const keyword = el.textContent;
    const tipText = el.dataset.tip;

    const tip = document.createElement('span');
    tip.className = 'tip';

    // ヘッダ（タイトル + コピー + 閉じる）
    const header = document.createElement('div');
    header.className = 'tip-header';
    const title = document.createElement('span');
    title.className = 'tip-title';
    title.textContent = '【' + keyword + '】';
    header.appendChild(title);

    const btnGroup = document.createElement('span');
    const copyBtn = document.createElement('button');
    copyBtn.className = 'tip-copy-btn';
    copyBtn.textContent = '📋 コピー';
    copyBtn.onclick = function(e) {{
        e.stopPropagation();
        navigator.clipboard.writeText(keyword + ': ' + tipText).then(() => {{
            copyBtn.textContent = '✓ コピー済み';
            setTimeout(() => copyBtn.textContent = '📋 コピー', 1500);
        }});
    }};
    const closeBtn = document.createElement('button');
    closeBtn.className = 'tip-close-btn';
    closeBtn.textContent = '✕';
    closeBtn.onclick = function(e) {{
        e.stopPropagation();
        tip.classList.remove('pinned');
    }};
    btnGroup.appendChild(copyBtn);
    btnGroup.appendChild(closeBtn);
    header.appendChild(btnGroup);

    const body = document.createElement('div');
    body.textContent = tipText;

    tip.appendChild(header);
    tip.appendChild(body);
    el.appendChild(tip);

    // クリックで固定/解除
    el.addEventListener('click', function(e) {{
        e.stopPropagation();
        // 他の固定を解除
        document.querySelectorAll('.tip.pinned').forEach(t => {{
            if (t !== tip) t.classList.remove('pinned');
        }});
        tip.classList.toggle('pinned');
    }});
}});

// ページ上のどこかをクリックしたら固定解除
document.addEventListener('click', function() {{
    document.querySelectorAll('.tip.pinned').forEach(t => t.classList.remove('pinned'));
}});

function toggleTrainData(e) {{
    e.preventDefault();
    const hidden = document.getElementById('train-data-hidden');
    const link = document.getElementById('train-data-toggle');
    if (hidden.style.display === 'none') {{
        hidden.style.display = '';
        link.textContent = '省略';
    }} else {{
        hidden.style.display = 'none';
        link.textContent = 'すべて表示 (' + (hidden.querySelectorAll('tr').length + 4) + '件)';
    }}
}}

function showTab(idx) {{
    document.querySelectorAll('.tab-content').forEach((el, i) => {{
        el.style.display = i === idx ? 'block' : 'none';
    }});
    document.querySelectorAll('.tab-btn').forEach((el, i) => {{
        el.classList.toggle('active', i === idx);
    }});
    window.scrollTo(0, 0);
}}

// キーボードナビゲーション
document.addEventListener('keydown', (e) => {{
    const btns = document.querySelectorAll('.tab-btn');
    const current = [...btns].findIndex(b => b.classList.contains('active'));
    if (e.key === 'ArrowRight' && current < btns.length - 1) showTab(current + 1);
    if (e.key === 'ArrowLeft' && current > 0) showTab(current - 1);
}});
</script>
</body>
</html>'''
