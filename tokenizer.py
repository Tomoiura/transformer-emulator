"""日本語トークナイザ（fugashi/MeCab ベース）"""

import numpy as np

try:
    import fugashi
    _FUGASHI_AVAILABLE = True
except ImportError:
    _FUGASHI_AVAILABLE = False


class JapaneseTokenizer:
    """形態素解析ベースの日本語トークナイザ。

    特殊トークン:
      - <PAD> (id=0): パディング
      - <BOS> (id=1): 文頭
      - <EOS> (id=2): 文末
      - <UNK> (id=3): 未知語
    """

    SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __init__(self, max_tokens: int = 16):
        self.max_tokens = max_tokens
        # 語彙: 特殊トークン + 動的に追加
        self.token2id: dict = dict(self.SPECIAL_TOKENS)
        self.id2token: dict = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._next_id = len(self.SPECIAL_TOKENS)

        if _FUGASHI_AVAILABLE:
            self.tagger = fugashi.Tagger()
        else:
            self.tagger = None
            print("[WARN] fugashi が見つかりません。文字単位トークン化にフォールバックします。")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def _morpheme_tokenize(self, text: str) -> list[str]:
        """形態素解析でトークン化"""
        if self.tagger is not None:
            return [word.surface for word in self.tagger(text)]
        # フォールバック: 文字単位
        return list(text)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = self._next_id
            self.id2token[self._next_id] = token
            self._next_id += 1
        return self.token2id[token]

    def tokenize(self, text: str) -> dict:
        """テキストをトークン化し、情報を返す。

        Returns:
            dict with keys:
                tokens: トークン文字列のリスト（BOS付き）
                token_ids: トークンIDのリスト
                vocab: 現在の語彙辞書
        """
        morphemes = self._morpheme_tokenize(text)

        # BOS を先頭に付与（Decoder モデルの慣例）
        tokens = ["<BOS>"] + morphemes
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]

        # 語彙に登録 & ID 変換
        token_ids = []
        for t in tokens:
            tid = self._add_token(t)
            token_ids.append(tid)

        return {
            "tokens": tokens,
            "token_ids": np.array(token_ids, dtype=np.int32),
            "vocab": dict(self.token2id),
        }

    def decode_ids(self, ids) -> list[str]:
        """トークンIDリストを文字列リストに変換"""
        return [self.id2token.get(int(i), "<UNK>") for i in ids]

    def print_info(self, result: dict):
        """トークン化結果を表示"""
        print("=" * 60)
        print("TOKENIZATION")
        print("=" * 60)
        print(f"  トークン数: {len(result['tokens'])}")
        print(f"  語彙サイズ: {len(result['vocab'])}")
        print()
        print("  ID | トークン")
        print("  ---+----------")
        for token, tid in zip(result["tokens"], result["token_ids"]):
            print(f"  {tid:>3} | {token}")
        print()
