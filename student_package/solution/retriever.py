"""
检索器: 借鉴 chuangzhi-summer-camp 的 TF-IDF cosine + BM25 RRF 融合。

由于本项目没有外部知识库的本地副本, 用例间相互检索作为 "类似历史问答" 上下文。
为避免训练泄漏, 检索时排除查询自身的 index。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _featurize(text: str) -> list:
    s = (text or "").lower()
    feats = list(_WORD_RE.findall(s))
    compact = re.sub(r"\s+", "", s)
    for n in (2, 3):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                feats.append(f"#{n}#{compact[i:i + n]}")
    return feats


class HybridRetriever:
    def __init__(self):
        self._docs: List[dict] = []
        self._doc_tf: List[Counter] = []
        self._doc_len: List[int] = []
        self._inverted: dict = defaultdict(list)
        self._df: Counter = Counter()
        self._N = 0
        self._total_len = 0
        self._doc_norm_cache: List[float] = []

    def add(self, doc: dict, text: str) -> None:
        ctr = Counter(_featurize(text))
        idx = self._N
        self._N += 1
        self._docs.append(doc)
        self._doc_tf.append(ctr)
        dl = sum(ctr.values())
        self._doc_len.append(dl)
        self._total_len += dl
        self._doc_norm_cache.append(0.0)
        for t in ctr:
            self._inverted[t].append(idx)
            self._df[t] += 1

    def _idf(self, t: str) -> float:
        df = self._df.get(t, 0)
        if df == 0:
            return 0.0
        return math.log((self._N + 1.0) / (df + 0.5))

    def _doc_norm(self, idx: int) -> float:
        n = self._doc_norm_cache[idx]
        if n > 0:
            return n
        s = 0.0
        for t, c in self._doc_tf[idx].items():
            w = c * self._idf(t)
            s += w * w
        n = math.sqrt(s) or 1.0
        self._doc_norm_cache[idx] = n
        return n

    def _tfidf(self, text: str, k: int) -> List[Tuple[float, int]]:
        if self._N == 0:
            return []
        q_ctr = Counter(_featurize(text))
        q_vec = {t: c * self._idf(t) for t, c in q_ctr.items() if self._df.get(t, 0) > 0}
        if not q_vec:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scores: dict = {}
        for t, qv in q_vec.items():
            wt = self._idf(t)
            for d in self._inverted.get(t, ()):
                dc = self._doc_tf[d].get(t, 0)
                if dc:
                    scores[d] = scores.get(d, 0.0) + qv * dc * wt
        ranked = [(dot / (q_norm * self._doc_norm(d)), d) for d, dot in scores.items()]
        ranked.sort(reverse=True)
        return ranked[:k]

    def _bm25(self, text: str, k: int, k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, int]]:
        if self._N == 0:
            return []
        avgdl = (self._total_len / self._N) or 1.0
        q_ctr = Counter(_featurize(text))
        q_tokens = [t for t in q_ctr if self._df.get(t, 0) > 0]
        if not q_tokens:
            return []
        scores: dict = {}
        for t in q_tokens:
            idf = self._idf(t)
            if idf <= 0:
                continue
            for d in self._inverted.get(t, ()):
                f = self._doc_tf[d].get(t, 0)
                if not f:
                    continue
                dl = self._doc_len[d] or 1
                denom = f + k1 * (1.0 - b + b * dl / avgdl)
                scores[d] = scores.get(d, 0.0) + idf * f * (k1 + 1.0) / denom
        ranked = [(s, d) for d, s in scores.items()]
        ranked.sort(reverse=True)
        return ranked[:k]

    def search(self, text: str, k: int = 5, exclude_index: int | None = None) -> List[dict]:
        """RRF 融合 TF-IDF cosine + BM25, 返回 top-k 文档(原 dict)。"""
        if self._N == 0:
            return []
        M = max(k * 3, 30)
        tfidf = self._tfidf(text, M)
        bm25 = self._bm25(text, M)
        if not tfidf and not bm25:
            return []
        RRF_K = 60.0
        rrf: dict = {}
        for rank, (_, d) in enumerate(tfidf):
            rrf[d] = rrf.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (_, d) in enumerate(bm25):
            rrf[d] = rrf.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: -x[1])
        out: List[dict] = []
        for d, _s in ranked:
            doc = self._docs[d]
            if exclude_index is not None and doc.get("index") == exclude_index:
                continue
            out.append(doc)
            if len(out) >= k:
                break
        return out
