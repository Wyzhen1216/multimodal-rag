"""RRF 融合包装器 —— 把 DashVectorRetriever 和 BM25Retriever 融合成一个检索器。

接口与 DashVectorRetriever.search 完全一致，调用方零改动。
"""
from __future__ import annotations

import concurrent.futures
from typing import List, Optional


class HybridRetriever:
    def __init__(
        self,
        vector_ret,
        bm25_ret,
        k: int = 60,
        vec_weight: float = 1.0,
        bm25_weight: float = 1.0,
        topk: int = 5,
    ):
        self.vector_ret = vector_ret
        self.bm25_ret = bm25_ret
        self.k = k
        self.vec_weight = vec_weight
        self.bm25_weight = bm25_weight
        self.topk = topk

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        k = topk or self.topk
        fetch_k = max(k * 2, 10)

        def _vec():
            try:
                return self.vector_ret.search(
                    text=text, image_path=image_path, topk=fetch_k
                )
            except Exception:
                return []

        def _bm25():
            try:
                return self.bm25_ret.search(text=text, topk=fetch_k)
            except Exception:
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_vec = pool.submit(_vec)
            f_bm25 = pool.submit(_bm25)
            vec_docs = f_vec.result()
            bm25_docs = f_bm25.result()

        if not vec_docs and not bm25_docs:
            return []
        if not bm25_docs:
            return vec_docs[:k]
        if not vec_docs:
            return bm25_docs[:k]

        scores: dict = {}
        for rank, doc in enumerate(vec_docs, start=1):
            scores[doc] = scores.get(doc, 0.0) + self.vec_weight / (self.k + rank)
        for rank, doc in enumerate(bm25_docs, start=1):
            scores[doc] = scores.get(doc, 0.0) + self.bm25_weight / (self.k + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]
