"""IB (Information-Bottleneck) Reranker —— LLM 二分类相关性过滤。

基于 opt-v4-plus-hyde 的 llm_rerank_contexts 思路：
  对前 max_rerank 条召回段用 SYS_CTX_RERANK 判断相关/无关，过滤无关段。
  推理失败时保留（避免误删）。

接口与 DashVectorRetriever.search 完全一致。
"""
from __future__ import annotations

from typing import List, Optional


class IBRerankRetriever:
    def __init__(
        self,
        inner,
        client,
        max_rerank: int = 4,
        keep_min: int = 2,
        max_tokens: int = 8,
    ):
        self.inner = inner
        self.client = client
        self.max_rerank = max_rerank
        self.keep_min = keep_min
        self.max_tokens = max_tokens

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        from .prompts import SYS_CTX_RERANK

        contexts = self.inner.search(text=text, image_path=image_path, topk=topk)
        if not contexts or len(contexts) <= self.keep_min:
            return contexts

        candidates = contexts[: self.max_rerank]
        relevant: List[str] = []
        for ctx in candidates:
            user = f"[题目] {text}\n[资料片段] {ctx[:300]}\n判断是否相关："
            try:
                r = self.client.generate(
                    SYS_CTX_RERANK, user, temperature=0.0, max_tokens=self.max_tokens,
                )
                rs = (r or "").strip()
                is_relevant = "相关" in rs and "无关" not in rs
            except Exception:
                is_relevant = True
            if is_relevant:
                relevant.append(ctx)

        remaining = contexts[self.max_rerank:]
        result = relevant + remaining
        if len(result) < self.keep_min:
            return contexts[: self.keep_min]
        return result
