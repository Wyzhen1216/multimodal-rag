"""HyDE 检索包装器 —— 先用 LLM 生成假设背景，再拼到 query 检索。

HyDE = Hypothetical Document Embeddings：
  原 query "图中所示建筑名称" → 单独检索容易召回到泛化段落
  HyDE: 先让 LLM 答"考古/历史背景知识"，把回答拼到 query
  → 检索 embedding 更接近文档分布，召回更精准

接口与 DashVectorRetriever.search 完全一致，调用方零改动。
"""
from __future__ import annotations

import os
from typing import List, Optional


_HYDE_SYS = "你是考古/历史专家。围绕题目给出 1-2 句相关背景知识陈述句，用于文献检索。直接输出陈述句，不要前缀，不超过 60 字。"


class HyDERetriever:
    def __init__(self, inner, client, max_tokens: int = 60, prefix: bool = False):
        self.inner = inner
        self.client = client
        self.max_tokens = max_tokens
        self.prefix = prefix  # True: hyde 放前；False: 原 query 放前

    def _hyde(self, text: str) -> str:
        if not text:
            return ""
        try:
            r = self.client.generate(
                _HYDE_SYS, text, temperature=0.0, max_tokens=self.max_tokens,
            )
        except Exception:
            return ""
        s = (r or "").strip().splitlines()
        return (s[0] if s else "").strip()

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        hyde = self._hyde(text)
        if hyde:
            q = f"{hyde} {text}" if self.prefix else f"{text} {hyde}"
        else:
            q = text
        return self.inner.search(text=q, image_path=image_path, topk=topk)
