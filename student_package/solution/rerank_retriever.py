"""Cross-encoder reranker 包装器 —— 检索后用 cross-encoder 对 top-k 重排。

模型：BAAI/bge-reranker-v2-m3（XLMRoberta，~2.2GB safetensors）
- 输入：(query, passage) pair → 单个分数
- 比 bi-encoder 检索精度高，但慢：每对 ~10-30ms（GPU）
- 适合 retrieve top-20 → rerank top-5 这种 funnel

接口与 DashVectorRetriever.search 完全一致，调用方零改动。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


_DEFAULT_MODEL = "/autodl-fs/data/bge-reranker-v2-m3"


class RerankRetriever:
    def __init__(
        self,
        inner,
        model_path: Optional[str] = None,
        fetch_topk: int = 20,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.inner = inner
        self.model_path = model_path or os.environ.get(
            "RERANK_MODEL_PATH", _DEFAULT_MODEL
        )
        self.fetch_topk = fetch_topk
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._disabled = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._disabled:
            return
        if not Path(self.model_path).exists():
            print(
                f"[rerank] model not found at {self.model_path}, disabling rerank",
                flush=True,
            )
            self._disabled = True
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                .to(self.device)
                .eval()
            )
            print(
                f"[rerank] loaded {Path(self.model_path).name} on {self.device}",
                flush=True,
            )
        except Exception as e:
            print(f"[rerank] load failed: {e}; disabling rerank", flush=True)
            self._disabled = True

    def _score(self, query: str, passages: List[str]) -> List[float]:
        pairs = [[query, p] for p in passages]
        with self._torch.inference_mode():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**inputs).logits.view(-1).float()
        return logits.cpu().numpy().tolist()

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        k = topk or 5
        # 召回扩大：取 max(fetch_topk, k*4) 个候选
        fetch_k = max(self.fetch_topk, k * 4)
        candidates = self.inner.search(text=text, image_path=image_path, topk=fetch_k)
        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        self._ensure_loaded()
        if self._disabled:
            return candidates[:k]

        try:
            scores = self._score(text, candidates)
        except Exception as e:
            print(f"[rerank] score failed: {e}", flush=True)
            return candidates[:k]

        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:k]]
