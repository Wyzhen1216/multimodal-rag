"""BM25 稀疏检索器 —— 与 DashVectorRetriever 同接口。

设计要点：
- lazy load：首次 search 才反序列化 pickle（避免 import 时占内存）
- silent disable：索引文件缺失/反序列化失败 → search 返回 []，对齐现有失败语义
- 线程安全：rank_bm25 是 read-only 查询，无需锁
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

_DEFAULT_INDEX = Path(__file__).resolve().parent / "bm25_index.pkl"


class BM25Retriever:
    def __init__(self, index_path: Optional[str] = None, topk: int = 5):
        self.index_path = Path(
            index_path or os.environ.get("BM25_INDEX_PATH", str(_DEFAULT_INDEX))
        )
        self.topk = topk
        self._bm25 = None
        self._corpus_texts: List[str] = []
        self._loaded = False
        self._disabled = False
        self._jieba = None
        self._stopwords = None

    def _ensure_loaded(self) -> None:
        if self._loaded or self._disabled:
            return
        if not self.index_path.exists():
            print(
                f"[bm25] index not found at {self.index_path}, disabling BM25",
                flush=True,
            )
            self._disabled = True
            return
        try:
            with open(self.index_path, "rb") as f:
                obj = pickle.load(f)
            self._bm25 = obj["bm25"]
            self._corpus_texts = obj["corpus_texts"]
            print(
                f"[bm25] loaded index: {len(self._corpus_texts)} passages "
                f"from {self.index_path}",
                flush=True,
            )
        except Exception as e:
            print(f"[bm25] load failed: {e}; disabling BM25", flush=True)
            self._disabled = True
            return

        try:
            import jieba

            jieba.initialize()
            self._jieba = jieba
            self._stopwords = _DEFAULT_STOPWORDS
        except Exception as e:
            print(f"[bm25] jieba init failed: {e}; disabling BM25", flush=True)
            self._disabled = True
            return

        self._loaded = True

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = [
            t.strip()
            for t in self._jieba.lcut(text)
            if t.strip() and t.strip() not in self._stopwords
        ]
        return tokens

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        self._ensure_loaded()
        if self._disabled:
            return []
        k = topk or self.topk
        tokens = self._tokenize(text)
        if not tokens:
            return []
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as e:
            print(f"[bm25] score failed: {e}", flush=True)
            return []
        import numpy as np

        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self._corpus_texts[i] for i in idx if scores[i] > 0]


_DEFAULT_STOPWORDS = frozenset(
    [
        "的", "了", "是", "在", "和", "与", "或", "及", "等",
        "也", "都", "就", "而", "但", "并", "于", "为", "以",
        "对", "从", "向", "到", "由", "至", "之", "其", "此",
        "该", "上", "下", "中", "内", "外", "前", "后", "里",
        "有", "无", "可", "能", "要", "会", "将", "已", "未",
        "请", "什么", "哪个", "哪些", "哪一", "如何", "怎样",
        "是否", "吗", "呢", "啊", "呀", "吧", "哦", "嗯",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        " ", "\t", "\n", "\r",
        "，", "。", "、", "：", "；", "？", "！", "（", "）", "“", "”", "‘", "’",
        ",", ".", ":", ";", "?", "!", "(", ")", "\"", "'",
        "图", "图中", "图片", "如图", "所示", "答案", "问题", "题目",
    ]
)
