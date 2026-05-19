"""读 bm25_corpus.jsonl → jieba 分词 → BM25Okapi → pickle 落盘。

用法：
    python scripts/build_bm25_index.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

CORPUS_PATH = PKG / "data" / "bm25_corpus.jsonl"
INDEX_PATH = PKG / "solution" / "bm25_index.pkl"


def main():
    import jieba
    from rank_bm25 import BM25Okapi

    if not CORPUS_PATH.exists():
        print(f"[idx] corpus not found at {CORPUS_PATH}", flush=True)
        sys.exit(1)

    stopwords = _STOPWORDS

    print(f"[idx] reading {CORPUS_PATH}", flush=True)
    corpus_texts: list = []
    corpus_ids: list = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = (rec.get("text") or "").strip()
            if not t:
                continue
            corpus_texts.append(t)
            corpus_ids.append(rec.get("id") or "")
    print(f"[idx] {len(corpus_texts)} passages loaded", flush=True)

    jieba.initialize()
    print("[idx] tokenizing...", flush=True)
    t0 = time.perf_counter()
    tokenized: list = []
    for i, txt in enumerate(corpus_texts, 1):
        toks = [
            t.strip()
            for t in jieba.lcut(txt)
            if t.strip() and t.strip() not in stopwords
        ]
        tokenized.append(toks)
        if i % 2000 == 0:
            print(f"[idx]   tokenized {i}/{len(corpus_texts)}", flush=True)
    print(f"[idx] tokenize done in {time.perf_counter()-t0:.1f}s", flush=True)

    print("[idx] building BM25Okapi...", flush=True)
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    print(f"[idx] BM25 built in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[idx] writing pickle → {INDEX_PATH}", flush=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "corpus_texts": corpus_texts,
                "corpus_ids": corpus_ids,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"[idx] DONE: {len(corpus_texts)} passages, {size_mb:.1f} MB", flush=True)


_STOPWORDS = frozenset(
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
        "，", "。", "、", "：", "；", "？", "！", "（", "）",
        "“", "”", "‘", "’",
        ",", ".", ":", ";", "?", "!", "(", ")", "\"", "'",
        "图", "图中", "图片", "如图", "所示", "答案", "问题", "题目",
    ]
)


if __name__ == "__main__":
    main()
