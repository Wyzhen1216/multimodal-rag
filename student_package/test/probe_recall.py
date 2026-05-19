"""探查错题的检索召回情况：dump 各 query 的 top-k 命中文本。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

os.environ.setdefault("QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY")
os.environ.setdefault("QWEN_MODEL_NAME", "qwen2.5-vl-7b-instruct")
os.environ.setdefault(
    "DASHVECTOR_ENDPOINT",
    "vrs-cn-71j49hani00061.dashvector.cn-hangzhou.aliyuncs.com",
)
os.environ.setdefault(
    "DASHVECTOR_API_KEY",
    "sk-REPLACE-WITH-YOUR-OWN-VECTOR-KEY",
)
os.environ.setdefault("DASHVECTOR_COLLECTION", "ccv_cb_student_20260508")
os.environ.setdefault("DASHVECTOR_VECTOR_FIELD", "text")
os.environ.setdefault("GME_MODEL_PATH", "/autodl-fs/data/gme-Qwen2-VL-7B-Instruct")
os.environ.setdefault("DASHSCOPE_API_KEY", os.environ["QWEN_API_KEY"])

from solution.retriever import DashVectorRetriever  # noqa: E402
from solution.main import (  # noqa: E402
    build_query, parse_choice_options, diverge_queries, rewrite_query,
)
from solution.llm_client import build_client  # noqa: E402

GOLD_PATH = PKG / "data" / "public_examples.full.json"

# 选择题错题 + 简答题错题
TARGET_IDX = {358, 504, 609, 630, 689, 5344, 5462, 2211, 2377, 2467}


def fmt_doc(d: str, n: int = 220) -> str:
    s = (d or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def main() -> int:
    items = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    targets = [r for r in items if int(r.get("index", -1)) in TARGET_IDX]
    print(f"loaded {len(targets)} target items")

    retriever = DashVectorRetriever(topk=5)
    client = build_client(prefer="api")

    out = []
    for r in targets:
        idx = r["index"]
        sec = r.get("section", "")
        q = r.get("question", "")
        gold = r.get("answer", "")
        print("\n" + "=" * 80)
        print(f"[idx={idx}] sec={sec}")
        print(f"  question: {q}")
        print(f"  gold:     {gold!r}")

        block = {"index": idx, "section": sec, "question": q, "gold": gold, "queries": []}

        if "选择" in sec:
            options = parse_choice_options(q)
            stem = build_query(q, section=sec)
            print(f"  stem(query base): {stem}")
            for letter, opt_text in options:
                qstr = f"{stem[:50]} {opt_text}".strip()
                docs = retriever.search(text=qstr, topk=5)
                print(f"  -- option {letter}. {opt_text}  query={qstr!r}")
                for i, d in enumerate(docs, 1):
                    print(f"     [{i}] {fmt_doc(d)}")
                block["queries"].append({"letter": letter, "query": qstr, "docs": docs})
        elif "简答" in sec and "VQA" not in sec:
            # 走 diverge_queries 同款流程
            try:
                qs = diverge_queries(client, q)
            except Exception as e:
                qs = [q]
                print(f"  diverge_queries failed: {e}")
            print(f"  diverged queries: {qs}")
            seen = set()
            merged = []
            for qstr in qs:
                docs = retriever.search(text=qstr, topk=5)
                print(f"  -- query: {qstr!r}")
                for i, d in enumerate(docs, 1):
                    print(f"     [{i}] {fmt_doc(d)}")
                block["queries"].append({"query": qstr, "docs": docs})
                for d in docs:
                    if d not in seen:
                        seen.add(d)
                        merged.append(d)
            print(f"  merged unique top-{len(merged)}")
        else:
            qstr = q
            docs = retriever.search(text=qstr, topk=5)
            print(f"  -- query: {qstr!r}")
            for i, d in enumerate(docs, 1):
                print(f"     [{i}] {fmt_doc(d)}")
            block["queries"].append({"query": qstr, "docs": docs})

        out.append(block)

    out_path = HERE / "probe_recall_output.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
