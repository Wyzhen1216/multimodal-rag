"""自举采集 BM25 corpus：跑 mock + public 所有题目 → DashVector 拿 topk 段落 → 去重落盘。

用法：
    python scripts/build_bm25_corpus.py                    # 全量
    python scripts/build_bm25_corpus.py --limit 20         # 调试，只跑 20 题
    python scripts/build_bm25_corpus.py --resume           # 跳过已采集的 query
    python scripts/build_bm25_corpus.py --topk 30          # 每题拿 30 段
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

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

from solution.retriever import DashVectorRetriever  # noqa: E402
from solution.main import build_query  # noqa: E402

CORPUS_PATH = PKG / "data" / "bm25_corpus.jsonl"
STATE_PATH = PKG / "data" / "bm25_corpus.state.json"

MIN_LEN = 30
MAX_LEN = 2000


def load_all_items() -> list:
    items: list = []
    seen_keys: set = set()

    def _add(rec: dict, source_tag: str):
        idx = rec.get("index")
        q = (rec.get("question") or "").strip()
        sec = rec.get("section", "")
        if not q:
            return
        key = (source_tag, idx, q[:80])
        if key in seen_keys:
            return
        seen_keys.add(key)
        items.append(
            {
                "source": source_tag,
                "index": idx,
                "section": sec,
                "question": q,
            }
        )

    paths = []
    pub = PKG / "data" / "public_examples.full.json"
    if pub.exists():
        paths.append(("public", pub))
    mix = PKG / "data" / "mock" / "history_mixed_qa.json"
    if mix.exists():
        paths.append(("mock_mixed", mix))
    for sub in glob.glob(str(PKG / "data" / "mock" / "*" / "*.json")):
        paths.append(("mock_sub", Path(sub)))

    for tag, p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[corpus] skip {p}: {e}", flush=True)
            continue
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict):
                    _add(r, f"{tag}:{p.stem}")
    return items


def passage_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_state(resume: bool) -> tuple:
    if not resume:
        return {}, set()
    if not STATE_PATH.exists():
        return {}, set()
    try:
        st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        done = set(st.get("done_keys", []))
        passages = st.get("passages", {})
        return passages, done
    except Exception:
        return {}, set()


def save_state(passages: dict, done: set):
    STATE_PATH.write_text(
        json.dumps(
            {"done_keys": sorted(done), "passages": passages},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def dump_corpus(passages: dict):
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for pid, rec in passages.items():
            f.write(
                json.dumps(
                    {
                        "id": pid,
                        "text": rec["text"],
                        "source_queries": rec["source_queries"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试）")
    p.add_argument("--topk", type=int, default=20, help="每题取 topk 段")
    p.add_argument("--resume", action="store_true", help="跳过已采集的 query")
    p.add_argument(
        "--save-every", type=int, default=100, help="每 N 题落盘一次 state"
    )
    args = p.parse_args()

    items = load_all_items()
    print(f"[corpus] loaded {len(items)} questions from mock+public", flush=True)
    if args.limit > 0:
        items = items[: args.limit]
        print(f"[corpus] --limit {args.limit}: only first {len(items)}", flush=True)

    passages, done = load_state(args.resume)
    print(
        f"[corpus] resume state: {len(passages)} passages, {len(done)} queries done",
        flush=True,
    )

    retriever = DashVectorRetriever(topk=args.topk)

    t_start = time.perf_counter()
    n_new_q = 0
    for i, it in enumerate(items, 1):
        key = f'{it["source"]}#{it["index"]}'
        if key in done:
            continue
        q = build_query(it["question"], section=it["section"])
        if not q:
            done.add(key)
            continue
        try:
            docs = retriever.search(text=q, topk=args.topk)
        except Exception as e:
            print(f"[corpus] search failed for {key}: {e}", flush=True)
            docs = []

        for d in docs:
            d = (d or "").strip()
            if not d or len(d) < MIN_LEN or len(d) > MAX_LEN:
                continue
            pid = passage_id(d)
            if pid not in passages:
                passages[pid] = {"text": d, "source_queries": [key]}
            else:
                if key not in passages[pid]["source_queries"]:
                    passages[pid]["source_queries"].append(key)
        done.add(key)
        n_new_q += 1

        if i % 20 == 0 or i == len(items):
            elapsed = time.perf_counter() - t_start
            rate = n_new_q / elapsed if elapsed > 0 else 0
            eta = (len(items) - i) / rate if rate > 0 else 0
            print(
                f"[corpus] {i}/{len(items)} q={key} | "
                f"passages={len(passages)} | "
                f"rate={rate:.2f} q/s | eta={eta:.0f}s",
                flush=True,
            )
        if n_new_q > 0 and n_new_q % args.save_every == 0:
            save_state(passages, done)
            print(f"[corpus] state checkpoint at {n_new_q} new queries", flush=True)

    save_state(passages, done)
    dump_corpus(passages)
    print(
        f"[corpus] DONE: {len(passages)} unique passages, "
        f"corpus written to {CORPUS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
