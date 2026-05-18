"""
solution.main — 多模态历史 RAG Agent 入口。

支持两种策略:
  --strategy baseline  : 无检索, 直接调用 Qwen2.5-VL
  --strategy rag       : 用 TF-IDF + BM25 RRF 融合检索, 取相似历史题作 few-shot 上下文

输出:
  predictions.jsonl
  time_cost.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.normalization import normalize_answer  # noqa: E402

from .llm_client import QwenVLClient
from .prompts import SYSTEM_BASE, build_baseline_user, build_rag_user
from .retriever import HybridRetriever


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal RAG Agent runner")
    p.add_argument("--input", required=True, help="输入 JSON (数组)")
    p.add_argument("--output", required=True, help="输出 predictions.jsonl")
    p.add_argument(
        "--strategy",
        choices=("baseline", "rag"),
        default="baseline",
        help="baseline = 无检索; rag = 借鉴 chuangzhi-summer-camp 的 BM25+TF-IDF RRF",
    )
    p.add_argument("--topk", type=int, default=4, help="RAG 检索条数")
    p.add_argument("--limit", type=int, default=0, help=">0 时仅跑前 N 条 (调试用)")
    p.add_argument("--corpus", default="", help="检索语料 JSON 路径; 默认与 --input 相同")
    return p.parse_args(argv)


def load_items(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def build_retriever(corpus: list[dict]) -> HybridRetriever:
    r = HybridRetriever()
    for item in corpus:
        text = str(item.get("question") or "")
        r.add(item, text)
    return r


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_path = out_path.parent / "time_cost.jsonl"

    items = load_items(in_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    retriever = None
    if args.strategy == "rag":
        corpus_path = Path(args.corpus) if args.corpus else in_path
        corpus = load_items(corpus_path)
        retriever = build_retriever(corpus)

    client = QwenVLClient()

    n_total = len(items)
    print(f"[main] strategy={args.strategy}  items={n_total}  model={client.model}", flush=True)

    with out_path.open("w", encoding="utf-8") as fout, time_path.open("w", encoding="utf-8") as ftime:
        for i, item in enumerate(items, 1):
            idx = item.get("index")
            section = str(item.get("section") or "")
            image_url = str(item.get("image_url") or "")

            t0 = time.perf_counter()
            if args.strategy == "rag" and retriever is not None:
                query = str(item.get("question") or "")
                retrieved = retriever.search(query, k=args.topk, exclude_index=idx)
                user_text = build_rag_user(item, retrieved)
            else:
                user_text = build_baseline_user(item)
            retrieval_cost = time.perf_counter() - t0

            messages = client.build_messages(SYSTEM_BASE, user_text, image_url)

            t1 = time.perf_counter()
            try:
                raw = client.chat(messages)
                err = ""
            except Exception as e:
                raw = ""
                err = f"{type(e).__name__}: {e}"
            infer_cost = time.perf_counter() - t1

            answer = normalize_answer(section, raw)
            out_row = {"index": idx, "answer": answer, "raw_answer": raw}
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            ftime.write(
                json.dumps(
                    {"index": idx, "retrieval_cost": round(retrieval_cost, 4), "infer_cost": round(infer_cost, 4)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            ftime.flush()

            tag = "OK" if not err else "ERR"
            print(f"[{i}/{n_total}] idx={idx} sec={section} {tag} ans={answer!r} t_infer={infer_cost:.2f}s", flush=True)
            if err:
                print(f"        {err}", flush=True)

    print(f"[main] done. predictions -> {out_path}\n         time_cost  -> {time_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
