"""
solution.main —— 多模态历史 RAG Agent 主入口。

流程：
  1. 加载输入 (data/public_examples.full.json)
  2. 一次性初始化 LLM 客户端 + DashVector retriever
  3. 按题逐条处理：
       a. 题型路由：简答VQA 跳过检索
       b. 构造检索 query（题干 + 选项）
       c. DashVector 检索 top-k 文献片段 (retrieval_cost 计时)
       d. 按题型分支构造 prompt
       e. Qwen2.5-VL 推理 (infer_cost 计时)
       f. 简答VQA：多次采样投票
       g. 答案归一化 + 兜底
       h. append 写入 predictions.jsonl + time_cost.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .normalization import normalize_answer  # noqa: E402

from .llm_client import build_client, resolve_local_image  # noqa: E402
from .prompts import (  # noqa: E402
    build_user_message,
    fallback_answer,
    fewshot_block,
    system_prompt_for,
)
from .retriever import build_retriever  # noqa: E402


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal History RAG Agent")
    p.add_argument("--input", required=True, help="输入 JSON 数组路径")
    p.add_argument("--output", required=True, help="输出 predictions.jsonl 路径")
    p.add_argument("--topk", type=int, default=5, help="DashVector top-k")
    p.add_argument("--limit", type=int, default=0, help=">0 时仅跑前 N 题（调试）")
    p.add_argument("--no-fewshot", action="store_true", help="禁用 few-shot")
    p.add_argument(
        "--vqa-samples",
        type=int,
        default=5,
        help="简答VQA 多次采样次数 (>=1, 1 = 不采样, 5 是默认值)",
    )
    p.add_argument(
        "--choice-samples",
        type=int,
        default=3,
        help="选择题多次采样投票次数 (1 = 关闭, 3 是默认值, 5 更稳但贵)",
    )
    p.add_argument(
        "--judge-samples",
        type=int,
        default=3,
        help="判断题多次采样投票次数 (1 = 关闭, 3 是默认值)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="若 predictions.jsonl 已存在，跳过已完成的 index",
    )
    p.add_argument("--llm", choices=("auto", "api", "local"), default="auto")
    return p.parse_args(argv)


# ============================================================================
# Helpers
# ============================================================================

def load_items(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def load_done_indexes(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row.get("index"))
            except Exception:
                pass
    return done


def build_query(question: str) -> str:
    """检索 query：直接使用题干即可（题干通常已包含关键考古学术语和选项内容）。"""
    return (question or "").strip()


def majority_vote(answers: List[str], section: str) -> str:
    """对多次采样做归一化后投票，取多数票；票数相同时取第一个非空。"""
    normed = [normalize_answer(section, a) for a in answers]
    normed = [a for a in normed if a]
    if not normed:
        return ""
    counter = Counter(normed)
    top, _ = counter.most_common(1)[0]
    return top


def pick_answer(raw: str, section: str, fallback_if_empty: bool = True) -> str:
    """对模型输出做归一化；失败兜底。"""
    ans = normalize_answer(section, raw)
    if ans:
        return ans
    if fallback_if_empty:
        return fallback_answer(section)
    return ""


# ============================================================================
# 主循环
# ============================================================================

def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_path = out_path.parent / "time_cost.jsonl"

    items = load_items(in_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    done = load_done_indexes(out_path) if args.resume else set()
    if done:
        print(f"[main] resume mode: skip {len(done)} done indexes", flush=True)

    # 一次性初始化
    retriever = build_retriever()
    client = build_client(prefer=None if args.llm == "auto" else args.llm)
    print(f"[main] llm={client.name}  retriever={type(retriever).__name__}", flush=True)
    print(f"[main] total items: {len(items)}", flush=True)

    # 文件以 append 打开，支持续跑
    open_mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(open_mode, encoding="utf-8")
    ftime = time_path.open(open_mode, encoding="utf-8")

    try:
        for i, item in enumerate(items, 1):
            idx = item.get("index")
            if idx in done:
                continue
            section = str(item.get("section") or "")
            question = str(item.get("question") or "")
            image_url = str(item.get("image_url") or "")
            image_path = resolve_local_image(image_url) if image_url else None

            is_short_vqa = section == "简答VQA"
            skip_retrieval = is_short_vqa  # 简答VQA 跳过检索

            # ---- 检索 ----
            t0 = time.perf_counter()
            contexts: List[str] = []
            if not skip_retrieval:
                query = build_query(question)
                try:
                    contexts = retriever.search(
                        text=query,
                        image_path=str(image_path) if image_path else None,
                        topk=args.topk,
                    )
                except Exception as e:
                    print(f"[idx={idx}] retrieval error: {e}", flush=True)
                    contexts = []
            retrieval_cost = time.perf_counter() - t0

            # ---- Prompt ----
            system = system_prompt_for(section)
            user = build_user_message(section, question, contexts)
            if not args.no_fewshot:
                user = fewshot_block(section) + user

            # ---- 推理 ----
            t1 = time.perf_counter()
            raw = ""
            try:
                if is_short_vqa and args.vqa_samples > 1:
                    # 简答VQA: 多次采样投票
                    samples = client.generate_n(
                        system, user, image_path,
                        n=args.vqa_samples,
                        temperature=0.7,
                        max_tokens=64,
                    )
                    voted = majority_vote(samples, section)
                    raw = samples[0]
                    answer = voted or pick_answer(samples[0], section)
                elif "选择" in section and args.choice_samples > 1:
                    # 选择题: CoT + 多次采样投票 (sampling 必须 temp>0)
                    samples = client.generate_n(
                        system, user, image_path,
                        n=args.choice_samples,
                        temperature=0.5,
                        max_tokens=320,
                    )
                    voted = majority_vote(samples, section)
                    raw = samples[0]
                    answer = voted or pick_answer(samples[0], section)
                elif "判断" in section and args.judge_samples > 1:
                    # 判断题: CoT + 多次采样投票
                    samples = client.generate_n(
                        system, user, image_path,
                        n=args.judge_samples,
                        temperature=0.5,
                        max_tokens=200,
                    )
                    voted = majority_vote(samples, section)
                    raw = samples[0]
                    answer = voted or pick_answer(samples[0], section)
                else:
                    # 单次贪心 -- 选择题 CoT 给 320 tokens, 判断题直答只需 16
                    if "选择" in section:
                        max_tok = 320
                    elif "判断" in section:
                        max_tok = 16
                    else:
                        max_tok = 64
                    raw = client.generate(
                        system, user, image_path,
                        temperature=0.0,
                        max_tokens=max_tok,
                    )
                    answer = pick_answer(raw, section)
            except Exception as e:
                print(f"[idx={idx}] infer error: {e}", flush=True)
                answer = fallback_answer(section)
                raw = ""
            infer_cost = time.perf_counter() - t1

            # ---- 落盘 ----
            fout.write(json.dumps(
                {"index": idx, "answer": answer, "raw_answer": raw},
                ensure_ascii=False,
            ) + "\n")
            fout.flush()
            ftime.write(json.dumps(
                {
                    "index": idx,
                    "retrieval_cost": round(retrieval_cost, 4),
                    "infer_cost": round(infer_cost, 4),
                },
                ensure_ascii=False,
            ) + "\n")
            ftime.flush()

            print(
                f"[{i}/{len(items)}] idx={idx} sec={section} "
                f"ans={answer!r} t_ret={retrieval_cost:.2f}s t_inf={infer_cost:.2f}s "
                f"ctx={len(contexts)}",
                flush=True,
            )
    finally:
        fout.close()
        ftime.close()

    print(f"[main] done. predictions -> {out_path}\n         time_cost  -> {time_path}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
