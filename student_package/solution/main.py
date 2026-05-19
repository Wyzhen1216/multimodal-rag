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
    SYS_DETECT_MULTI,
    SYS_IMAGE_DESC,
    SYS_OPTION_AS_JUDGE,
    SYS_OPTION_AS_JUDGE_VQA,
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


def build_query(question: str, section: str = "") -> str:
    """基础清洗：选择题去掉 'A.xxx B.xxx' 选项文字，避免选项内容污染向量检索。"""
    q = (question or "").strip()
    if "选择" in section:
        import re
        q = re.split(r"(?<![A-Za-z])A[.．。、\s]", q)[0].strip()
    return q


_Q_SYS = "你是考古学信息检索助手，帮助生成精准的文献检索查询。"

_Q_REWRITE_TMPL = (
    '将以下考古学题目改写为一句【陈述句】，用于文献检索。'
    '要求：①保留关键实体（人名/地名/器物名/朝代/时期）；'
    '②去掉"下列哪种""什么""是否"等问句结构；'
    '③直接输出陈述句，不超过40字，不要任何前缀。\n'
    '示例：\n'
    '  输入：湖北某遗址出土的龙山文化器物中哪种口缘呈花边状？\n'
    '  输出：湖北某遗址出土龙山文化器物花边状口缘特征\n'
    '题干：{q}\n输出：'
)


def rewrite_query(client, question: str, section: str) -> str:
    """用 LLM 把题目改写为陈述句，提升向量检索语义精度；失败时退回原始 query。"""
    base = build_query(question, section=section)
    if not base:
        return question
    try:
        r = client.generate(_Q_SYS, _Q_REWRITE_TMPL.format(q=base), temperature=0.0, max_tokens=60)
        s = r.strip().splitlines()[0].strip()
        return s if s else base
    except Exception:
        return base


def extract_valid_options(question: str) -> set:
    """从题干中解析实际存在的选项字母，防止模型幻想不存在的 E/F/G 选项。"""
    import re
    return set(re.findall(r"(?<![A-Za-z])([A-H])[.．。、\s]", question))


def parse_choice_options(question: str) -> list:
    """
    解析选择题题干，返回 [(letter, option_text), ...]。
    用于选项感知检索：把每个选项内容作为单独的检索query。
    """
    import re
    # 匹配 "A. xxx" "B、xxx" "C xxx" 等模式直到下一个字母或结尾
    # 选项之间用字母 + 分隔符 区分
    pattern = re.compile(r"(?<![A-Za-z])([A-H])[.．。、\s]+([^A-H]+?)(?=(?:(?<![A-Za-z])[A-H][.．。、\s])|$)", re.DOTALL)
    matches = pattern.findall(question)
    return [(letter, text.strip(' \t\n、。，,.')) for letter, text in matches if text.strip()]


def option_aware_contexts(retriever, client, question: str, section: str, topk: int) -> list:
    """
    选择题的选项感知检索：
      1) 用 LLM 改写后的主题 query 取 topk 文档
      2) 对题干中每个选项内容，单独再各取 1-2 篇文档（带主题约束）
      3) 合并去重，按 RRF-like 顺序排
    这样每个选项都至少有专属证据可供模型判断。
    """
    base_query = rewrite_query(client, question, section=section)
    main_docs = retriever.search(text=base_query, topk=topk)

    options = parse_choice_options(question)
    if not options:
        return main_docs

    # 题干主题（去掉选项）作为前缀，让选项query更聚焦
    topic_prefix = build_query(question, section=section)[:30]

    option_docs = []
    for letter, opt_text in options:
        if not opt_text or len(opt_text) > 30:
            continue
        opt_query = f"{topic_prefix} {opt_text}".strip()
        try:
            docs = retriever.search(text=opt_query, topk=2)
            option_docs.extend(docs)
        except Exception:
            continue

    # 合并：主query结果优先，选项query结果补充
    seen = set()
    merged = []
    for d in main_docs + option_docs:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return merged[:topk + 4]   # 给模型多点上下文


# ============================================================================
# 选择题 → 4 个判断题（核心改动）
# ============================================================================

_MULTI_KW = ("哪些", "包括", "有哪些", "哪几", "哪几种", "哪几个", "哪几位", "都有")
_SINGLE_KW = ("哪一", "哪个", "哪种", "是什么", "是哪", "最有可能", "最可能", "主要是", "主要的", "属于哪")


def detect_multi_select(question: str, client) -> bool:
    """判定单/多选。先用关键字（高置信），再用 LLM 兜底。"""
    q = question or ""
    # 1) 强多选关键字（先判定，且要排除"哪一"包含"哪"的情况）
    if any(k in q for k in _MULTI_KW):
        return True
    # 2) 强单选关键字
    if any(k in q for k in _SINGLE_KW):
        return False
    # 3) LLM 兜底
    try:
        r = client.generate(SYS_DETECT_MULTI, f"题目：{q}\n输出 single 或 multi：",
                            temperature=0.0, max_tokens=8)
        return "multi" in (r or "").lower()
    except Exception:
        return False  # 默认单选，避免多选幻觉


def _judge_one_option(
    retriever, client, question_stem: str, letter: str, opt_text: str,
    image_path, is_vqa: bool, topk: int,
) -> str:
    """对单个选项做一次 RAG + 判断；返回 '正确' / '错误' / ''。"""
    # 检索：把题干主题 + 选项内容拼接做 query；VQA 题也允许 image
    query = f"{question_stem[:50]} {opt_text}".strip()
    contexts = []
    try:
        contexts = retriever.search(
            text=query,
            image_path=str(image_path) if image_path else None,
            topk=topk,
        )
    except Exception:
        contexts = []

    ctx_block = ""
    if contexts:
        lines = [f"[资料{i+1}] {c[:300]}" for i, c in enumerate(contexts[:topk])]
        ctx_block = "以下是检索到的相关资料：\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{ctx_block}"
        f"[题干] {question_stem}\n"
        f"[待判选项] {letter}. {opt_text}\n\n"
        f"请判断 '{letter}. {opt_text}' 是否是题干所问的正确答案。仅输出 '正确' 或 '错误'："
    )
    sys_p = SYS_OPTION_AS_JUDGE_VQA if is_vqa else SYS_OPTION_AS_JUDGE
    try:
        raw = client.generate(sys_p, user, image_path, temperature=0.0, max_tokens=8)
    except Exception:
        return ""
    from .normalization import normalize_judgment_answer
    return normalize_judgment_answer(raw)


def choice_as_judge(
    retriever, client, question: str, section: str,
    image_path, topk: int,
) -> tuple[str, list]:
    """
    把选择题拆成 N 个判断题（每个选项独立判定）：
      - 单选：取所有 '正确' 中第一个，若全 '错误' 则退回常规推理
      - 多选：取所有 '正确' 字母连写
    返回 (answer_letters, all_retrieved_contexts)
    """
    options = parse_choice_options(question)
    if not options:
        return "", []

    # 题干去掉选项部分
    stem = build_query(question, section=section)
    is_vqa = "VQA" in section
    is_multi = detect_multi_select(question, client)

    correct_letters: list = []
    all_ctx: list = []
    for letter, opt_text in options:
        verdict = _judge_one_option(
            retriever, client, stem, letter, opt_text,
            image_path, is_vqa, topk,
        )
        print(f"  [option {letter}] {opt_text[:20]} -> {verdict}", flush=True)
        if verdict == "正确":
            correct_letters.append(letter)

    # 全部判定为 '错误' → 回退到常规选择题推理（让模型直接看题选）
    if not correct_letters:
        fallback = _fallback_choice_pick(client, question, image_path, is_vqa)
        if fallback:
            return fallback, all_ctx
        return "", all_ctx

    if is_multi:
        return "".join(sorted(correct_letters)), all_ctx
    # 单选：如果模型判定多个为"正确"，让模型在这些里挑一个
    if len(correct_letters) == 1:
        return correct_letters[0], all_ctx
    # 多个候选 → 让模型二次裁决
    cand_str = ", ".join(f"{l}. {dict(options)[l]}" for l in correct_letters)
    user = (
        f"[题干] {question}\n\n"
        f"上一步初筛后剩下候选：{cand_str}\n"
        f"请从候选中选出最贴切的【唯一】答案，仅输出字母："
    )
    try:
        from .prompts import SYS_CHOICE_DIRECT
        raw = client.generate(SYS_CHOICE_DIRECT, user, image_path,
                              temperature=0.0, max_tokens=8)
        ans = normalize_answer(section, raw)
        if ans and ans[0] in correct_letters:
            return ans[0], all_ctx
    except Exception:
        pass
    return correct_letters[0], all_ctx


def _fallback_choice_pick(client, question: str, image_path, is_vqa: bool) -> str:
    """逐选项判断全失败时的兜底：让模型用自身知识直接选一个（无 RAG）。"""
    from .prompts import SYS_CHOICE_DIRECT, SYS_CHOICE_VQA
    sys_p = SYS_CHOICE_VQA if is_vqa else SYS_CHOICE_DIRECT
    user = (
        f"[题目] {question}\n\n"
        "请直接输出选项字母（单选 1 个，多选连写多个，无分隔符）："
    )
    try:
        raw = client.generate(sys_p, user, image_path, temperature=0.0, max_tokens=16)
        from .normalization import normalize_choice_answer
        return normalize_choice_answer(raw)
    except Exception:
        return ""


# ============================================================================
# VQA 图像描述 → 文本检索对齐
# ============================================================================

def describe_image(client, image_path, question: str) -> str:
    """让 LLM 看图给出 1-2 句描述，用于辅助文本检索；失败时返回 ''。"""
    if image_path is None:
        return ""
    user = f"题目：{question}\n请描述图中关键内容（≤50字，不要前缀）："
    try:
        r = client.generate(SYS_IMAGE_DESC, user, image_path,
                            temperature=0.0, max_tokens=80)
        return (r or "").strip().splitlines()[0].strip()[:120]
    except Exception:
        return ""


# ============================================================================
# 简答 QA: 多 query 发散检索
# ============================================================================

_DIVERGE_SYS = "你是考古学信息检索助手。"
_DIVERGE_TMPL = (
    '为以下简答题生成 3 个不同的【陈述句】检索query，覆盖不同表达方式 / 相关实体。'
    '每行一个，不超过 35 字，不要编号、不要前缀。\n'
    '题目：{q}\n'
    '3 条 query：'
)


def diverge_queries(client, question: str) -> list[str]:
    """简答题发散：让 LLM 生成多个 query 变体，失败时只返回原始题干。"""
    try:
        r = client.generate(_DIVERGE_SYS, _DIVERGE_TMPL.format(q=question),
                            temperature=0.5, max_tokens=120)
        lines = [l.strip(' \t-•0123456789.、，。') for l in r.splitlines() if l.strip()]
        return [l for l in lines if 5 <= len(l) <= 60][:3]
    except Exception:
        return []


def majority_vote(answers: List[str], section: str) -> str:
    """对多次采样做归一化后投票，取多数票；票数相同时取第一个非空。"""
    normed = [normalize_answer(section, a) for a in answers]
    normed = [a for a in normed if a]
    if not normed:
        return ""
    counter = Counter(normed)
    top, _ = counter.most_common(1)[0]
    return top


def pick_answer(raw: str, section: str, fallback_if_empty: bool = True,
                valid_options: set | None = None) -> str:
    """对模型输出做归一化；失败兜底。valid_options 非空时过滤掉题目中不存在的选项字母。"""
    ans = normalize_answer(section, raw)
    if ans and "选择" in section and valid_options:
        # 去掉幻想出来的选项（例如题目只有 A-D，模型却输出 E）
        filtered = "".join(c for c in ans if c in valid_options)
        if filtered:
            ans = filtered
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
            # 跳过常规检索的条件：
            #   简答VQA - 纯视觉题，看图即可
            #   选择题（QA/VQA）- 走 choice_as_judge 分支，内部自带检索
            is_choice = "选择" in section
            skip_retrieval = is_short_vqa or is_choice
            # 选择题：解析题目中实际存在的选项，防止模型幻想 E/F/G
            valid_options = extract_valid_options(question) if is_choice else None

            # ---- 分支 1: 选择题 → 拆分为 N 个判断题 ----
            if is_choice:
                t0 = time.perf_counter()
                contexts: List[str] = []
                raw = ""
                try:
                    answer, contexts = choice_as_judge(
                        retriever, client, question, section, image_path,
                        topk=args.topk,
                    )
                    if valid_options and answer:
                        filtered = "".join(c for c in answer if c in valid_options)
                        if filtered:
                            answer = filtered
                    if not answer:
                        answer = fallback_answer(section)
                    raw = answer
                except Exception as e:
                    print(f"[idx={idx}] choice_as_judge error: {e}", flush=True)
                    answer = fallback_answer(section)
                infer_cost = time.perf_counter() - t0
                retrieval_cost = 0.0

                fout.write(json.dumps(
                    {"index": idx, "answer": answer, "raw_answer": raw},
                    ensure_ascii=False,
                ) + "\n")
                fout.flush()
                ftime.write(json.dumps(
                    {"index": idx,
                     "retrieval_cost": round(retrieval_cost, 4),
                     "infer_cost": round(infer_cost, 4)},
                    ensure_ascii=False,
                ) + "\n")
                ftime.flush()
                print(
                    f"[{i}/{len(items)}] idx={idx} sec={section} "
                    f"ans={answer!r} t_inf={infer_cost:.2f}s (choice-as-judge)",
                    flush=True,
                )
                continue

            # ---- 分支 2: 判断 / 简答 走常规检索 + 推理 ----
            # ---- 检索 ----
            t0 = time.perf_counter()
            contexts = []
            if not skip_retrieval:
                try:
                    if section == "判断QA":
                        # 判断QA: 题干本身就是陈述句，直接用作 query
                        query = build_query(question, section=section)
                        contexts = retriever.search(text=query, topk=10)
                        print(f"[idx={idx}] direct query={query[:60]} ctx={len(contexts)}", flush=True)
                    elif section == "简答QA":
                        # 简答QA: 发散检索（主 query + 多个改写 query）合并去重
                        base_q = rewrite_query(client, question, section=section)
                        variants = diverge_queries(client, question)
                        queries = [base_q] + [v for v in variants if v and v != base_q]
                        seen = set()
                        merged = []
                        for q in queries[:4]:
                            try:
                                docs = retriever.search(text=q, topk=6)
                            except Exception:
                                docs = []
                            for d in docs:
                                if d not in seen:
                                    seen.add(d)
                                    merged.append(d)
                        contexts = merged[:12]
                        print(f"[idx={idx}] diverged {len(queries)} queries, ctx={len(contexts)}", flush=True)
                    else:
                        # 判断VQA: 图像描述 + 题干 拼成 text query 做向量检索
                        base_q = build_query(question, section=section)
                        img_desc = describe_image(client, image_path, question) if image_path else ""
                        query = f"{base_q} {img_desc}".strip() if img_desc else base_q
                        contexts = retriever.search(
                            text=query,
                            image_path=str(image_path) if image_path else None,
                            topk=args.topk,
                        )
                        print(f"[idx={idx}] VQA query={query[:80]} ctx={len(contexts)}", flush=True)
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
                elif "判断" in section and args.judge_samples > 1:
                    # 判断题: 多次采样投票
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
                    # 单次贪心
                    if "判断" in section:
                        max_tok = 16
                    else:
                        max_tok = 64
                    raw = client.generate(
                        system, user, image_path,
                        temperature=0.0,
                        max_tokens=max_tok,
                    )
                    answer = pick_answer(raw, section, valid_options=valid_options)
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
