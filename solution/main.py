"""
solution.main —— 多模态历史 RAG Agent 主入口。

流程：
  1. 加载输入 (data/public_examples.full.json)
  2. 一次性初始化 LLM 客户端 + DashVector retriever
  3. 按题逐条处理（所有题型均做 RAG 检索，满足赛题要求）：
       a. 题型路由：选择题走 choice_as_judge（每选项独立检索 + 判断）；
          其余题型走外层检索一次
       b. 构造检索 query（题干 + 选项 / 题干 + 图像描述）
       c. DashVector 检索 top-k 文献片段 (retrieval_cost 计时)
       d. 按题型分支构造 prompt
       e. Qwen2.5-VL 推理 (infer_cost 计时)
       f. 简答VQA：图像优先，RAG 仅作辅助；视觉直读题不注入 ctx
       g. 答案归一化 + 兜底
       h. append 写入 predictions.jsonl + time_cost.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .normalization import normalize_answer  # noqa: E402

from .llm_client import build_client, resolve_local_image  # noqa: E402
from .prompts import (  # noqa: E402
    SYS_ANSWER_REFLECT,
    SYS_CTX_RERANK,
    SYS_DETECT_MULTI,
    SYS_ENTITY_RECHECK,
    SYS_EXTRACT_ANSWER_IN_Q,
    SYS_IMAGE_AS_QUERY,
    SYS_IMAGE_DESC,
    SYS_JUDGE_SOFT,
    SYS_JUDGE_SOFT_VQA,
    SYS_LENIENT_RECHECK,
    SYS_OVD_ENTITY_EXTRACT,
    SYS_OPTION_AS_JUDGE,
    SYS_OPTION_AS_JUDGE_VQA,
    SYS_PREDICT_COUNT,
    SYS_REVERSE_QUERY,
    SYS_SHORT_EXTRACT,
    SYS_SINGLE_CONTRAST,
    build_extract_user_message,
    build_judge_user_soft,
    build_short_vqa_user,
    build_user_message,
    fallback_answer,
    fewshot_block,
    system_prompt_for,
)
from .retriever import build_retriever  # noqa: E402
from .startup_check import run_startup_checks  # noqa: E402


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
    p.add_argument("--llm", choices=("auto", "api", "local"), default="local")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数（>1 启用 ThreadPoolExecutor；GPU embed 会自动串行化）",
    )
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


_HYDE_SYS = (
    "你是一名中国考古学与历史学专家。请对以下问题给出最可能的简短答案（不超过30字）。"
    "即使不完全确定也要给出最有可能的答案，不要回答'不知道'。直接输出答案，不要前缀。"
)


def _generate_hyde_answer(client, question: str) -> str:
    """HyDE（Hypothetical Document Embedding）：生成假设答案作为额外检索 query。

    假设答案与目标文档在语义空间中更接近，能召回正式 query 覆盖不到的证据文献。
    失败时返回空串，不影响主流程。
    """
    try:
        r = client.generate(
            _HYDE_SYS, f"问题：{question}",
            temperature=0.3, max_tokens=60,
        )
        s = (r or "").strip().splitlines()[0].strip()
        if s and 3 <= len(s) <= 60:
            return s
    except Exception:
        pass
    return ""


def _generate_hyde_answer_with_context(client, question: str, context_str: str) -> str:
    """HyDE 第二轮：借助第一轮检索上下文，生成更精确的假设答案 query。

    第一轮粗检索后，用已找到的文献片段精化假设答案，
    使其更贴近目标文献的原文表述，从而命中更精确的文档。
    """
    user = (
        f"参考资料：\n{context_str}\n\n"
        f"问题：{question}\n"
        f"请根据以上资料给出更准确的答案（不超过15字，直接输出答案本身，不要前缀）："
    )
    try:
        r = client.generate(
            _HYDE_SYS, user,
            temperature=0.1, max_tokens=30,
        )
        s = (r or "").strip().splitlines()[0].strip()
        if s and 3 <= len(s) <= 30:
            return s
    except Exception:
        pass
    return ""


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


# ============================================================================
# 噪声 chunk 过滤（T1）：过滤图说/摘要/章节标题等无信息量短文本
# ============================================================================

_NOISE_PREFIXES = (
    "图", "表", "Fig", "Table", "[", "（图", "(图",
    "摘要", "Abstract", "ABSTRACT",
    "前言", "目录", "参考文献", "致谢", "Key words", "关键词",
)


def _is_noise_chunk(text: str) -> bool:
    """判定 chunk 是否为噪声：过短 / 元数据 / 章节标题 / 图表说明等。"""
    if not text:
        return True
    t = text.strip()
    if len(t) < 50:
        return True
    if any(t.startswith(p) for p in _NOISE_PREFIXES):
        return True
    return False


def _filter_noise(contexts: list, keep_min: int = 2) -> list:
    """过滤噪声 chunk；全空时保留前 keep_min 条以免后续无 ctx。"""
    if not contexts:
        return []
    filtered = [c for c in contexts if not _is_noise_chunk(c)]
    if not filtered:
        return contexts[:keep_min]
    return filtered


def llm_rerank_contexts(
    client,
    question: str,
    contexts: list,
    max_rerank: int = 6,
    keep_min: int = 2,
) -> list:
    """信息瓶颈式 LLM 重排（思路2 IB-Reranking）：过滤与题目完全无关的 chunk。

    对前 max_rerank 条逐一用 SYS_CTX_RERANK 判断相关性；
    无关 chunk 被丢弃；过滤后不足 keep_min 时回退原始前 keep_min 条。
    max_rerank 以后的 chunk 原样保留（控制额外 LLM 调用数）。
    """
    if not contexts or len(contexts) <= keep_min:
        return contexts
    candidates = contexts[:max_rerank]
    relevant = []
    for ctx in candidates:
        user = f"[题目] {question}\n[资料片段] {ctx[:300]}\n判断是否相关："
        try:
            r = client.generate(SYS_CTX_RERANK, user, temperature=0.0, max_tokens=8)
            r_str = (r or "").strip()
            is_relevant = "相关" in r_str and "无关" not in r_str
        except Exception:
            is_relevant = True  # 推理失败时保留，避免误删
        if is_relevant:
            relevant.append(ctx)
    remaining = contexts[max_rerank:]
    result = relevant + remaining
    return result if len(result) >= keep_min else contexts[:keep_min]


def _is_inscription_char_recognition(question: str) -> bool:
    """简答VQA 题面问'铭文字识别' 且不问'代表/称谓' 时返回 True。
    用途: 这类题加 fewshot 反而让模型在视觉判断时被无关示例带偏。
    """
    if "字" not in question:
        return False
    bias_words = ("代表", "称谓", "含义", "意思", "意义", "象征")
    if any(w in question for w in bias_words):
        return False
    return True


def _is_visual_direct_read(question: str) -> bool:
    """简答VQA 中需要直接从图像读取的题型：识字、数数、方向、握持物体等。

    这类题 RAG context 注入反而引入噪音，应做检索（满足赛题要求）但不把结果注入 prompt。
    """
    direct_keywords = (
        "什么字", "哪个字", "第几字", "第三字", "第几列", "第一列", "铭文",
        "数量是多少", "穿孔数", "有几个", "有多少", "共有几",
        "朝向是", "方向是", "朝哪", "向哪",
        "握持的是", "握着什么", "持有什么",
    )
    return any(kw in question for kw in direct_keywords)


# ============================================================================
# 题干自含答案抽取（T2）：简答 QA 前置短路
# ============================================================================

def extract_answer_from_question(client, question: str) -> str:
    """检测题干本身是否已含答案；命中返回答案，否则返回 ''。

    例：'江西收集的西汉铜钾的器物名称是什么？' → '铜钾'
    """
    if not question:
        return ""
    try:
        r = client.generate(
            SYS_EXTRACT_ANSWER_IN_Q, question,
            temperature=0.0, max_tokens=24,
        )
    except Exception:
        return ""
    s = (r or "").strip().splitlines()[0].strip(" \t、。，,.；;:'\"`")
    if not s:
        return ""
    if s.upper() == "NO" or "NO" == s.upper().strip():
        return ""
    if len(s) > 16:
        return ""
    if s not in question:
        return ""
    return s


# ============================================================================
# 反向 query 生成（T3）：针对每个选项的"假设成立时资料应有的陈述"
# ============================================================================

def make_reverse_query(client, question_stem: str, opt_text: str) -> str:
    """对单个选项生成反向陈述 query。失败返回 ''。"""
    if not opt_text:
        return ""
    user = (
        f"[题干] {question_stem}\n"
        f"[选项] {opt_text}\n"
        f"输出陈述 query："
    )
    try:
        r = client.generate(
            SYS_REVERSE_QUERY, user,
            temperature=0.0, max_tokens=60,
        )
    except Exception:
        return ""
    s = (r or "").strip().splitlines()[0].strip(" \t、。，,.;:'\"`")
    if not s or len(s) < 5 or len(s) > 80:
        return ""
    # 反向 query 不能等于选项本身或题干本身
    if s == opt_text or s == question_stem:
        return ""
    return s


# ============================================================================
# 多选答案数量预测（T5）：用作"数量不足时再次复查"的软提示
# ============================================================================

def predict_answer_count(client, question: str, options: list) -> int:
    """预测多选题答案个数；失败返回 0（视作未预测）。"""
    if not options:
        return 0
    opts_str = " ".join(f"{l}.{t}" for l, t in options)
    try:
        r = client.generate(
            SYS_PREDICT_COUNT,
            f"[题目] {question}\n[选项] {opts_str}\n预测答案个数：",
            temperature=0.0, max_tokens=8,
        )
    except Exception:
        return 0
    s = (r or "").strip()
    for ch in s:
        if ch in "1234":
            n = int(ch)
            return n if 1 <= n <= len(options) else 0
    return 0


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
    # 3) 选项数 >= 5 → 通常是多选
    opts = parse_choice_options(q)
    if len(opts) >= 5:
        return True
    # 4) LLM 兜底
    try:
        r = client.generate(SYS_DETECT_MULTI, f"题目：{q}\n输出 single 或 multi：",
                            temperature=0.0, max_tokens=8)
        return "multi" in (r or "").lower()
    except Exception:
        return False  # 默认单选，避免多选幻觉


def _judge_one_option(
    retriever, client, question_stem: str, letter: str, opt_text: str,
    image_path, is_vqa: bool, topk: int,
) -> tuple[str, list, float]:
    """对单个选项做一次 RAG + 判断；返回 (verdict, contexts, retrieval_cost)。

    T1/T3: 扩大召回 topk * 3 → 噪声过滤 → 合并反向 query 检索 → 取 topk。
    retrieval_cost 累计本选项所有 retriever.search 耗时（供选择题 time_cost 上报）。
    """
    ret_cost = 0.0

    # 正向 query：题干前缀 + 选项内容
    fwd_query = f"{question_stem[:50]} {opt_text}".strip()
    fwd_docs: list = []
    _t = time.perf_counter()
    try:
        fwd_docs = retriever.search(
            text=fwd_query,
            image_path=str(image_path) if image_path else None,
            topk=topk * 3,
        )
    except Exception:
        fwd_docs = []
    ret_cost += time.perf_counter() - _t

    # 反向 query（T3）：让 LLM 生成"假设此选项成立时资料应有的陈述"
    rev_docs: list = []
    rev_q = make_reverse_query(client, question_stem, opt_text)
    if rev_q:
        _t = time.perf_counter()
        try:
            rev_docs = retriever.search(
                text=rev_q,
                image_path=str(image_path) if image_path else None,
                topk=topk,
            )
        except Exception:
            rev_docs = []
        ret_cost += time.perf_counter() - _t

    # 合并 → 噪声过滤 → 去重 → 取 topk
    seen = set()
    merged: list = []
    for d in fwd_docs + rev_docs:
        if d in seen:
            continue
        seen.add(d)
        merged.append(d)
    contexts = _filter_noise(merged)[:topk]

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
        return "", contexts, ret_cost
    from .normalization import normalize_judgment_answer
    return normalize_judgment_answer(raw), contexts, ret_cost


def _recheck_option_lenient(
    retriever, client, question_stem: str, letter: str, opt_text: str,
    already_correct_letters: list, options: list,
    image_path, is_vqa: bool, topk: int,
) -> str:
    """对'错误'选项做更宽松的复查：是否也可能是题干所问的【一个】答案。

    多选漏判修复——首轮太保守的选项给一次复活机会。
    """
    sibling_str = ", ".join(f"{l}.{dict(options)[l]}" for l in already_correct_letters)
    query = f"{question_stem[:50]} {opt_text}"
    contexts: list = []
    try:
        contexts = retriever.search(
            text=query,
            image_path=str(image_path) if image_path else None,
            topk=topk,
        )
    except Exception:
        pass

    ctx_block = ""
    if contexts:
        lines = [f"[资料{i+1}] {c[:300]}" for i, c in enumerate(contexts[:topk])]
        ctx_block = "以下是检索到的资料：\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{ctx_block}"
        f"[题干] {question_stem}\n"
        f"该题为多选题，已确定 [{sibling_str}] 属于答案。\n"
        f"[复查选项] {letter}. {opt_text}\n\n"
        f"请判断 '{letter}. {opt_text}' 是否也属于题干所问的【一个】答案（多选可能漏选）。"
        f"仅输出 '正确' 或 '错误'："
    )
    sys_p = SYS_OPTION_AS_JUDGE_VQA if is_vqa else SYS_OPTION_AS_JUDGE
    try:
        raw = client.generate(sys_p, user, image_path, temperature=0.0, max_tokens=8)
    except Exception:
        return ""
    from .normalization import normalize_judgment_answer
    return normalize_judgment_answer(raw)


def _recheck_when_only_one(
    retriever, client, question_stem: str, letter: str, opt_text: str,
    only_correct_letter: str, only_correct_text: str, options: list,
    image_path, is_vqa: bool, topk: int,
) -> str:
    """多选题仅 1 个正确选项时的二次激进复查。

    多选题极少只有一个答案；用更强的暗示让模型重新评估剩余选项，
    并显式要求"不确定时倾向于'正确'"以对抗首轮过保守判定。
    """
    query = f"{question_stem[:50]} {opt_text}"
    contexts: list = []
    try:
        contexts = retriever.search(
            text=query,
            image_path=str(image_path) if image_path else None,
            topk=topk,
        )
    except Exception:
        pass

    ctx_block = ""
    if contexts:
        lines = [f"[资料{i+1}] {c[:300]}" for i, c in enumerate(contexts[:topk])]
        ctx_block = "以下是检索到的资料：\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{ctx_block}"
        f"[题干] {question_stem}\n"
        f"该题为【多选题】，但首轮判定后【仅 1 个】选项被认为正确："
        f"{only_correct_letter}. {only_correct_text}。\n"
        f"多选题极少只有 1 个答案，可能存在漏选。\n"
        f"[再次复查] {letter}. {opt_text}\n\n"
        f"请认真重新评估：'{letter}. {opt_text}' 是否也属于题干所问的答案之一？\n"
        f"原则：不确定时倾向 '正确'，仅在明显反例时输出 '错误'。仅输出 '正确' 或 '错误'："
    )
    sys_p = SYS_OPTION_AS_JUDGE_VQA if is_vqa else SYS_OPTION_AS_JUDGE
    try:
        raw = client.generate(sys_p, user, image_path, temperature=0.0, max_tokens=8)
    except Exception:
        return ""
    from .normalization import normalize_judgment_answer
    return normalize_judgment_answer(raw)


def _recheck_count_mismatch(
    retriever, client, question_stem: str, letter: str, opt_text: str,
    current_correct: list, expected_count: int, options: list,
    image_path, is_vqa: bool, topk: int,
) -> str:
    """多选题"数量不足"复查（T5）：预测有 N 个答案，目前仅判出 M < N 个。

    与 _recheck_when_only_one 的区别：用预测数量作为软约束 hint。
    """
    query = f"{question_stem[:50]} {opt_text}"
    contexts: list = []
    try:
        contexts = retriever.search(
            text=query,
            image_path=str(image_path) if image_path else None,
            topk=topk,
        )
    except Exception:
        pass

    ctx_block = ""
    if contexts:
        lines = [f"[资料{i+1}] {c[:300]}" for i, c in enumerate(contexts[:topk])]
        ctx_block = "以下是检索到的资料：\n" + "\n".join(lines) + "\n\n"

    opts_map = dict(options)
    cur_str = ", ".join(f"{l}.{opts_map.get(l,'')}" for l in current_correct)
    cur_n = len(current_correct)
    user = (
        f"{ctx_block}"
        f"[题干] {question_stem}\n"
        f"该题为【多选题】，预估约有 {expected_count} 个正确答案，"
        f"目前仅判出 {cur_n} 个：[{cur_str}]。\n"
        f"数量不足，可能存在漏选。\n"
        f"[再次复查] {letter}. {opt_text}\n\n"
        f"请仔细评估：'{letter}. {opt_text}' 是否也属于题干所问的答案之一？\n"
        f"原则：在资料不明确反驳时倾向 '正确'，仅在明显矛盾时输出 '错误'。"
        f"仅输出 '正确' 或 '错误'："
    )
    sys_p = SYS_OPTION_AS_JUDGE_VQA if is_vqa else SYS_OPTION_AS_JUDGE
    try:
        raw = client.generate(sys_p, user, image_path, temperature=0.0, max_tokens=8)
    except Exception:
        return ""
    from .normalization import normalize_judgment_answer
    return normalize_judgment_answer(raw)


def _single_contrast_pick(
    client, question: str, candidates: list, ctx_per_letter: dict,
    image_path, is_vqa: bool,
) -> str:
    """单选多候选 → 带 RAG 证据的对比裁决，输出一个字母。"""
    blocks = []
    opts_map = dict(candidates)
    for letter, opt_text in candidates:
        ctxs = ctx_per_letter.get(letter, [])
        ctx_text = "\n".join(f"  - {c[:200]}" for c in ctxs[:3]) or "  (无证据)"
        blocks.append(f"[候选 {letter}. {opt_text}]\n{ctx_text}")
    body = "\n\n".join(blocks)

    user = (
        f"[题目] {question}\n\n"
        f"以下是每个候选的资料证据：\n\n{body}\n\n"
        f"请综合证据从这些候选里选出【唯一】最贴切的答案，仅输出字母："
    )
    try:
        raw = client.generate(SYS_SINGLE_CONTRAST, user, image_path,
                              temperature=0.0, max_tokens=8)
        from .normalization import normalize_choice_answer
        ans = normalize_choice_answer(raw)
        cand_letters = [l for l, _ in candidates]
        if ans and ans[0] in cand_letters:
            return ans[0]
    except Exception:
        pass
    return candidates[0][0] if candidates else ""


def _reflect_answer(
    retriever, client, question: str, section: str, answer: str,
    options: list, image_path, is_vqa: bool, topk: int,
) -> bool:
    """对最终答案做一次反思判定：写成判断题，仅 '正确'/'错误'。"""
    if not answer or not options:
        return True
    opt_map = dict(options)
    chosen_text = " + ".join(f"{l}.{opt_map.get(l,'')}" for l in answer if l in opt_map)
    if not chosen_text:
        return True
    query = f"{question[:60]} {chosen_text}"
    contexts: list = []
    try:
        contexts = retriever.search(
            text=query,
            image_path=str(image_path) if image_path else None,
            topk=topk,
        )
    except Exception:
        pass
    ctx_block = ""
    if contexts:
        lines = [f"[资料{i+1}] {c[:300]}" for i, c in enumerate(contexts[:topk])]
        ctx_block = "以下是相关资料：\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{ctx_block}"
        f"[题目] {question}\n"
        f"[拟定答案] {answer}（{chosen_text}）\n\n"
        f"请判断该答案是否正确：仅输出 '正确' 或 '错误'。"
    )
    try:
        raw = client.generate(SYS_ANSWER_REFLECT, user, image_path,
                              temperature=0.0, max_tokens=8)
        from .normalization import normalize_judgment_answer
        return normalize_judgment_answer(raw) != "错误"
    except Exception:
        return True


def _entity_probe_vqa(
    client,
    question: str,
    answer: str,
    image_path,
    contexts: list,
) -> str:
    """Visual-RAG 实体探针（思路4）：检查简答VQA 答案实体是否在图中可见。

    调用 SYS_ENTITY_RECHECK 判断可见性：
      - '可见' / '不适用' → 返回原答案
      - '不可见' → 触发 SYS_LENIENT_RECHECK（图像优先宽容复审）
    """
    if not image_path or not answer:
        return answer
    user_probe = f"[题目] {question}\n[拟定答案] {answer}"
    try:
        r = client.generate(
            SYS_ENTITY_RECHECK, user_probe, image_path,
            temperature=0.0, max_tokens=8,
        )
        verdict = (r or "").strip()
    except Exception:
        return answer

    if "不可见" not in verdict:
        return answer  # 可见 / 不适用 → 维持原答案

    print(f"  [entity_probe] answer={answer!r} verdict={verdict!r} → lenient recheck", flush=True)

    # 触发宽容复审：图像优先，资料仅辅助
    ctx_block = ""
    if contexts:
        blocks = [f"[参考{i + 1}] {c[:200]}" for i, c in enumerate(contexts[:3])]
        ctx_block = "\n".join(blocks) + "\n\n"
    recheck_user = (
        f"{ctx_block}"
        f"[题目] {question}\n"
        f"[前轮答案（可能有误）] {answer}\n\n"
        f"请重新综合图像和题目，给出最稳妥的答案："
    )
    try:
        new_raw = client.generate(
            SYS_LENIENT_RECHECK, recheck_user, image_path,
            temperature=0.0, max_tokens=32,
        )
        new_ans = normalize_answer("简答VQA", new_raw)
        if new_ans:
            print(f"  [entity_probe] recheck → {new_ans!r}", flush=True)
            return new_ans
    except Exception:
        pass
    return answer


def choice_as_judge(
    retriever, client, question: str, section: str,
    image_path, topk: int,
) -> tuple[str, list, float]:
    """
    把选择题拆成 N 个判断题（每个选项独立判定）：
      - 单选：取所有 '正确' 中第一个，多个候选时带 RAG 对比裁决
      - 多选：取所有 '正确' 字母连写；对 '错误' 选项做宽松复查（漏判修复）
      - 最终答案做一次反思；反思为 '错误' 时单选触发兜底

    返回 (answer, contexts_for_log, retrieval_cost)。retrieval_cost 累加所有
    选项检索耗时（并行执行下为各线程耗时之和，反映 RAG 实际开销）。
    """
    options = parse_choice_options(question)
    if not options:
        return "", []

    stem = build_query(question, section=section)
    is_vqa = "VQA" in section
    is_multi = detect_multi_select(question, client)

    correct_letters: list = []
    wrong_letters: list = []
    ctx_per_letter: dict = {}
    ret_cost_total: float = 0.0

    # 4 选项并行判定（_judge_one_option 内部 retriever+LLM 均线程安全）
    with ThreadPoolExecutor(max_workers=len(options)) as pool:
        future_map = {
            pool.submit(
                _judge_one_option,
                retriever, client, stem, letter, opt_text,
                image_path, is_vqa, topk,
            ): (letter, opt_text)
            for letter, opt_text in options
        }
        results_map: dict = {}
        for fut in as_completed(future_map):
            letter, opt_text = future_map[fut]
            try:
                verdict, ctxs, ret_cost = fut.result()
            except Exception as e:
                print(f"  [option {letter}] error: {e}", flush=True)
                verdict, ctxs, ret_cost = "", [], 0.0
            results_map[letter] = (verdict, ctxs)
            ret_cost_total += ret_cost

    for letter, opt_text in options:  # 按原顺序输出
        verdict, ctxs = results_map[letter]
        ctx_per_letter[letter] = ctxs
        print(f"  [option {letter}] {opt_text[:20]} -> {verdict} ctx={len(ctxs)}", flush=True)
        if verdict == "正确":
            correct_letters.append(letter)
        else:
            wrong_letters.append(letter)

    # 多选漏判修复：对每个被判'错误'的选项做一次宽松复查
    if is_multi and correct_letters and wrong_letters:
        for letter in list(wrong_letters):
            opt_text = dict(options)[letter]
            recheck = _recheck_option_lenient(
                retriever, client, stem, letter, opt_text,
                correct_letters, options, image_path, is_vqa, topk,
            )
            print(f"  [recheck {letter}] {opt_text[:20]} -> {recheck}", flush=True)
            if recheck == "正确":
                correct_letters.append(letter)
                wrong_letters.remove(letter)

    # 多选题但仅 1 个正确 → 二次激进复查（多选极少只有 1 个答案）
    if is_multi and len(correct_letters) == 1 and wrong_letters:
        only_letter = correct_letters[0]
        only_text = dict(options)[only_letter]
        for letter in list(wrong_letters):
            opt_text = dict(options)[letter]
            recheck2 = _recheck_when_only_one(
                retriever, client, stem, letter, opt_text,
                only_letter, only_text, options, image_path, is_vqa, topk,
            )
            print(f"  [recheck2 {letter}] {opt_text[:20]} -> {recheck2}", flush=True)
            if recheck2 == "正确":
                correct_letters.append(letter)
                wrong_letters.remove(letter)

    # 数量不足复查（T5）：多选题预测有 N 个答案，目前仅判出 M<N 个
    # 当前正确数 <= 2 时一律触发（多选答案数均值 2.5-3，过短大概率漏选）
    if is_multi and wrong_letters and 1 <= len(correct_letters) <= 2:
        expected = predict_answer_count(client, question, options)
        # floor: current<=2 时至少认为有 3 个答案
        threshold = max(expected, 3)
        if threshold > len(correct_letters):
            print(
                f"  [count_predict] predicted={expected} threshold={threshold} "
                f"current={len(correct_letters)} → recheck",
                flush=True,
            )
            for letter in list(wrong_letters):
                opt_text = dict(options)[letter]
                v3 = _recheck_count_mismatch(
                    retriever, client, stem, letter, opt_text,
                    correct_letters, threshold, options, image_path, is_vqa, topk,
                )
                print(f"  [recheck3 {letter}] {opt_text[:20]} -> {v3}", flush=True)
                if v3 == "正确":
                    correct_letters.append(letter)
                    wrong_letters.remove(letter)

    # 全部判错 → 回退常规推理
    if not correct_letters:
        fallback = _fallback_choice_pick(client, question, image_path, is_vqa)
        return (fallback or ""), []

    if is_multi:
        answer = "".join(sorted(correct_letters))
    elif len(correct_letters) == 1:
        answer = correct_letters[0]
    else:
        # 单选多候选 → 带 RAG 证据的对比裁决
        cands = [(l, dict(options)[l]) for l in correct_letters]
        answer = _single_contrast_pick(
            client, question, cands, ctx_per_letter, image_path, is_vqa,
        )

    # 答案反思：检测明显错误（漏选/错选）
    if answer:
        ok = _reflect_answer(
            retriever, client, question, section, answer,
            options, image_path, is_vqa, topk,
        )
        print(f"  [reflect] ans={answer} -> {'OK' if ok else 'RETRY'}", flush=True)
        if not ok and not is_multi:
            # 单选反思失败 → 兜底重选
            fallback = _fallback_choice_pick(client, question, image_path, is_vqa)
            if fallback and fallback != answer:
                answer = fallback

    return answer, []


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


def dmqr_build_image_query(
    client,
    question: str,
    image_path,
    section: str,
) -> str:
    """DMQR-RAG 两阶段图文对齐检索 query（思路1）。

    Phase A: 视觉特征抽取 (SYS_IMAGE_DESC) → img_desc
    Phase B: 图像特征 ⊕ 题干 → 陈述式检索 query (SYS_IMAGE_AS_QUERY)

    无图像或 Phase A/B 失败时退化为纯文本 rewrite_query。
    """
    if image_path is None:
        return rewrite_query(client, question, section=section)

    # Phase A：视觉特征抽取
    img_desc = describe_image(client, image_path, question)
    if not img_desc:
        return rewrite_query(client, question, section=section)

    # Phase B：图像特征 ⊕ 题干 → 陈述式检索句
    user = (
        f"【题目】{question}\n"
        f"【图像特征】{img_desc}\n"
        f"输出陈述句检索 query："
    )
    try:
        r = client.generate(SYS_IMAGE_AS_QUERY, user, temperature=0.0, max_tokens=60)
        s = (r or "").strip().splitlines()[0].strip()
        if s and 5 <= len(s) <= 100:
            return s
    except Exception:
        pass
    # Phase B 失败时退回拼接
    base = build_query(question, section=section)
    return f"{base} {img_desc}".strip()


def _ovd_extract_entities(client, question: str) -> list:
    """GranuRAG Phase 1（思路6）：从题干提取 1-3 个可视化实体词，用于元素级别检索增强。"""
    try:
        r = client.generate(
            SYS_OVD_ENTITY_EXTRACT,
            f"[题目] {question}\n实体词列表：",
            temperature=0.0, max_tokens=32,
        )
        lines = [(ln or "").strip() for ln in (r or "").splitlines() if ln.strip()]
        entities = [ln for ln in lines if ln and ln.upper() != "NA" and 1 <= len(ln) <= 8]
        return entities[:3]
    except Exception:
        return []


def _ovd_enhanced_retrieval(
    retriever,
    client,
    question: str,
    image_path,
    topk: int = 2,
) -> list:
    """GranuRAG 元素级别检索增强（思路6）：实体感知 multi-query 检索，补充场景级检索的盲区。"""
    entities = _ovd_extract_entities(client, question)
    if not entities:
        return []
    stem = build_query(question, section="简答VQA")[:30]
    entity_docs: list = []
    seen: set = set()
    for entity in entities:
        query = f"{stem} {entity}".strip()
        try:
            docs = retriever.search(
                text=query,
                image_path=str(image_path) if image_path else None,
                topk=topk,
            )
            for d in docs:
                if d not in seen:
                    seen.add(d)
                    entity_docs.append(d)
        except Exception:
            continue
    return entity_docs


# ============================================================================
# 简答 QA: 多 query 发散检索
# ============================================================================

_DIVERGE_SYS = "你是考古学信息检索助手。"
_DIVERGE_TMPL = (
    '为以下简答题生成 3 个不同的【陈述句】检索query，覆盖不同表达方式 / 相关实体。'
    '每行一个，不超过 35 字，不要编号、不要前缀。'
    '【重要】：不要生成含 A/B/C/D 选项格式的句子，不要生成问句，只输出陈述句。\n'
    '题目：{q}\n'
    '3 条 query：'
)


def diverge_queries(client, question: str) -> list[str]:
    """简答题发散：让 LLM 生成多个 query 变体，失败时只返回原始题干。"""
    try:
        r = client.generate(_DIVERGE_SYS, _DIVERGE_TMPL.format(q=question),
                            temperature=0.5, max_tokens=120)
        lines = [l.strip(' \t-•0123456789.、，。') for l in r.splitlines() if l.strip()]
        # 过滤掉选择题格式（含 A. B. C. D.）和问句
        import re as _re
        def _is_valid_query(s: str) -> bool:
            if not (5 <= len(s) <= 60):
                return False
            if _re.search(r'[A-D][.．、：]', s):  # 含选项格式
                return False
            if s.endswith('？') or s.endswith('?'):  # 问句
                return False
            return True
        return [l for l in lines if _is_valid_query(l)][:3]
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

def _process_one_item(
    i: int,
    item: dict,
    n_total: int,
    retriever,
    client,
    args,
    fout,
    ftime,
    write_lock: threading.Lock,
    done: set,
) -> None:
    idx = item.get("index")
    if idx in done:
        return
    section = str(item.get("section") or "")
    question = str(item.get("question") or "")
    image_url = str(item.get("image_url") or "")
    image_path = resolve_local_image(image_url) if image_url else None

    is_short_vqa = section == "简答VQA"
    is_choice = "选择" in section
    skip_retrieval = is_choice  # 简答VQA 现在也做轻量 RAG
    valid_options = extract_valid_options(question) if is_choice else None

    # ---- 分支 1: 选择题 ----
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
        with write_lock:
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
            f"[{i}/{n_total}] idx={idx} sec={section} "
            f"ans={answer!r} t_inf={infer_cost:.2f}s (choice-as-judge)",
            flush=True,
        )
        return

    # ---- 分支 2: 判断 / 简答 ----
    if section == "简答QA":
        short_circuit_ans = extract_answer_from_question(client, question)
        if short_circuit_ans:
            t0 = time.perf_counter()
            answer = pick_answer(short_circuit_ans, section)
            infer_cost = time.perf_counter() - t0
            with write_lock:
                fout.write(json.dumps(
                    {"index": idx, "answer": answer, "raw_answer": short_circuit_ans},
                    ensure_ascii=False,
                ) + "\n")
                fout.flush()
                ftime.write(json.dumps(
                    {"index": idx, "retrieval_cost": 0.0,
                     "infer_cost": round(infer_cost, 4)},
                    ensure_ascii=False,
                ) + "\n")
                ftime.flush()
            print(
                f"[{i}/{n_total}] idx={idx} sec={section} "
                f"ans={answer!r} (T2 short-circuit: 题干自含答案)",
                flush=True,
            )
            return

    # ---- 检索 ----
    t0 = time.perf_counter()
    contexts = []
    if not skip_retrieval:
        try:
            if section == "判断QA":
                query = build_query(question, section=section)
                contexts = retriever.search(text=query, topk=10)
                # 信息瓶颈式重排：过滤与题目无关的噪声 chunk
                contexts = llm_rerank_contexts(client, question, contexts, max_rerank=4)
                print(f"[idx={idx}] direct query={query[:60]} ctx={len(contexts)}", flush=True)
            elif section == "简答QA":
                base_q = rewrite_query(client, question, section=section)
                variants = diverge_queries(client, question)
                # HyDE Round 1：生成初始假设答案，扩展检索覆盖面
                hyde_q1 = _generate_hyde_answer(client, question)
                queries_r1 = [base_q] + [v for v in variants if v and v != base_q]
                if hyde_q1 and hyde_q1 not in queries_r1:
                    queries_r1.append(hyde_q1)
                seen: set = set()
                merged: List[str] = []
                for q in queries_r1[:5]:
                    try:
                        docs = retriever.search(text=q, topk=8)
                    except Exception:
                        docs = []
                    for d in docs:
                        if d not in seen:
                            seen.add(d)
                            merged.append(d)
                # HyDE Round 2：借助第一轮结果精化假设答案，命中更精确的文献
                ctx_r1 = _filter_noise(merged, keep_min=4)[:6]
                if ctx_r1:
                    ctx_preview = "\n".join(
                        f"[资料{i+1}] {c[:200]}" for i, c in enumerate(ctx_r1[:4])
                    )
                    hyde_q2 = _generate_hyde_answer_with_context(client, question, ctx_preview)
                    if hyde_q2 and hyde_q2 not in seen:
                        try:
                            for d in retriever.search(text=hyde_q2, topk=8):
                                if d not in seen:
                                    seen.add(d)
                                    merged.append(d)
                        except Exception:
                            pass
                merged = _filter_noise(merged, keep_min=4)
                merged = llm_rerank_contexts(client, question, merged, max_rerank=4)
                contexts = merged[:12]
                print(f"[idx={idx}] 简答QA iter-HyDE {len(queries_r1)}+1 queries, ctx={len(contexts)}", flush=True)
            elif is_short_vqa:
                # 简答VQA：视觉直读题（铭文/数数/方向）走轻量检索；其余走 DMQR
                if _is_visual_direct_read(question):
                    query = build_query(question, section=section)
                    contexts = retriever.search(
                        text=query,
                        image_path=str(image_path) if image_path else None,
                        topk=args.topk,
                    )
                    print(f"[idx={idx}] VQA(short/direct) query={query[:60]} ctx={len(contexts)}", flush=True)
                else:
                    # 知识先行：先用题干文本检索领域背景，再用 DMQR 补充图像相关文献
                    text_q = rewrite_query(client, question, section=section)
                    text_docs = retriever.search(
                        text=text_q,
                        image_path=None,   # 纯文本路径，得到干净的知识背景
                        topk=args.topk,
                    )
                    dmqr_q = dmqr_build_image_query(client, question, image_path, section)
                    img_docs = retriever.search(
                        text=dmqr_q,
                        image_path=str(image_path) if image_path else None,
                        topk=args.topk,
                    )
                    # 合并：文本路径优先，DMQR 补充去重
                    seen: set = set(text_docs)
                    merged_vqa: list = list(text_docs)
                    for d in img_docs:
                        if d not in seen:
                            seen.add(d)
                            merged_vqa.append(d)
                    contexts = _filter_noise(merged_vqa)[:args.topk + 2]
                    print(f"[idx={idx}] VQA(short/kfirst+dmqr) text_q={text_q[:40]} ctx={len(contexts)}", flush=True)
            else:
                # 判断VQA：知识先行检索（纯文本 query，避免 describe_image 幻觉污染）
                # 先用题干文字找到准确的领域知识背景，再让模型带着背景看图作判断
                query = build_query(question, section=section)
                contexts = retriever.search(
                    text=query,
                    image_path=None,   # 不传图像，避免器物图与文献页图视觉错配
                    topk=args.topk,
                )
                print(f"[idx={idx}] VQA(judge/kfirst) query={query[:60]} ctx={len(contexts)}", flush=True)
        except Exception as e:
            print(f"[idx={idx}] retrieval error: {e}", flush=True)
            contexts = []
    retrieval_cost = time.perf_counter() - t0

    # ---- Prompt ----
    # 判断题：使用软化 RAG（RAG 结果仅作参考，模型自身知识为主）
    if "判断" in section:
        system = SYS_JUDGE_SOFT_VQA if "VQA" in section else SYS_JUDGE_SOFT
        user = build_judge_user_soft(section, question, contexts)
    elif is_short_vqa:
        # 简答VQA：图像优先，RAG 仅作辅助背景参考
        vqa_ctx = [] if _is_visual_direct_read(question) else contexts
        system = system_prompt_for(section)
        user = build_short_vqa_user(question, vqa_ctx)
    elif section == "简答QA":
        # 简答QA：两阶段提取模式 —— 从文献中定位原文术语，而非凭空生成
        system = SYS_SHORT_EXTRACT
        user = build_extract_user_message(question, contexts)
    else:
        system = system_prompt_for(section)
        user = build_user_message(section, question, contexts)
    if not args.no_fewshot:
        if is_short_vqa and _is_inscription_char_recognition(question):
            pass
        else:
            user = fewshot_block(section) + user

    # ---- 推理 ----
    t1 = time.perf_counter()
    raw = ""
    try:
        if is_short_vqa and args.vqa_samples > 1:
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
            samples = client.generate_n(
                system, user, image_path,
                n=args.judge_samples,
                temperature=0.5,
                max_tokens=16,
            )
            voted = majority_vote(samples, section)
            raw = samples[0]
            answer = voted or pick_answer(samples[0], section)
        else:
            max_tok = 16 if "判断" in section else 64
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
    with write_lock:
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
        f"[{i}/{n_total}] idx={idx} sec={section} "
        f"ans={answer!r} t_ret={retrieval_cost:.2f}s t_inf={infer_cost:.2f}s "
        f"ctx={len(contexts)}",
        flush=True,
    )


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

    # 启动自检：验证 LLM / 检索 / 归一化在正式跑题前均可用
    run_startup_checks(client, retriever)

    print(f"[main] total items: {len(items)}", flush=True)

    # 文件以 append 打开，支持续跑
    open_mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(open_mode, encoding="utf-8")
    ftime = time_path.open(open_mode, encoding="utf-8")
    write_lock = threading.Lock()

    try:
        n_total = len(items)
        workers = max(1, getattr(args, "workers", 1))
        if workers == 1:
            for i, item in enumerate(items, 1):
                _process_one_item(i, item, n_total, retriever, client, args,
                                  fout, ftime, write_lock, done)
        else:
            print(f"[main] 并发模式: workers={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_one_item,
                        i, item, n_total, retriever, client, args,
                        fout, ftime, write_lock, done,
                    ): i
                    for i, item in enumerate(items, 1)
                }
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        print(f"[main] worker error: {exc}", flush=True)
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
