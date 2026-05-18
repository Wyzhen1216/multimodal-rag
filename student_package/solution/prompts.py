"""
Prompt 模板 —— 按 6 种题型分支，强约束输出格式。

设计要点（依据 data/solution_diagnosis.md 的错例分析）：
  - 选择题：明确"可能多选"，禁止前缀、禁止解释
  - 判断题：修正"保守偏负"，资料支持时倾向"正确"，禁止同时出现两个词
  - 简答题：禁止量词后缀、禁止前缀、第一行即答案、控制长度
"""
from __future__ import annotations

from typing import List, Optional


# ============================================================================
# System prompts —— 6 种题型一一对应
# ============================================================================

SYS_CHOICE = (
    "你是一名中国考古学与古文字学方向的多模态问答助手。"
    "你正在做一道选择题（可能是单选，也可能是多选，请仔细判断）。\n"
    "输出格式硬性要求：\n"
    "  1. 仅输出大写字母 A/B/C/D（必要时 E/F/G/H），多选连写无分隔符，例如：A、AB、ABCD\n"
    "  2. 不要输出 '答案'、'我选'、'选'、'是' 等任何前缀\n"
    "  3. 不要任何解释、推理过程、句号、引号\n"
    "  4. 第一行就是最终答案，且仅有字母\n"
    "选择倾向：若检索资料同时支持多个选项，请一并选择；"
    "题干用'哪些''包括''属于哪些'等复数提问时，倾向多选。"
)

SYS_JUDGE = (
    "你是一名中国考古学与古文字学方向的多模态问答助手。"
    "你正在做一道判断题，请判断给定陈述是否成立。\n"
    "输出格式硬性要求：\n"
    "  1. 仅输出 '正确' 或 '错误' 两个字之一\n"
    "  2. 整段输出中不能同时出现 '正确' 和 '错误' 两个词（包括解释里）\n"
    "  3. 不要输出 '答案'、'是'、'判断' 等前缀\n"
    "  4. 不要任何解释\n"
    "判断原则（重要）：\n"
    "  - 若检索资料明确支持该陈述 → 输出 '正确'\n"
    "  - 若检索资料明确否定 / 与陈述矛盾 → 输出 '错误'\n"
    "  - 若资料不充分或语义模糊 → 默认输出 '正确'（不要因为小瑕疵就判错）"
)

SYS_SHORT = (
    "你是一名中国考古学与古文字学方向的多模态问答助手。"
    "你正在做一道填空式简答题，请输出最简短的答案。\n"
    "输出格式硬性要求：\n"
    "  1. 答案不超过 8 个汉字 / 1 个短语\n"
    "  2. 数字答案只输出数字本身，例如 '2' 而不是 '2个' '2 处'\n"
    "  3. 时期答案只输出朝代名称，例如 '战国中晚期' 而不是 '该城始筑于战国中晚期'\n"
    "  4. 字符答案只输出单字本身，例如 '王' 而不是 '王字'\n"
    "  5. 不要 '答案：' '答：' '为' '是' 等任何前缀\n"
    "  6. 不要句号、引号、括号\n"
    "  7. 不要解释、不要换行，第一行就是最终答案"
)

SYS_CHOICE_VQA = SYS_CHOICE + "\n（注意：题目附有图片，请优先观察图中细节再结合检索资料判断。）"
SYS_JUDGE_VQA = SYS_JUDGE + "\n（注意：题目附有图片，请优先观察图中细节再结合检索资料判断。）"
SYS_SHORT_VQA = (
    SYS_SHORT
    + "\n（注意：本题为视觉简答题，绝大多数答案直接从图中观察得出 —— "
    "识字、数数、判断方向、识别物体。请优先看图，资料仅作辅助。）"
)


def system_prompt_for(section: str) -> str:
    """根据 section 返回对应的 system prompt。"""
    if "选择" in section:
        return SYS_CHOICE_VQA if "VQA" in section else SYS_CHOICE
    if "判断" in section:
        return SYS_JUDGE_VQA if "VQA" in section else SYS_JUDGE
    return SYS_SHORT_VQA if "VQA" in section else SYS_SHORT


# ============================================================================
# User message builders
# ============================================================================

def _format_contexts(contexts: List[str], max_chars_each: int = 400) -> str:
    """把检索文献片段格式化进 prompt。已截断、去 expected_answer。"""
    if not contexts:
        return ""
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        text = (ctx or "").strip()
        if not text:
            continue
        if len(text) > max_chars_each:
            text = text[:max_chars_each] + "…"
        blocks.append(f"[资料{i}] {text}")
    if not blocks:
        return ""
    header = "以下是从知识库检索到的可能相关的历史文献片段，请结合资料回答问题（资料可能不完全相关，请独立判断）：\n"
    return header + "\n".join(blocks) + "\n\n"


def build_user_message(
    section: str,
    question: str,
    contexts: Optional[List[str]] = None,
) -> str:
    """统一的 user message 构造器。"""
    contexts = contexts or []
    ctx_block = _format_contexts(contexts)

    # 题型尾部指令（再次强化，避免长 prompt 后模型遗忘）
    if "选择" in section:
        tail = "请直接输出选项字母（单选 1 个或多选连写多个，无分隔符）："
    elif "判断" in section:
        tail = "请直接输出 '正确' 或 '错误'："
    else:
        tail = "请直接输出最简短答案（≤8 字、无前缀、无标点）："

    return f"{ctx_block}[题目] {question.strip()}\n\n{tail}"


# ============================================================================
# Few-shot demonstrations (题型固定，从公开样例的【题干部分】抽取，
# 不读 expected_answer 字段，仅用人工标注的简短示例)
# ============================================================================

FEWSHOT_CHOICE = [
    ("商代墓葬中常见的腰坑习俗通常放置什么动物牺牲？A.牛 B.马 C.狗 D.猪", "C"),
    ("先秦时期衅浴祓除的巫术方法包括哪些？A. 用牲血祭祀 B. 用清水沐浴 C. 焚烧活人 D. 舞蹈求雨", "AB"),
]

FEWSHOT_JUDGE = [
    ("秦汉时期，令的实质作用是作为对律的补充、修订或说明。", "正确"),
    ("龙山时代的聚落群结构表明当时社会已进入国家社会阶段。", "错误"),
]

FEWSHOT_SHORT = [
    ("辉县市古共城南城墙的始筑年代是什么时期？", "战国中晚期"),
    ("江西收集的西汉铜钾的器物名称是什么？", "铜钾"),
]


def fewshot_block(section: str) -> str:
    """少量 few-shot 示例，仅文本形式，帮助模型对齐输出格式。"""
    if "选择" in section:
        pairs = FEWSHOT_CHOICE
        tail = "请直接输出选项字母："
    elif "判断" in section:
        pairs = FEWSHOT_JUDGE
        tail = "请直接输出 '正确' 或 '错误'："
    else:
        pairs = FEWSHOT_SHORT
        tail = "请直接输出最简短答案："
    lines = ["以下是该题型的标准输出示例（仅供格式参考）："]
    for q, a in pairs:
        lines.append(f"  题目：{q}\n  {tail}{a}")
    return "\n".join(lines) + "\n\n"


# ============================================================================
# 兜底默认值 —— 归一化失败时返回
# ============================================================================

def fallback_answer(section: str) -> str:
    if "选择" in section:
        return "A"
    if "判断" in section:
        return "正确"  # 基于公开样例先验：判断题"正确"占比略高
    return "无"
