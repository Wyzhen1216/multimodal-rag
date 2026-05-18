"""
Prompt 模板 —— 按 6 种题型分支，强约束输出格式。

本版本对前一版的关键修正：
  1. Few-shot 示例**不再**来自公开样例（避免答案泄漏）。
     改用完全合成的题（题目与公开/隐藏题库无重叠），仅用于演示输出格式。
  2. 判断题先验**移除**（"不确定时倾向正确"删掉）。
     原因：模型对细颗粒度考古事实没有真实判断力，先验只会让分数跟随
     公开样例的 gold 分布（判断VQA 公开样例 8/9 是"正确"，先验直接白嫖 88.9%）。
  3. 选择题"倾向多选"先验**移除**，避免单选题被错答成多选。

闭卷下指望模型答对历史专业事实是不现实的。真正的精度提升必须依赖
DashVector 检索（评测环境用本地 GME-Qwen2-VL）。
"""
from __future__ import annotations

from typing import List, Optional


# ============================================================================
# System prompts —— 6 种题型，仅做格式约束，不灌先验
# ============================================================================

SYS_CHOICE = (
    "你是一名多模态问答助手。请根据题目和检索到的资料回答选择题。\n"
    "输出格式硬性要求：\n"
    "  1. 仅输出选项字母 A/B/C/D（必要时 E/F/G/H）\n"
    "  2. 单选题输出 1 个字母；多选题连写多个字母无分隔符（例如：A、AB、ABCD）\n"
    "  3. 单/多选请根据题干语义判断，不要默认偏向某一种\n"
    "  4. 不要输出 '答案'、'我选'、'选'、'是' 等前缀\n"
    "  5. 不要任何解释、推理过程、句号、引号\n"
    "  6. 第一行就是最终答案，且仅含字母"
)

SYS_JUDGE = (
    "你是一名多模态问答助手。请判断给定陈述是否符合事实/资料。\n"
    "输出格式硬性要求：\n"
    "  1. 仅输出 '正确' 或 '错误' 两字之一\n"
    "  2. 整段输出中不能同时出现 '正确' 和 '错误' 两个词\n"
    "  3. 不要输出 '答案'、'是'、'判断' 等前缀\n"
    "  4. 不要任何解释\n"
    "判断时严格依据检索到的资料；若资料不充分，请凭对题目的客观理解判断，"
    "不要倾向任何一边。"
)

SYS_SHORT = (
    "你是一名多模态问答助手。请输出最简短的答案。\n"
    "输出格式硬性要求：\n"
    "  1. 答案不超过 8 个汉字 / 1 个短语\n"
    "  2. 数字答案只输出数字本身，例如 '2' 而不是 '2个' '2 处'\n"
    "  3. 时期答案只输出朝代名称，例如 '春秋早期' 而不是 '该器物始于春秋早期'\n"
    "  4. 字符答案只输出单字本身，例如 '王' 而不是 '王字'\n"
    "  5. 不要 '答案：' '答：' '为' '是' 等前缀\n"
    "  6. 不要句号、引号、括号\n"
    "  7. 不要解释、不要换行，第一行就是最终答案"
)

SYS_CHOICE_VQA = SYS_CHOICE + "\n（注意：题目附有图片，请结合图中细节判断。）"
SYS_JUDGE_VQA = SYS_JUDGE + "\n（注意：题目附有图片，请结合图中细节判断。）"
SYS_SHORT_VQA = (
    SYS_SHORT
    + "\n（注意：本题为视觉简答题，绝大多数答案直接从图中观察得出 ——"
    "识字、数数、判断方向、识别物体。请优先看图。）"
)


def system_prompt_for(section: str) -> str:
    if "选择" in section:
        return SYS_CHOICE_VQA if "VQA" in section else SYS_CHOICE
    if "判断" in section:
        return SYS_JUDGE_VQA if "VQA" in section else SYS_JUDGE
    return SYS_SHORT_VQA if "VQA" in section else SYS_SHORT


# ============================================================================
# 检索结果格式化
# ============================================================================

def _format_contexts(contexts: List[str], max_chars_each: int = 400) -> str:
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
    header = (
        "以下是从知识库检索到的可能相关的历史文献片段，"
        "请结合资料回答问题（资料可能不完全相关，请独立判断）：\n"
    )
    return header + "\n".join(blocks) + "\n\n"


def build_user_message(
    section: str,
    question: str,
    contexts: Optional[List[str]] = None,
) -> str:
    contexts = contexts or []
    ctx_block = _format_contexts(contexts)

    if "选择" in section:
        tail = "请直接输出选项字母（单选 1 个或多选连写多个，无分隔符）："
    elif "判断" in section:
        tail = "请直接输出 '正确' 或 '错误'："
    else:
        tail = "请直接输出最简短答案（≤8 字、无前缀、无标点）："

    return f"{ctx_block}[题目] {question.strip()}\n\n{tail}"


# ============================================================================
# Few-shot 示例 —— 全部合成，题目与公开/隐藏题库无重叠
# 目的：仅演示输出格式，不传授知识
# ============================================================================

# 合成选择题：用通用常识题（不涉及中国考古学专业知识），且这类题不会出现在题库中
FEWSHOT_CHOICE = [
    # 单选演示：1 个字母
    ("以下哪种颜色由红色和黄色混合而成？A. 紫色 B. 橙色 C. 绿色 D. 棕色", "B"),
    # 多选演示：多个字母无分隔符
    ("以下哪些是哺乳动物？A. 鲸鱼 B. 鳄鱼 C. 蝙蝠 D. 海豚", "ACD"),
]

# 合成判断题：极其明显的常识，避免引入考古学先验
FEWSHOT_JUDGE = [
    ("水在标准大气压下沸点为 100 摄氏度。", "正确"),
    ("太阳从西边升起。", "错误"),
]

# 合成简答题：极其明显的常识，演示"无前缀无量词"的格式
FEWSHOT_SHORT = [
    ("一年有多少个月？", "12"),
    ("地球的天然卫星叫什么？", "月球"),
]


def fewshot_block(section: str) -> str:
    if "选择" in section:
        pairs = FEWSHOT_CHOICE
        tail = "请直接输出选项字母："
    elif "判断" in section:
        pairs = FEWSHOT_JUDGE
        tail = "请直接输出 '正确' 或 '错误'："
    else:
        pairs = FEWSHOT_SHORT
        tail = "请直接输出最简短答案："
    lines = ["以下是输出格式演示（仅供格式参考，与本题无关）："]
    for q, a in pairs:
        lines.append(f"  题目：{q}\n  {tail}{a}")
    return "\n".join(lines) + "\n\n"


# ============================================================================
# 兜底默认值 —— 归一化失败时返回
# 注意：兜底是为了避免"答案为空"的合规失败，不应承担"调先验"的功能
# ============================================================================

def fallback_answer(section: str) -> str:
    if "选择" in section:
        return "A"
    if "判断" in section:
        return "正确"
    return "无"
