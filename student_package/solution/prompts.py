"""
Prompt 构造: baseline (无检索) 与 rag (带检索 few-shot)。
"""
from __future__ import annotations

from typing import List


SYSTEM_BASE = (
    "你是一名历史学领域的多模态问答助手。请根据问题(可能附带图片)给出准确答案。\n"
    "答案格式约定:\n"
    "  - 选择题(QA/VQA): 仅输出选项字母(单选: A; 多选: 无分隔符如 AB / ABCD)\n"
    "  - 判断题(QA/VQA): 仅输出 '正确' 或 '错误'\n"
    "  - 简答题(QA/VQA): 输出简短答案文本, 无解释、无前缀\n"
    "请直接以 '答案: <答案>' 这一行结束你的输出。"
)


def section_hint(section: str) -> str:
    if "选择" in section:
        return "题型: 选择题。请仅输出选项字母 (例如 A 或 ABD)。"
    if "判断" in section:
        return "题型: 判断题。请仅输出 '正确' 或 '错误'。"
    return "题型: 简答题。请输出简短答案, 不要解释。"


def build_baseline_user(item: dict) -> str:
    section = str(item.get("section") or "")
    question = str(item.get("question") or "")
    return f"{section_hint(section)}\n\n问题:\n{question}\n\n请输出: 答案: <你的答案>"


def build_rag_user(item: dict, retrieved: List[dict]) -> str:
    """RAG: 用检索到的相似历史题目+答案作为 few-shot demonstrations。"""
    section = str(item.get("section") or "")
    question = str(item.get("question") or "")
    blocks: List[str] = [section_hint(section), ""]
    if retrieved:
        blocks.append("以下是若干检索到的相似历史题目作为参考(可参考其题型与思路, 但请独立判断本题答案):")
        for i, r in enumerate(retrieved, 1):
            r_sec = str(r.get("section") or "")
            r_q = str(r.get("question") or "")
            r_a = str(r.get("expected_answer") or "")
            blocks.append(f"[参考{i}] ({r_sec}) 题目: {r_q}\n          参考答案: {r_a}")
        blocks.append("")
    blocks.append(f"现在请回答下面这道题。\n问题:\n{question}\n\n请输出: 答案: <你的答案>")
    return "\n".join(blocks)
