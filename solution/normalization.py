"""
solution/normalization.py
==========================
答案归一化 —— solution 自带版本，不依赖 src/normalization.py。

设计目标:
  1. 兼容 CoT 输出 —— "分析: ... \n 答案: X" 这种两段式格式
  2. 多个 "答案:" 出现时取最后一个 (CoT 推理过程可能先错再纠正)
  3. 选择题: 把模型瞎写的杂乱文本里抽出 A/B/C/D 字母序列
  4. 判断题: 去掉 "分析里同时出现正确/错误" 的歧义
  5. 简答题: 答案行 / 末行 / 首行三档兜底

设计原则:
  - 这是 solution 内部的"预清洗" —— 我们写进 predictions.jsonl 的 answer 字段
    必须是已经被这一层清干净的纯答案串 (A / ABD / 正确 / 死罪)
  - 评测脚本会用 src/normalization 再做一次 normalize,所以即便我们这里漏掉,
    评测端的归一化也会做"最低保底"的清洗 —— 但是评测端的版本简陋,不抗 CoT
  - 因此本 solution 的清洗必须比 src/ 的版本更强
"""
from __future__ import annotations

import re
import unicodedata


# ============================================================================
# 入口
# ============================================================================

def normalize_answer(section: str, text: str) -> str:
    if "选择" in section:
        return normalize_choice_answer(text)
    if "判断" in section:
        return normalize_judgment_answer(text)
    return normalize_short_answer(text)


# ============================================================================
# 选择题: 支持 CoT 输出, 优先取最后一个 "答案/最终答案/Label" 行
# ============================================================================

_CHOICE_LETTER = r"[A-HＡ-Ｈ]"
_CHOICE_SEQ = rf"{_CHOICE_LETTER}(?:\s*[、,，/和与]?\s*{_CHOICE_LETTER})*"


def normalize_choice_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    # 1) 强信号: "最终答案/答案/Label: X" -- 用 findall + 取最后一个
    #    (CoT 输出的最终答案通常在末尾;前面"选项 A 不对"是噪声)
    strong = re.findall(
        rf"(?:最终答案|答案\s*行|答案|Label)\s*[:：]?\s*({_CHOICE_SEQ})",
        cleaned,
        re.IGNORECASE,
    )
    if strong:
        result = normalize_choice_letters(strong[-1])
        if result:
            return result

    # 2) 弱信号: "选项/选择 X" -- 同样取最后一个
    loose = re.findall(
        rf"(?:选项|选择)\s*[:：]?\s*({_CHOICE_SEQ})",
        cleaned,
        re.IGNORECASE,
    )
    if loose:
        result = normalize_choice_letters(loose[-1])
        if result:
            return result

    # 3) 全文最后一行如果就是字母,直接拿
    last_line = cleaned.splitlines()[-1].strip()
    last_only = re.match(
        rf"^\s*({_CHOICE_SEQ})\s*[.。!?！？]?\s*$",
        last_line,
        re.IGNORECASE,
    )
    if last_only:
        result = normalize_choice_letters(last_only.group(1))
        if result:
            return result

    # 4) 首行就是字母 (保留旧行为)
    leading = re.match(
        rf"^\s*({_CHOICE_SEQ})(?:[.。:：、\s]|$)",
        cleaned,
        re.IGNORECASE,
    )
    if leading:
        result = normalize_choice_letters(leading.group(1))
        if result:
            return result

    return ""


# ============================================================================
# 判断题: 取最后一次 "答案: 正确/错误"
# ============================================================================

def normalize_judgment_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    # 1) 显式答案行 -- 取最后一个
    matches = re.findall(
        r"(?:最终答案|答案|结论|判断|Label)\s*[:：]?\s*(正确|错误)",
        cleaned,
    )
    if matches:
        return matches[-1]

    # 2) 全文只出现一种判断词
    has_correct = "正确" in cleaned
    has_wrong = "错误" in cleaned
    if has_correct and not has_wrong:
        return "正确"
    if has_wrong and not has_correct:
        return "错误"

    # 3) 取最后一行的判断词
    last_line = cleaned.splitlines()[-1].strip()
    if last_line in ("正确", "错误"):
        return last_line

    # 4) 末尾词
    tail = re.search(r"(正确|错误)[.。!?！？\s]*$", cleaned)
    if tail:
        return tail.group(1)

    return cleaned if cleaned in {"正确", "错误"} else ""


# ============================================================================
# 简答题: 取 "答案:" 后的内容 / 末行 / 首行
# ============================================================================

# 需要 strip 的引号类字符（含 Unicode 弯引号，用转义避免源文件编码歧义）
_QUOTE_CHARS = (
    " \t\r\n\"'"
    "“”"   # " "  左右双弯引号
    "‘’"   # ' '  左右单弯引号
    "《》"   # 《》
    "<>"
    "（）"   # （）全角括号
    "()"
    "[]"
    "【】"   # 【】
    "{}"
)


def normalize_short_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    # 1) 优先 "答案: ..." (取最后一处,CoT 安全)
    explicits = re.findall(r"(?:最终答案|答案|答)\s*(?:是|为)?\s*[:：]\s*([^\n]+)", cleaned)
    if explicits:
        cleaned = explicits[-1].strip()
    else:
        # 2) 否则取最后一行(短答题模型本来就该一行作答)
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if not lines:
            return ""
        cleaned = lines[-1] if len(lines[-1]) <= 20 else lines[0]

    # 切到第一个分隔符为止
    cleaned = re.split(r"[。；;]\s*", cleaned)[0].strip()
    cleaned = re.split(r"(?:依据|理由)\s*[:：]|因为", cleaned)[0].strip()
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"^(?:答案|最终答案|答)(?:是|为)?[:：]?", "", cleaned)
    cleaned = cleaned.strip(_QUOTE_CHARS)
    cleaned = cleaned.rstrip(".,;:!?,。;:!?")
    # 方向词规范化：视觉简答题输出冗余前缀（向上→上 等），仅处理无歧义格式问题
    _DIR_MAP = {"向上": "上", "向下": "下", "向左": "左", "向右": "右"}
    if cleaned in _DIR_MAP:
        cleaned = _DIR_MAP[cleaned]
    return cleaned


# ============================================================================
# 工具
# ============================================================================

def normalize_choice_letters(text: str) -> str:
    letters = []
    full_width = "ＡＢＣＤＥＦＧＨ"
    for match in re.finditer(r"[A-HＡ-Ｈ]", str(text), re.IGNORECASE):
        letter = match.group(0).upper()
        if letter in full_width:
            letter = chr(ord("A") + full_width.index(letter))
        letters.append(letter)
    return "".join(sorted(set(letters)))
