from __future__ import annotations

import re
import unicodedata


def normalize_answer(section: str, text: str) -> str:
    if "选择" in section:
        return normalize_choice_answer(text)
    if "判断" in section:
        return normalize_judgment_answer(text)
    return normalize_short_answer(text)


def normalize_choice_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    explicit = re.search(
        r"(?:答案|选项|选择|最终答案)\s*[:：]?\s*([A-HＡ-Ｈ](?:\s*[、,，/和与]?\s*[A-HＡ-Ｈ])*)",
        cleaned,
        re.IGNORECASE,
    )
    if explicit:
        return normalize_choice_letters(explicit.group(1))
    leading = re.match(
        r"^\s*([A-HＡ-Ｈ](?:\s*[、,，/和与]?\s*[A-HＡ-Ｈ])*)(?:[.。:：、\s]|$)",
        cleaned,
        re.IGNORECASE,
    )
    return normalize_choice_letters(leading.group(1)) if leading else ""


def normalize_judgment_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    explicit = re.search(r"(?:答案|结论|判断|最终答案)\s*[:：]?\s*(正确|错误)", cleaned)
    if explicit:
        return explicit.group(1)
    has_correct = "正确" in cleaned
    has_wrong = "错误" in cleaned
    if has_correct and not has_wrong:
        return "正确"
    if has_wrong and not has_correct:
        return "错误"
    return cleaned if cleaned in {"正确", "错误"} else ""


def normalize_short_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    explicit = re.search(r"(?:答案|最终答案|答)\s*(?:是|为)?\s*[:：]\s*(.+)", cleaned)
    if explicit:
        cleaned = explicit.group(1).strip()
    first_line = cleaned.splitlines()[0].strip()
    first_line = re.split(r"[。；;]\s*", first_line)[0].strip()
    first_line = re.split(r"(?:依据|理由)\s*[:：]|因为", first_line)[0].strip()
    first_line = unicodedata.normalize("NFKC", first_line)
    first_line = re.sub(r"\s+", "", first_line)
    first_line = re.sub(r"^(?:答案|最终答案|答)(?:是|为)?[:：]?", "", first_line)
    first_line = first_line.strip(" \t\r\n\"'“”‘’《》<>（）()[]【】{}")
    return first_line.rstrip(".,;:!?，。；：！？")


def normalize_choice_letters(text: str) -> str:
    letters = []
    full_width = "ＡＢＣＤＥＦＧＨ"
    for match in re.finditer(r"[A-HＡ-Ｈ]", str(text), re.IGNORECASE):
        letter = match.group(0).upper()
        if letter in full_width:
            letter = chr(ord("A") + full_width.index(letter))
        letters.append(letter)
    return "".join(sorted(set(letters)))

