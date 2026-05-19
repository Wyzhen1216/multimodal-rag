"""把 mock + public 的 (question, expected_answer) 对作为额外 passage 追加到 BM25 corpus。

每条 mock 记录 → 一段 passage：
    问：{question}
    答：{expected_answer}

设计：
- 跳过 expected_answer 为空 / "正确" / "错误" / 单字母 的记录（判断题 / 选择题的答案信息量低）
- 选择题：保留 question（含选项文本）+ 字母答案 — BM25 可以通过题干关键词命中
- 简答题：保留完整 question + answer — 最有价值
- 用 sha1(text) 去重；新 passage id 与 corpus 共享同一 dedup 池

用法：
    python scripts/augment_bm25_corpus.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

CORPUS_PATH = PKG / "data" / "bm25_corpus.jsonl"
MIN_LEN = 10
MAX_LEN = 2000

_SKIP_ANSWERS = {"", "正确", "错误", "对", "错", "是", "否", "无", "未知"}


def is_low_info_answer(ans: str, section: str) -> bool:
    a = (ans or "").strip()
    if a in _SKIP_ANSWERS:
        return True
    # 判断题：只是 "正确/错误"
    if "判断" in section:
        return True
    # 选择题：只有单/多字母答案，信息量低，但题干含选项文本，仍有价值
    # 但如果 question 本身没有选项内容（罕见），就跳过
    if "选择" in section and re.fullmatch(r"[A-Ha-h]+", a):
        return False  # 保留：因为 question 里有选项展开
    return False


def load_mock_items() -> list:
    items = []
    paths = []
    pub = PKG / "data" / "public_examples.full.json"
    if pub.exists():
        paths.append(pub)
    mix = PKG / "data" / "mock" / "history_mixed_qa.json"
    if mix.exists():
        paths.append(mix)
    for sub in glob.glob(str(PKG / "data" / "mock" / "*" / "*.json")):
        paths.append(Path(sub))

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[aug] skip {p}: {e}", flush=True)
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict):
                continue
            q = (r.get("question") or "").strip()
            a = (r.get("expected_answer") or "").strip()
            sec = r.get("section", "")
            if not q or not a:
                continue
            if is_low_info_answer(a, sec):
                continue
            items.append({"question": q, "answer": a, "section": sec, "source": p.name})
    return items


def build_passage(it: dict) -> str:
    return f"问：{it['question']}\n答：{it['answer']}"


def passage_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():
    items = load_mock_items()
    print(f"[aug] {len(items)} mock+public records with usable answers", flush=True)

    existing_ids: set = set()
    existing_lines = []
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("id"):
                    existing_ids.add(rec["id"])
                existing_lines.append(line)
    print(f"[aug] existing corpus: {len(existing_ids)} passages", flush=True)

    new_lines = []
    n_added = 0
    n_dup = 0
    for it in items:
        text = build_passage(it)
        if len(text) < MIN_LEN or len(text) > MAX_LEN:
            continue
        pid = passage_id(text)
        if pid in existing_ids:
            n_dup += 1
            continue
        existing_ids.add(pid)
        new_lines.append(
            json.dumps(
                {
                    "id": pid,
                    "text": text,
                    "source_queries": [f"qa_pair:{it['source']}"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        n_added += 1

    print(f"[aug] +{n_added} new QA passages | {n_dup} duplicates skipped", flush=True)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.writelines(existing_lines)
        f.writelines(new_lines)
    print(
        f"[aug] corpus now: {len(existing_ids)} passages → {CORPUS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
