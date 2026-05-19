"""
split_mock_dataset.py
=====================
把混合题型的 mock JSON（如 history_mixed_qa.json）按 section 拆分到
data/mock/<section_dir>/ 下，方便 test/run_mock_test.py 自动发现。

section 中文名 → 目录映射:
    选择QA   → SelectQA
    选择VQA  → SelectVQA
    简答QA   → AnswerQA
    简答VQA  → AnswerVQA

输出文件命名: <input_stem>_<section_dir>.json
  - 同一输入拆出多个 section 时不冲突
  - 不同输入拆到同一 section 时也不会互相覆盖（按输入文件名区分）

索引冲突提示:
  - 官方 public_examples 用 < 10000 的 index
  - 现有 augmented mock 已使用 10000+
  - 若新混合文件 index 落在已占区间，建议加 --index-offset

用法:
    python scripts/split_mock_dataset.py data/mock/history_mixed_qa.json
    python scripts/split_mock_dataset.py data/mock/foo.json --index-offset 100000
    python scripts/split_mock_dataset.py data/mock/foo.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


SECTION_TO_DIR = {
    "选择QA": "SelectQA",
    "选择VQA": "SelectVQA",
    "简答QA": "AnswerQA",
    "简答VQA": "AnswerVQA",
}

REQUIRED_KEYS = {"index", "section", "question", "expected_answer"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按 section 拆分 mock 混合题型 JSON")
    p.add_argument("input", help="输入 JSON 数组文件")
    p.add_argument(
        "--out-root",
        default="",
        help="输出根目录（默认: 输入文件所在目录）",
    )
    p.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="为每条记录的 index 加偏移，避免与官方/既有 mock 冲突（0 = 不变）",
    )
    p.add_argument(
        "--name-prefix",
        default="",
        help="输出文件名前缀（默认: 取输入文件 stem）",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印分布，不写文件")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")
    return p.parse_args()


def _load_array(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入必须是 JSON 数组")
    return data


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"FATAL: 输入不存在 {in_path}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve() if args.out_root else in_path.parent
    prefix = args.name_prefix or in_path.stem

    print(f"[split] 读入 {in_path}")
    try:
        data = _load_array(in_path)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    print(f"[split] 总记录数: {len(data)}")

    buckets: dict[str, list[dict]] = {}
    unknown: list[str] = []
    missing_keys: list[int] = []

    for i, r in enumerate(data):
        if not isinstance(r, dict):
            missing_keys.append(i)
            continue
        miss = REQUIRED_KEYS - r.keys()
        if miss:
            print(f"[split] 第 {i} 条缺字段 {miss}，跳过", file=sys.stderr)
            missing_keys.append(i)
            continue
        sec = r.get("section") or "?"
        if sec not in SECTION_TO_DIR:
            unknown.append(sec)
            continue
        if args.index_offset and isinstance(r.get("index"), int):
            r = {**r, "index": r["index"] + args.index_offset}
        buckets.setdefault(sec, []).append(r)

    if unknown:
        print(f"[split] 未知 section（跳过）: {dict(Counter(unknown))}", file=sys.stderr)
    if missing_keys:
        print(f"[split] 缺字段记录数: {len(missing_keys)}", file=sys.stderr)

    print("[split] 分布:")
    for sec in sorted(buckets):
        print(f"          {sec:8s} -> {SECTION_TO_DIR[sec]:10s}  {len(buckets[sec]):4d} 条")

    # 索引区间提示
    for sec, recs in buckets.items():
        idxs = [r["index"] for r in recs if isinstance(r.get("index"), int)]
        if idxs:
            lo, hi = min(idxs), max(idxs)
            warn = " ⚠️  与官方区间 (<10000) 重叠" if lo < 10000 else ""
            print(f"          {SECTION_TO_DIR[sec]:10s}  index 范围 {lo} ~ {hi}{warn}")

    if args.dry_run:
        print("[split] dry-run，不写文件")
        return 0

    written: list[Path] = []
    for sec, recs in buckets.items():
        sub = out_root / SECTION_TO_DIR[sec]
        sub.mkdir(parents=True, exist_ok=True)
        out_file = sub / f"{prefix}_{SECTION_TO_DIR[sec]}.json"
        if out_file.exists() and not args.overwrite:
            print(f"[split] SKIP (已存在，加 --overwrite 覆盖): {out_file}")
            continue
        out_file.write_text(
            json.dumps(recs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        size_mb = out_file.stat().st_size / 1024 / 1024
        written.append(out_file)
        print(f"[split] 写入 {out_file}  ({len(recs)} 条, {size_mb:.1f} MB)")

    print(f"[split] 完成，共 {len(written)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
