"""
run_mock_test.py
================
泛化性测试：跑 data/mock/ 目录下的扩展测试集（非官方 50 题），
验证 solution/main.py 在分布外样本上的表现。

mock 目录结构（按题型分子目录）：
    data/mock/SelectQA/<任意>.json     ← 已有 323 条
    data/mock/SelectVQA/<任意>.json    ← 空，未来添加
    data/mock/AnswerQA/<任意>.json     ← 空
    data/mock/AnswerVQA/<任意>.json    ← 空

每个 JSON 文件需为数组，格式与 public_examples.full.json 一致：
    {"index", "section", "question", "image_url", "expected_answer"}

用法:
    # 全量 (默认 enhanced)
    python test/run_mock_test.py

    # 只跑 50 条随机样本，快验
    python test/run_mock_test.py --sample 50 --seed 42

    # 只跑某题型
    python test/run_mock_test.py --section 选择QA --sample 30

    # baseline 对比
    python test/run_mock_test.py --mode baseline --sample 50 --seed 42

注意：mock 索引从 10000 起，与官方测试集不冲突；evaluate.py / error_analysis
仍按 section 自动分组，无需改动。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
OUTPUT_ROOT = HERE / "outputs"
DEFAULT_MOCK_ROOT = PKG_DIR / "data" / "mock"

# 与官方 50 题对齐的基线（用于打印泛化 gap）
OFFICIAL_BASELINE_ACC = 0.78  # 当前 Round 6 = 39/50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mock 数据集泛化性测试执行器")
    p.add_argument(
        "--mode",
        choices=("baseline", "enhanced"),
        default="enhanced",
        help="baseline = 仅题型分支 prompt; enhanced = + few-shot + VQA 投票",
    )
    p.add_argument(
        "--mock-root",
        default=str(DEFAULT_MOCK_ROOT),
        help=f"mock 数据集根目录（默认: {DEFAULT_MOCK_ROOT}）",
    )
    p.add_argument(
        "--section",
        default="",
        help="只跑指定 section（如 '选择QA'），留空跑全部题型",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="随机抽 N 条（与 --limit 互斥；0 表示不抽样）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="--sample 的随机种子（默认 42，保证可复现）",
    )
    p.add_argument("--limit", type=int, default=0, help="顺序取前 N 条")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数（传给 solution.main --workers）。本地模型下默认 1（GPU 争用）",
    )
    p.add_argument(
        "--llm",
        choices=("auto", "api", "local"),
        default="local",
        help="LLM 模式：local=本地 Qwen2.5-VL（默认），api=DashScope API",
    )
    p.add_argument(
        "--vqa-samples",
        type=int,
        default=0,
        help="覆盖默认：0 表示按 mode 决定",
    )
    p.add_argument("--tag", default="", help="输出目录后缀标签")
    return p.parse_args()


def _load_json_array(p: Path) -> list[dict]:
    return json.loads(p.read_text(encoding="utf-8"))


def discover_mock_records(mock_root: Path, section_filter: str = "") -> tuple[list[dict], list[str]]:
    """扫描 mock_root 下所有子目录里的 *.json，合并为单一记录列表。

    返回 (records, source_log)。每条记录会带 `_source` 字段（仅用于日志）。
    """
    if not mock_root.exists():
        raise FileNotFoundError(f"mock 数据根目录不存在: {mock_root}")

    all_records: list[dict] = []
    source_log: list[str] = []

    # 按子目录名排序，确保稳定
    subdirs = sorted(d for d in mock_root.iterdir() if d.is_dir())
    if not subdirs:
        # 直接放在 mock_root 下也支持
        json_files = sorted(mock_root.glob("*.json"))
        for jf in json_files:
            recs = _load_json_array(jf)
            for r in recs:
                if section_filter and r.get("section") != section_filter:
                    continue
                all_records.append(r)
            source_log.append(f"{jf.name}: {len(recs)} 条")
        return all_records, source_log

    for sub in subdirs:
        json_files = sorted(sub.glob("*.json"))
        if not json_files:
            source_log.append(f"{sub.name}/: (空)")
            continue
        sub_count = 0
        for jf in json_files:
            try:
                recs = _load_json_array(jf)
            except Exception as e:
                source_log.append(f"{sub.name}/{jf.name}: ERROR {e}")
                continue
            for r in recs:
                if section_filter and r.get("section") != section_filter:
                    continue
                all_records.append(r)
            sub_count += len(recs)
        source_log.append(f"{sub.name}/: {sub_count} 条 ({len(json_files)} 文件)")
    return all_records, source_log


def _build_cmd(args: argparse.Namespace, input_path: Path, pred_path: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "solution.main",
        "--input", str(input_path),
        "--output", str(pred_path),
        "--topk", str(args.topk),
        "--llm", args.llm,
        "--workers", str(args.workers),
    ]
    if args.mode == "baseline":
        cmd += ["--no-fewshot", "--vqa-samples", "1"]
    else:
        vqa_n = args.vqa_samples if args.vqa_samples > 0 else 3
        cmd += ["--vqa-samples", str(vqa_n)]
    return cmd


def _resolve_python() -> str:
    """与 run_test.py 一致：优先使用本地 .venv-student_package_solution。"""
    venv_py = PKG_DIR / ".venv-student_package_solution" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> int:
    args = parse_args()
    mock_root = Path(args.mock_root)

    # ---- 1) 读取并筛选 ----
    records, source_log = discover_mock_records(mock_root, args.section)
    print(f"[mock] 扫描根目录: {mock_root}")
    for line in source_log:
        print(f"[mock]   {line}")
    if args.section:
        print(f"[mock] 过滤 section = {args.section}")
    print(f"[mock] 合并后总数: {len(records)}")

    if not records:
        print("[mock] FATAL: 0 条样本，请检查 --mock-root / --section", file=sys.stderr)
        return 2

    # ---- 2) 抽样 / 截断 ----
    if args.sample and args.limit:
        print("[mock] FATAL: --sample 与 --limit 互斥", file=sys.stderr)
        return 2
    if args.sample:
        if args.sample >= len(records):
            print(f"[mock] --sample {args.sample} >= 总数 {len(records)}，跑全部")
        else:
            rng = random.Random(args.seed)
            records = rng.sample(records, args.sample)
            print(f"[mock] 随机抽样 {args.sample} 条 (seed={args.seed})")
    elif args.limit:
        records = records[: args.limit]
        print(f"[mock] 顺序取前 {args.limit} 条")

    # 按 section 统计输入分布
    sec_count: dict[str, int] = {}
    for r in records:
        sec = str(r.get("section") or "?")
        sec_count[sec] = sec_count.get(sec, 0) + 1
    print(f"[mock] 输入分布: {sec_count}")

    # ---- 3) 写临时合并 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = OUTPUT_ROOT / f"mock_{ts}_{args.mode}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_input = out_dir / "mock_input.json"
    merged_input.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[mock] 合并输入 -> {merged_input}")

    pred_path = out_dir / "predictions.jsonl"
    time_path = out_dir / "time_cost.jsonl"
    log_path = out_dir / "run.log"
    meta_path = out_dir / "meta.json"
    metrics_path = out_dir / "metrics.json"

    # 快照 solution/
    snap = out_dir / "solution_snapshot"
    snap.mkdir(exist_ok=True)
    for f in (PKG_DIR / "solution").glob("*.py"):
        shutil.copy2(f, snap / f.name)
    req = PKG_DIR / "solution" / "requirements.txt"
    if req.exists():
        shutil.copy2(req, snap / "requirements.txt")

    # ---- 4) 环境变量（与 run_test.py 一致） ----
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY")
    env.setdefault("QWEN_MODEL_NAME", "qwen2.5-vl-7b-instruct")
    # 本地 Qwen2.5-VL chat 模型（--llm local 必需）
    env.setdefault("QWEN_MODEL_PATH", "/root/autodl-fs/model/Qwen2.5-VL-7B-Instruct")
    env.setdefault(
        "DASHVECTOR_ENDPOINT",
        "vrs-cn-71j49hani00061.dashvector.cn-hangzhou.aliyuncs.com",
    )
    env.setdefault(
        "DASHVECTOR_API_KEY",
        "sk-REPLACE-WITH-YOUR-OWN-VECTOR-KEY",
    )
    env.setdefault("DASHVECTOR_COLLECTION", "ccv_cb_student_20260508")
    env.setdefault("DASHVECTOR_VECTOR_FIELD", "text")
    env.setdefault("GME_MODEL_PATH", "/autodl-fs/data/gme-Qwen2-VL-7B-Instruct")
    env.setdefault("DASHSCOPE_API_KEY", env["QWEN_API_KEY"])
    env.setdefault("DASHSCOPE_EMBED_MODEL", "multimodal-embedding-v1")

    # ---- 5) 跑 solution/main.py ----
    py = _resolve_python()
    cmd = _build_cmd(args, merged_input, pred_path)
    cmd[0] = py  # 替换为 .venv-student_package_solution python
    started = datetime.now()
    print(f"[mock] 输出目录: {out_dir}")
    print(f"[mock] 命令: {' '.join(cmd)}")
    print(f"[mock] cwd: {PKG_DIR}")
    print(f"[mock] 开始: {started.isoformat(timespec='seconds')}")
    print("-" * 60, flush=True)

    exit_code = 0
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# cmd: {' '.join(cmd)}\n")
            logf.write(f"# cwd: {PKG_DIR}\n")
            logf.write(f"# started_at: {started.isoformat(timespec='seconds')}\n")
            logf.write(f"# sample={args.sample} seed={args.seed} limit={args.limit} section={args.section!r}\n")
            logf.write("-" * 60 + "\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(PKG_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            exit_code = proc.wait()
    except KeyboardInterrupt:
        print("\n[mock] 中断", file=sys.stderr)
        exit_code = 130

    finished = datetime.now()
    elapsed = (finished - started).total_seconds()
    print("-" * 60)
    print(f"[mock] 推理结束: {finished.isoformat(timespec='seconds')}  耗时 {elapsed:.1f}s  exit={exit_code}")

    # ---- 6) 校验输出格式 ----
    print("[mock] 校验输出格式 ...")
    check_ret = subprocess.run(
        [
            py, str(PKG_DIR / "scripts" / "check_output.py"),
            "--input", str(merged_input),
            "--prediction", str(pred_path),
        ],
        cwd=str(PKG_DIR),
        env=env,
    )

    # ---- 7) 评分 ----
    print("[mock] 评分 ...")
    eval_ret = subprocess.run(
        [
            py, str(PKG_DIR / "scripts" / "evaluate.py"),
            "--gold", str(merged_input),
            "--prediction", str(pred_path),
            "--output", str(metrics_path),
        ],
        cwd=str(PKG_DIR),
        env={**env, "PYTHONPATH": str(PKG_DIR)},
    )

    # ---- 8) 错误分析 ----
    print("[mock] 错误样例分析 ...")
    sys.path.insert(0, str(HERE))
    from error_analysis import analyze  # noqa: E402

    summary = analyze(
        gold_path=merged_input,
        prediction_path=pred_path,
        time_cost_path=time_path,
        out_dir=out_dir,
        strategy=f"mock-{args.mode}",
    )

    meta = {
        "timestamp": ts,
        "mode": args.mode,
        "tag": args.tag,
        "mock_root": str(mock_root),
        "section_filter": args.section,
        "sample": args.sample,
        "seed": args.seed if args.sample else None,
        "limit": args.limit,
        "topk": args.topk,
        "input_distribution": sec_count,
        "cmd": cmd,
        "cwd": str(PKG_DIR),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": exit_code,
        "check_output_exit": check_ret.returncode,
        "evaluate_exit": eval_ret.returncode,
        "python": py,
        "platform": platform.platform(),
        "host": platform.node(),
        "summary": summary,
        "official_baseline_acc": OFFICIAL_BASELINE_ACC,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 9) 终端汇总 ----
    print()
    print("=" * 60)
    print(f"[mock] 总体: {summary['correct']}/{summary['total']} = {summary['accuracy']*100:.1f}%")
    by_sec = summary.get("by_section", {})
    if by_sec:
        print(f"[mock] 按 section:")
        for sec in sorted(by_sec):
            v = by_sec[sec]
            acc = v["correct"] / v["total"] * 100 if v["total"] else 0.0
            print(f"         {sec:10s}  {v['correct']:4d}/{v['total']:<4d}  {acc:5.1f}%")
    print()
    gap = summary["accuracy"] - OFFICIAL_BASELINE_ACC
    sign = "+" if gap >= 0 else ""
    print(f"[mock] 泛化 gap vs 官方 50 题 ({OFFICIAL_BASELINE_ACC*100:.0f}%): {sign}{gap*100:.1f} pp")
    print(f"[mock] 输出      = {out_dir}")
    print(f"[mock] metrics  = {metrics_path}")
    print(f"[mock] analysis = {out_dir/'error_analysis.md'}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
