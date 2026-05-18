"""
compare_strategies.py
=====================
一次跑两种策略 (baseline + rag), 落到同一个时间戳目录, 产出对比报告 compare.md。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
OUTPUT_ROOT = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对比 baseline 与 rag 两种策略")
    p.add_argument("--input", default=str(PKG_DIR / "data" / "public_examples.full.json"))
    p.add_argument("--gold", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--topk", type=int, default=4)
    return p.parse_args()


def _run_one(strategy: str, args: argparse.Namespace, out_dir: Path) -> dict:
    pred_path = out_dir / "predictions.jsonl"
    time_path = out_dir / "time_cost.jsonl"
    log_path = out_dir / "run.log"
    metrics_path = out_dir / "metrics.json"

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY")
    env.setdefault("QWEN_MODEL_NAME", "qwen2.5-vl-7b-instruct")

    cmd = [
        sys.executable, "-u", "-m", "solution.main",
        "--input", args.input,
        "--output", str(pred_path),
        "--strategy", strategy,
        "--topk", str(args.topk),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    started = datetime.now()
    print(f"\n[compare] ===== running {strategy} =====")
    print(f"[compare] 输出目录: {out_dir}")
    print(f"[compare] 命令: {' '.join(cmd)}")

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.write(f"# started_at: {started.isoformat(timespec='seconds')}\n")
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
        rc = proc.wait()

    # 评分
    subprocess.run(
        [
            sys.executable, str(PKG_DIR / "scripts" / "evaluate.py"),
            "--gold", args.gold,
            "--prediction", str(pred_path),
            "--output", str(metrics_path),
        ],
        cwd=str(PKG_DIR),
        env={**env, "PYTHONPATH": str(PKG_DIR)},
    )

    sys.path.insert(0, str(HERE))
    from error_analysis import analyze  # noqa: E402

    summary = analyze(
        gold_path=Path(args.gold),
        prediction_path=pred_path,
        time_cost_path=time_path,
        out_dir=out_dir,
        strategy=strategy,
    )
    summary["exit_code"] = rc
    summary["elapsed_seconds"] = round((datetime.now() - started).total_seconds(), 2)
    return summary


def _render_compare_md(b: dict, r: dict) -> str:
    md: list[str] = []
    md.append(f"# 策略对比 — {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append("## 总览")
    md.append("")
    md.append("| 策略 | 总数 | 正确 | 准确率 | 错误 | 平均检索 | 平均推理 | 耗时 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for s in (b, r):
        md.append(
            "| {strategy} | {total} | {correct} | {acc:.1f}% | {err} | "
            "{ret:.3f}s | {inf:.3f}s | {el:.1f}s |".format(
                strategy=s["strategy"],
                total=s["total"],
                correct=s["correct"],
                acc=s["accuracy"] * 100,
                err=s["error_count"],
                ret=s.get("avg_retrieval_cost", 0.0),
                inf=s.get("avg_infer_cost", 0.0),
                el=s.get("elapsed_seconds", 0.0),
            )
        )
    delta = (r["accuracy"] - b["accuracy"]) * 100
    md.append("")
    md.append(f"**RAG vs baseline 准确率差: {delta:+.1f} 个百分点**")
    md.append("")
    md.append("## 按题型对比")
    md.append("")
    sections = sorted(set(b["by_section"]) | set(r["by_section"]))
    md.append("| section | baseline acc | rag acc | Δ |")
    md.append("|---|---|---|---|")
    for sec in sections:
        ba = b["by_section"].get(sec, {}).get("accuracy", 0.0) * 100
        ra = r["by_section"].get(sec, {}).get("accuracy", 0.0) * 100
        md.append(f"| {sec} | {ba:.1f}% | {ra:.1f}% | {ra-ba:+.1f} |")
    md.append("")
    md.append("## 错因分布对比")
    md.append("")
    md.append("| 错因 | baseline | rag |")
    md.append("|---|---|---|")
    keys = sorted(set(b.get("error_types", {})) | set(r.get("error_types", {})))
    for k in keys:
        md.append(f"| {k} | {b.get('error_types', {}).get(k, 0)} | {r.get('error_types', {}).get(k, 0)} |")
    md.append("")
    md.append("## 详细错误清单")
    md.append("")
    md.append("- baseline: `baseline/error_analysis.md`, `baseline/errors.jsonl`")
    md.append("- rag:      `rag/error_analysis.md`, `rag/errors.jsonl`")
    return "\n".join(md)


def main() -> int:
    args = parse_args()
    if not args.gold:
        args.gold = args.input
    if not Path(args.input).exists():
        print(f"FATAL: input not found: {args.input}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = OUTPUT_ROOT / f"{ts}_compare"
    root.mkdir(parents=True, exist_ok=True)

    snap = root / "solution_snapshot"
    snap.mkdir(exist_ok=True)
    for f in (PKG_DIR / "solution").glob("*.py"):
        shutil.copy2(f, snap / f.name)
    req = PKG_DIR / "solution" / "requirements.txt"
    if req.exists():
        shutil.copy2(req, snap / "requirements.txt")

    b_dir = root / "baseline"
    r_dir = root / "rag"
    b_dir.mkdir(exist_ok=True)
    r_dir.mkdir(exist_ok=True)

    b_sum = _run_one("baseline", args, b_dir)
    r_sum = _run_one("rag", args, r_dir)

    md = _render_compare_md(b_sum, r_sum)
    (root / "compare.md").write_text(md, encoding="utf-8")
    (root / "compare.json").write_text(
        json.dumps({"baseline": b_sum, "rag": r_sum}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"[compare] baseline acc = {b_sum['accuracy']*100:.1f}%  rag acc = {r_sum['accuracy']*100:.1f}%")
    print(f"[compare] Δ = {(r_sum['accuracy']-b_sum['accuracy'])*100:+.1f} 个百分点")
    print(f"[compare] 报告: {root/'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
