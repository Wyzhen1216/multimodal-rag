"""
run_test.py
============
跑一次测试: 调 solution/main.py, 创建以时间戳命名的输出目录, 落预测、日志、metrics、错误分析。

用法:
    # 基线 (无检索)
    python test/run_test.py --strategy baseline

    # 检索策略 (借鉴 chuangzhi-summer-camp 的 BM25+TF-IDF RRF)
    python test/run_test.py --strategy rag

    # 仅跑前 5 条 (调试)
    python test/run_test.py --strategy baseline --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent                # student_package/test
PKG_DIR = HERE.parent                                  # student_package
OUTPUT_ROOT = HERE / "outputs"                         # student_package/test/outputs/<ts>


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单策略测试执行器")
    p.add_argument(
        "--strategy",
        choices=("baseline", "rag"),
        default="baseline",
        help="baseline = 无检索; rag = TF-IDF + BM25 RRF 检索",
    )
    p.add_argument("--input", default=str(PKG_DIR / "data" / "public_examples.full.json"))
    p.add_argument("--gold", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--tag", default="", help="附加在输出目录名后, 便于区分")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.gold:
        args.gold = args.input

    if not Path(args.input).exists():
        print(f"FATAL: input not found: {args.input}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = OUTPUT_ROOT / f"{ts}_{args.strategy}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    time_path = out_dir / "time_cost.jsonl"
    log_path = out_dir / "run.log"
    meta_path = out_dir / "meta.json"
    metrics_path = out_dir / "metrics.json"

    # ---- 快照 solution/ (便于回看本次测试代码) ----
    snap = out_dir / "solution_snapshot"
    snap.mkdir(exist_ok=True)
    for f in (PKG_DIR / "solution").glob("*.py"):
        shutil.copy2(f, snap / f.name)
    req = PKG_DIR / "solution" / "requirements.txt"
    if req.exists():
        shutil.copy2(req, snap / "requirements.txt")

    # ---- 调 solution.main ----
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY")
    env.setdefault("QWEN_MODEL_NAME", "qwen2.5-vl-7b-instruct")

    cmd = [
        sys.executable, "-u", "-m", "solution.main",
        "--input", args.input,
        "--output", str(pred_path),
        "--strategy", args.strategy,
        "--topk", str(args.topk),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    started = datetime.now()
    print(f"[run_test] 输出目录: {out_dir}")
    print(f"[run_test] 命令: {' '.join(cmd)}")
    print(f"[run_test] cwd: {PKG_DIR}")
    print(f"[run_test] 开始: {started.isoformat(timespec='seconds')}")
    print("-" * 60, flush=True)

    exit_code = 0
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"# cmd: {' '.join(cmd)}\n")
            logf.write(f"# cwd: {PKG_DIR}\n")
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
            exit_code = proc.wait()
    except KeyboardInterrupt:
        print("\n[run_test] 中断", file=sys.stderr)
        exit_code = 130

    finished = datetime.now()
    elapsed = (finished - started).total_seconds()

    print("-" * 60)
    print(f"[run_test] 结束: {finished.isoformat(timespec='seconds')}  耗时 {elapsed:.1f}s  exit={exit_code}")

    # ---- check_output.py 格式校验 ----
    print("[run_test] 校验输出格式 ...")
    check_cmd = [
        sys.executable, str(PKG_DIR / "scripts" / "check_output.py"),
        "--input", args.input,
        "--prediction", str(pred_path),
    ]
    check_ret = subprocess.run(check_cmd, cwd=str(PKG_DIR), env=env)

    # ---- evaluate.py 评分 ----
    print("[run_test] 评分 ...")
    eval_cmd = [
        sys.executable, str(PKG_DIR / "scripts" / "evaluate.py"),
        "--gold", args.gold,
        "--prediction", str(pred_path),
        "--output", str(metrics_path),
    ]
    eval_ret = subprocess.run(eval_cmd, cwd=str(PKG_DIR), env={**env, "PYTHONPATH": str(PKG_DIR)})

    # ---- 错误样例分析 ----
    print("[run_test] 错误样例分析 ...")
    sys.path.insert(0, str(HERE))
    from error_analysis import analyze  # noqa: E402

    summary = analyze(
        gold_path=Path(args.gold),
        prediction_path=pred_path,
        time_cost_path=time_path,
        out_dir=out_dir,
        strategy=args.strategy,
    )

    # ---- meta.json ----
    meta = {
        "timestamp": ts,
        "strategy": args.strategy,
        "tag": args.tag,
        "limit": args.limit,
        "topk": args.topk,
        "cmd": cmd,
        "cwd": str(PKG_DIR),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": exit_code,
        "check_output_exit": check_ret.returncode,
        "evaluate_exit": eval_ret.returncode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "summary": summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"[run_test] accuracy = {summary['accuracy']*100:.1f}%  ({summary['correct']}/{summary['total']})")
    print(f"[run_test] errors   = {summary['error_count']}")
    print(f"[run_test] 输出      = {out_dir}")
    print(f"[run_test] metrics  = {metrics_path}")
    print(f"[run_test] analysis = {out_dir/'error_analysis.md'}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
