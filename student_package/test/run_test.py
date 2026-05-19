"""
run_test.py
============
跑一次测试：调 solution/main.py，落预测、日志、metrics、错误分析。

模式（与新版 solution/main.py 对齐）：
  - baseline : 仅"题型分支 prompt"（关闭 few-shot 和 VQA 多采样投票）
  - enhanced : 默认全开（few-shot + VQA 多采样投票）

本地测试（无 DashVector / GME）：
  retriever 会自动降级为 NullRetriever，效果上 = 闭卷推理。
  正式评测环境需 export DASHVECTOR_* 和 GME_MODEL_PATH 才会启用检索。

用法:
    python test/run_test.py --mode baseline
    python test/run_test.py --mode enhanced
    python test/run_test.py --mode enhanced --limit 5    # 调试用
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


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
OUTPUT_ROOT = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单模式测试执行器")
    p.add_argument(
        "--mode",
        choices=("baseline", "enhanced"),
        default="enhanced",
        help="baseline = 仅题型分支 prompt; enhanced = + few-shot + VQA 投票",
    )
    p.add_argument("--input", default=str(PKG_DIR / "data" / "public_examples.full.json"))
    p.add_argument("--gold", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--vqa-samples", type=int, default=0, help="覆盖默认: 0 表示按 mode 决定")
    p.add_argument("--tag", default="")
    return p.parse_args()


def _resolve_python() -> str:
    """优先用 .venv-student_package_solution 的 python（与官测环境同栈：
    transformers>=4.49 + torch>=2.4），避免 base python 的 torch 2.1 触发
    transformers 关闭 PyTorch 后端的问题。"""
    venv_py = PKG_DIR / ".venv-student_package_solution" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _build_cmd(args: argparse.Namespace, pred_path: Path) -> list[str]:
    cmd = [
        _resolve_python(), "-u", "-m", "solution.main",
        "--input", args.input,
        "--output", str(pred_path),
        "--topk", str(args.topk),
        "--llm", "local",  # 官测环境强制本地模型
    ]
    if args.mode == "baseline":
        cmd += ["--no-fewshot", "--vqa-samples", "1"]
    else:
        vqa_n = args.vqa_samples if args.vqa_samples > 0 else 3
        cmd += ["--vqa-samples", str(vqa_n)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main() -> int:
    args = parse_args()
    if not args.gold:
        args.gold = args.input

    if not Path(args.input).exists():
        print(f"FATAL: input not found: {args.input}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = OUTPUT_ROOT / f"{ts}_{args.mode}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY")
    env.setdefault("QWEN_MODEL_NAME", "qwen2.5-vl-7b-instruct")
    # 本地 Qwen2.5-VL chat 模型（与官测环境同款；--llm local 必需）
    env.setdefault("QWEN_MODEL_PATH", "/root/autodl-fs/model/Qwen2.5-VL-7B-Instruct")
    # DashVector 凭证（评测环境同款），本地连得上但因没有本地 GME 会自动降级
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
    # 本地 GME 模型路径（评测环境同款；3584 维与 collection 匹配）
    env.setdefault("GME_MODEL_PATH", "/autodl-fs/data/gme-Qwen2-VL-7B-Instruct")
    # DashScope embedding 兜底（GME 加载失败时才用）
    env.setdefault("DASHSCOPE_API_KEY", env["QWEN_API_KEY"])
    env.setdefault("DASHSCOPE_EMBED_MODEL", "multimodal-embedding-v1")

    cmd = _build_cmd(args, pred_path)
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

    # check_output.py
    print("[run_test] 校验输出格式 ...")
    check_ret = subprocess.run(
        [
            sys.executable, str(PKG_DIR / "scripts" / "check_output.py"),
            "--input", args.input,
            "--prediction", str(pred_path),
        ],
        cwd=str(PKG_DIR),
        env=env,
    )

    # evaluate.py
    print("[run_test] 评分 ...")
    eval_ret = subprocess.run(
        [
            sys.executable, str(PKG_DIR / "scripts" / "evaluate.py"),
            "--gold", args.gold,
            "--prediction", str(pred_path),
            "--output", str(metrics_path),
        ],
        cwd=str(PKG_DIR),
        env={**env, "PYTHONPATH": str(PKG_DIR)},
    )

    # 错误分析
    print("[run_test] 错误样例分析 ...")
    sys.path.insert(0, str(HERE))
    from error_analysis import analyze  # noqa: E402

    summary = analyze(
        gold_path=Path(args.gold),
        prediction_path=pred_path,
        time_cost_path=time_path,
        out_dir=out_dir,
        strategy=args.mode,
    )

    meta = {
        "timestamp": ts,
        "mode": args.mode,
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
