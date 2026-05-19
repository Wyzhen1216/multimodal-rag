"""
solution/startup_check.py
==========================
启动自检模块 —— 在主循环开始前验证三项：

  1. 归一化正确性（静态，无需 LLM）
  2. LLM smoke test（各题型各发一条合成题）
  3. Retriever 连通性（失败仅警告，不退出）

调用方式：
    from .startup_check import run_startup_checks
    run_startup_checks(client, retriever)   # 失败时 sys.exit(1)
"""
from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # 仅用于类型提示，避免循环导入


# ============================================================================
# 1. 归一化静态检查
# ============================================================================

def _check_normalization() -> None:
    from .normalization import normalize_answer

    cases = [
        ("选择QA",  "答案: B",  "B"),
        ("选择QA",  "AB",       "AB"),
        ("判断QA",  "正确",     "正确"),
        ("判断QA",  "错误",     "错误"),
        ("简答QA",  "12",       "12"),
        ("简答VQA", "月球",     "月球"),
    ]
    for section, raw, expected in cases:
        result = normalize_answer(section, raw)
        if result != expected:
            raise RuntimeError(
                f"[startup] normalize FAIL: section={section!r} "
                f"input={raw!r} expected={expected!r} got={result!r}"
            )
    print("[startup] normalize check OK", flush=True)


# ============================================================================
# 2. LLM Smoke Test
# ============================================================================

_SMOKE_CASES = [
    ("选择QA",  "以下哪个是水果？A.苹果 B.铁 C.石头 D.木头"),
    ("选择VQA", "以下哪个颜色描述是红色？A.红色 B.蓝色 C.绿色 D.黄色"),
    ("判断QA",  "水在标准大气压下沸点为100摄氏度。"),
    ("判断VQA", "太阳从东边升起。"),
    ("简答QA",  "一年有多少个月？"),
    ("简答VQA", "天空通常是什么颜色？"),
]

_SMOKE_TIMEOUT = 120  # 单题最长等待秒数（本地模型首次推理可能较慢）


def _check_llm(client) -> None:
    from .prompts import system_prompt_for

    any_fail = False
    for section, question in _SMOKE_CASES:
        system = system_prompt_for(section)
        t0 = time.perf_counter()
        try:
            out = client.generate(system, question, image_path=None,
                                  temperature=0.0, max_tokens=16)
            elapsed = time.perf_counter() - t0
            if not out or not out.strip():
                print(
                    f"[startup] LLM smoke FAIL: {section} — 返回空字符串 ({elapsed:.1f}s)",
                    flush=True,
                )
                any_fail = True
            elif elapsed > _SMOKE_TIMEOUT:
                print(
                    f"[startup] LLM smoke WARN: {section} — 耗时过长 {elapsed:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"[startup] LLM smoke OK: {section} ({elapsed:.1f}s) → {out.strip()[:20]!r}",
                    flush=True,
                )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(
                f"[startup] LLM smoke FAIL: {section} — {e} ({elapsed:.1f}s)",
                flush=True,
            )
            any_fail = True

    if any_fail:
        print("[startup] LLM smoke test FAILED — 请检查 QWEN_MODEL_PATH 是否正确", flush=True)
        sys.exit(1)
    print("[startup] LLM smoke test ALL PASSED", flush=True)


# ============================================================================
# 3. Retriever 连通性检查
# ============================================================================

def _check_retriever(retriever) -> None:
    # NullRetriever 直接跳过
    if type(retriever).__name__ == "NullRetriever":
        print("[startup] retriever check: NullRetriever — 闭卷推理模式", flush=True)
        return

    # DashVectorRetriever 但已被禁用
    disabled = getattr(retriever, "_disabled", False)
    if disabled:
        print("[startup] retriever check: DashVectorRetriever disabled — 闭卷推理模式", flush=True)
        return

    try:
        t0 = time.perf_counter()
        results = retriever.search(text="考古遗址出土文物", topk=1)
        elapsed = time.perf_counter() - t0
        if isinstance(results, list):
            print(
                f"[startup] retriever check OK ({elapsed:.2f}s, returned {len(results)} doc(s))",
                flush=True,
            )
        else:
            print(
                f"[startup] retriever check WARN: 返回类型异常 {type(results)} — 继续运行",
                flush=True,
            )
    except Exception as e:
        print(f"[startup] retriever check WARN: {e} — 继续运行（闭卷模式）", flush=True)


# ============================================================================
# 入口
# ============================================================================

def run_startup_checks(client, retriever) -> None:
    """执行全部启动检查。LLM/归一化失败时 sys.exit(1)，检索失败仅警告。"""
    print("[startup] ====== STARTUP SELF-TEST BEGIN ======", flush=True)
    _check_normalization()
    _check_llm(client)
    _check_retriever(retriever)
    print("[startup] ====== ALL CHECKS PASSED ======\n", flush=True)
