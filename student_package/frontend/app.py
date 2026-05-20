"""Flask 后端 — 多模态历史 RAG Agent 演示界面。

启动:
    cd /root/autodl-fs/multimodal-rang/student_package
    .venv-student_package_solution/bin/python -m frontend.app
然后浏览器访问 http://localhost:5000

设计:
  - 单页 + 顶部 tab 切换 (首页/方法论/交互演示)
  - 数据来自最新 v2 预测结果（已落地，无需 GPU）
  - "自由提问"模式调用 DashScope qwen2.5-vl API（仅闭卷 LLM，不本地加载模型）
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

# ---- 数据加载 -------------------------------------------------------------

DATA = PKG / "data"
OUTPUTS = PKG / "test" / "outputs" / "20260520_081855_enhanced_fusion_AB_v2"
IMAGES_DIR = DATA / "images"

QUESTIONS_PATH = DATA / "public_examples.full.json"
PREDICTIONS_PATH = OUTPUTS / "predictions.jsonl"
METRICS_PATH = OUTPUTS / "metrics.json"
TIME_COST_PATH = OUTPUTS / "time_cost.jsonl"
RUN_LOG_PATH = OUTPUTS / "run.log"
ERROR_PATH = OUTPUTS / "errors.jsonl"


def _load_jsonl_by_idx(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    with p.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec.get("index")] = rec
            except Exception:
                pass
    return out


QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if QUESTIONS_PATH.exists() else []
QUESTIONS_BY_IDX = {q.get("index"): q for q in QUESTIONS}
PREDICTIONS = _load_jsonl_by_idx(PREDICTIONS_PATH)
TIME_COSTS = _load_jsonl_by_idx(TIME_COST_PATH)
METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
ERRORS = _load_jsonl_by_idx(ERROR_PATH)

# ---- 从 run.log 解析每题路由 / 检索路径 -----------------------------------

def _parse_routing_log() -> dict:
    """提取 run.log 里每题的关键日志行，给前端展示检索过程。

    log 格式：
      [idx=358] xxx                    (单题分支日志，自含 idx)
      [5/50] idx=358 sec=...           (汇总主行，4 选项并行后输出)
      [option A] xxx                   (选项子行，先于主行出现)
      [reflect/recheck/count_predict]  (后处理子行)

    解析策略：[option/reflect/recheck/count_predict] 这类**没有 idx 的子行**
    缓存到 pending_substr，遇到下一个含 idx 的主行时归到该 idx。
    [idx=...] 自含 idx 的行直接归类。
    """
    import re
    if not RUN_LOG_PATH.exists():
        return {}
    routing = {}
    pending_substrs = []  # 累积的子行，等下一个主行分配
    pat_idx = re.compile(r"idx=(\d+)")
    sub_prefixes = ("[option", "[reflect", "[recheck", "[count_predict")
    with RUN_LOG_PATH.open() as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.lstrip()
            if any(stripped.startswith(p) for p in sub_prefixes):
                pending_substrs.append(line)
                continue
            m = pat_idx.search(line)
            if m:
                idx = int(m.group(1))
                bucket = routing.setdefault(idx, [])
                # 先把之前累积的子行归到这个主行
                bucket.extend(pending_substrs)
                pending_substrs = []
                bucket.append(line)
    return routing


ROUTING_LOG = _parse_routing_log()

# ---- 题型 → 路由方法（与 main.py / router_retriever.py 一致）---------------

ROUTE_OF_SECTION = {
    "判断QA": "BM25 hybrid (DashVector + BM25 RRF)",
    "判断VQA": "BM25 hybrid (DashVector + BM25 RRF)",
    "简答QA": "HyDE + IB rerank → DashVector",
    "简答VQA": "vector-only (DashVector，识字直读不污染)",
    "选择QA": "vector-only + choice-as-judge",
    "选择VQA": "vector-only + choice-as-judge",
}

# ---- 优化阶段时间线（与 findings/技术报告.md 一致）-------------------------

OPTIMIZATION_TIMELINE = [
    {
        "name": "baseline",
        "title": "Baseline (vector-only)",
        "score_50": 76,
        "score_120": 48.3,
        "delta": "起点",
        "highlights": ["DashVector 单路检索", "题型分支 prompt", "VQA 多采样投票"],
    },
    {
        "name": "bm25",
        "title": "+ BM25 hybrid",
        "score_50": 74,
        "score_120": 50.8,
        "delta": "-1 题（噪声内）",
        "highlights": ["jieba 分词 + BM25Okapi", "5472 段干净自举 corpus", "RRF k=60 融合"],
    },
    {
        "name": "hyde",
        "title": "+ HyDE only",
        "score_50": 72,
        "score_120": None,
        "delta": "-2 题（判断QA -3）",
        "highlights": ["LLM 生成假设答案 anchor", "对判断题反而干扰", "需按题型路由才有效"],
    },
    {
        "name": "rerank",
        "title": "+ Reranker only",
        "score_50": 74,
        "score_120": None,
        "delta": "-1 题（选择QA -1）",
        "highlights": ["bge-reranker-v2-m3 cross-encoder", "对 choice-as-judge 反向证据干扰"],
    },
    {
        "name": "fusion",
        "title": "Fusion Router",
        "score_50": 74,
        "score_120": 51.7,
        "delta": "mock +3.4pp，官评待优化",
        "highlights": ["按题型分发", "判断→BM25 / 简答QA→HyDE+IB / 其他→vector"],
    },
    {
        "name": "v2",
        "title": "★ Fusion + Prompt v2",
        "score_50": 80,
        "score_120": 51.7,
        "delta": "+2 题，达成目标",
        "highlights": [
            "SYS_SHORT 重写：题问类型强制格式（省/帝/顿号/句号）",
            "SYS_SHORT_VQA 独立保留视觉直读",
            "SYS_JUDGE_SOFT 严格 TF 验证（精确含'该问题的答案是'才触发）",
        ],
    },
]

# ---- 自由提问模式（DashScope qwen2.5-vl-7b-instruct）---------------------

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
    "QWEN_API_KEY", "sk-REPLACE-WITH-YOUR-OWN-QWEN-KEY"
)


def _build_system_prompt(section: str) -> str:
    """与 solution/prompts.py 关键 prompt 对齐（仅闭卷推理，不带 RAG context）。"""
    if "判断" in section:
        return (
            "你是一名考古学/历史学问答助手。请判断给定陈述是否符合史实。"
            "仅输出'正确'或'错误'两字之一，不要任何前缀、解释或标点。"
            "不确定时，倾向于判断为'正确'。"
        )
    if "选择" in section:
        return (
            "你是一名考古学/历史学多模态问答助手。"
            "请仔细阅读题目和选项，输出正确的字母序列（如 'A' 或 'AB'）。"
            "对'下列哪些/包括'的多选题输出多个字母连写。"
        )
    # 简答 QA / VQA
    return (
        "你是一名多模态问答助手。请按题问的格式给出最简短答案。\n"
        "- '哪个/什么+省份' → 必须以'省'结尾\n"
        "- '哪位+皇帝' → 必须以'帝/皇帝'结尾\n"
        "- '哪几个/分别是' → 顿号'、'分隔 + 句号\n"
        "- '是什么刑罚' → 必须以'刑'结尾\n"
        "不要前缀、不要解释，第一行就是最终答案。"
    )


def _call_qwen_vl_api(system: str, user_text: str, image_url: Optional[str] = None) -> dict:
    """调用 DashScope OpenAI 兼容接口（不本地加载模型，演示模式专用）。"""
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai SDK 未安装，自由提问需要 'pip install openai'"}
    if not DASHSCOPE_API_KEY:
        return {"error": "DASHSCOPE_API_KEY 未设置，无法调用 LLM"}

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=DASHSCOPE_API_KEY,
    )

    user_content = [{"type": "text", "text": user_text}]
    if image_url:
        # 支持 URL 或本地文件相对路径
        if not image_url.startswith("http"):
            local = (DATA / image_url) if not Path(image_url).is_absolute() else Path(image_url)
            if local.exists():
                b64 = base64.b64encode(local.read_bytes()).decode("ascii")
                image_url = f"data:image/jpeg;base64,{b64}"
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    try:
        r = client.chat.completions.create(
            model="qwen2.5-vl-7b-instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return {"answer": r.choices[0].message.content or "", "model": "qwen2.5-vl-7b-instruct"}
    except Exception as e:
        return {"error": f"LLM 调用失败: {e}"}


# ---- Flask app -----------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/timeline")
def api_timeline():
    return jsonify(OPTIMIZATION_TIMELINE)


@app.route("/api/sections")
def api_sections():
    """各题型题数 + 路由方法。"""
    sections = {}
    for q in QUESTIONS:
        s = q.get("section", "未知")
        sections.setdefault(s, {"section": s, "count": 0, "indexes": [], "route": ROUTE_OF_SECTION.get(s, "vector-only")})
        sections[s]["count"] += 1
        sections[s]["indexes"].append(q["index"])
    return jsonify(list(sections.values()))


@app.route("/api/questions")
def api_questions():
    """题列表，可按 section 过滤。"""
    section = request.args.get("section", "")
    out = []
    for q in QUESTIONS:
        if section and q.get("section") != section:
            continue
        idx = q["index"]
        pred = PREDICTIONS.get(idx, {})
        is_correct = idx not in ERRORS  # errors.jsonl 含全部错题
        out.append({
            "index": idx,
            "section": q.get("section"),
            "question": q.get("question"),
            "image_url": q.get("image_url"),
            "expected_answer": q.get("expected_answer"),
            "predicted_answer": pred.get("answer"),
            "correct": is_correct,
        })
    return jsonify(out)


@app.route("/api/question/<int:idx>")
def api_question(idx: int):
    """单题详情：题面、图片、路由、检索过程、预测、对比。"""
    q = QUESTIONS_BY_IDX.get(idx)
    if not q:
        return jsonify({"error": "not found"}), 404
    section = q.get("section", "")
    pred = PREDICTIONS.get(idx, {})
    tc = TIME_COSTS.get(idx, {})
    is_correct = idx not in ERRORS
    log_lines = ROUTING_LOG.get(idx, [])
    return jsonify({
        "question": q,
        "section": section,
        "route": ROUTE_OF_SECTION.get(section, "vector-only"),
        "prediction": pred,
        "time_cost": tc,
        "correct": is_correct,
        "log_lines": log_lines,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """自由提问：调 DashScope qwen2.5-vl-7b-instruct（仅 LLM 闭卷，不带 RAG）。"""
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    section = (data.get("section") or "简答QA").strip()
    image_url = (data.get("image_url") or "").strip() or None
    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    system = _build_system_prompt(section)
    result = _call_qwen_vl_api(system, question, image_url)
    return jsonify({
        "section": section,
        "route": ROUTE_OF_SECTION.get(section, "vector-only"),
        "system_prompt_preview": system[:200],
        **result,
        "note": "演示模式：仅 LLM 闭卷推理（不调用 BM25/HyDE/IB rerank）。完整 RAG 路径请用 run_eval.sh。",
    })


@app.route("/images/<path:filename>")
def images(filename):
    """题目图片（本地或 URL）。"""
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/api/health")
def api_health():
    return jsonify({
        "questions_loaded": len(QUESTIONS),
        "predictions_loaded": len(PREDICTIONS),
        "metrics_total": METRICS.get("total"),
        "metrics_accuracy": METRICS.get("accuracy"),
    })


if __name__ == "__main__":
    print(f"[frontend] questions={len(QUESTIONS)} predictions={len(PREDICTIONS)} accuracy={METRICS.get('accuracy')}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
