"""
错误样例分析: 对比 predictions.jsonl 与 gold, 输出错误清单与统计。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List


def _load_json_array(p: Path) -> list[dict]:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(
    gold_path: Path,
    prediction_path: Path,
    time_cost_path: Path | None,
    out_dir: Path,
    strategy: str,
) -> dict:
    """落地 errors.jsonl + error_analysis.md, 返回汇总指标。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评分函数复用项目内的归一化逻辑
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.normalization import normalize_answer  # noqa: E402

    gold = _load_json_array(gold_path)
    preds = {str(r.get("index")): r for r in _load_jsonl(prediction_path)}
    times = {}
    if time_cost_path and time_cost_path.exists():
        times = {str(r.get("index")): r for r in _load_jsonl(time_cost_path)}

    by_section: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    errors: list[dict] = []
    retrieval_costs: list[float] = []
    infer_costs: list[float] = []

    for item in gold:
        idx = str(item.get("index"))
        section = str(item.get("section") or "")
        gold_ans = str(item.get("expected_answer") or "")
        pred_row = preds.get(idx) or {}
        pred_ans = str(pred_row.get("answer") or "")
        raw_ans = str(pred_row.get("raw_answer") or "")

        gold_norm = normalize_answer(section, gold_ans) or normalize_answer(section, f"答案:{gold_ans}")
        pred_norm = normalize_answer(section, pred_ans)
        ok = bool(pred_norm and gold_norm and pred_norm == gold_norm)

        by_section[section]["total"] += 1
        by_section[section]["correct"] += int(ok)

        if not ok:
            errors.append(
                {
                    "index": item.get("index"),
                    "section": section,
                    "question": item.get("question"),
                    "image_url": item.get("image_url") or "",
                    "expected": gold_ans,
                    "expected_norm": gold_norm,
                    "predicted": pred_ans,
                    "predicted_norm": pred_norm,
                    "raw_answer": raw_ans,
                }
            )

        if idx in times:
            r = times[idx]
            retrieval_costs.append(float(r.get("retrieval_cost") or 0.0))
            infer_costs.append(float(r.get("infer_cost") or 0.0))

    total = sum(v["total"] for v in by_section.values())
    correct = sum(v["correct"] for v in by_section.values())

    # ---- 落 errors.jsonl ----
    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- 错因聚类: 选择 -> 选项漂移; 判断 -> 反义; 简答 -> 拼写/范围 ----
    err_type_counter: Counter = Counter()
    for e in errors:
        sec = e["section"]
        exp_n = e["expected_norm"]
        pred_n = e["predicted_norm"]
        if not pred_n:
            err_type_counter["输出为空/无法解析"] += 1
        elif "选择" in sec:
            if len(pred_n) != len(exp_n):
                err_type_counter["选择题: 选项数量不一致"] += 1
            else:
                err_type_counter["选择题: 选项不同"] += 1
        elif "判断" in sec:
            err_type_counter["判断题: 正/误颠倒"] += 1
        else:
            err_type_counter["简答题: 内容不一致"] += 1

    err_by_section = Counter(e["section"] for e in errors)

    # ---- 落 error_analysis.md ----
    md: list[str] = []
    md.append(f"# 错误样例分析 — {strategy}")
    md.append("")
    md.append(f"- 输入 gold: `{gold_path}`")
    md.append(f"- 预测文件: `{prediction_path}`")
    md.append(f"- 总数: **{total}**  正确: **{correct}**  准确率: **{correct/total*100:.1f}%**  错误: **{len(errors)}**")
    if retrieval_costs:
        md.append(
            f"- 平均检索耗时: {sum(retrieval_costs)/len(retrieval_costs):.3f}s  "
            f"平均推理耗时: {sum(infer_costs)/len(infer_costs):.3f}s"
        )
    md.append("")
    md.append("## 按题型准确率")
    md.append("")
    md.append("| section | total | correct | accuracy |")
    md.append("|---|---|---|---|")
    for sec in sorted(by_section):
        v = by_section[sec]
        acc = v["correct"] / v["total"] * 100 if v["total"] else 0.0
        md.append(f"| {sec} | {v['total']} | {v['correct']} | {acc:.1f}% |")
    md.append("")
    md.append("## 错因分布")
    md.append("")
    md.append("| 错因类型 | 数量 |")
    md.append("|---|---|")
    for k, c in err_type_counter.most_common():
        md.append(f"| {k} | {c} |")
    md.append("")
    md.append("## 各题型错误数")
    md.append("")
    md.append("| section | error count |")
    md.append("|---|---|")
    for sec, c in err_by_section.most_common():
        md.append(f"| {sec} | {c} |")
    md.append("")
    md.append("## 错误样例 (按 index 排序)")
    md.append("")
    for e in sorted(errors, key=lambda x: x.get("index") or 0):
        md.append(f"### index={e['index']}  ({e['section']})")
        md.append("")
        md.append(f"- 期望: `{e['expected']}` (norm=`{e['expected_norm']}`)")
        md.append(f"- 预测: `{e['predicted']}` (norm=`{e['predicted_norm']}`)")
        if e["image_url"]:
            md.append(f"- image: {e['image_url']}")
        md.append("")
        md.append(f"**题目**: {e['question']}")
        md.append("")
        raw = (e["raw_answer"] or "").strip()
        if raw:
            md.append("**模型原始输出**:")
            md.append("")
            md.append("```")
            md.append(raw[:800])
            md.append("```")
        md.append("")

    (out_dir / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")

    return {
        "strategy": strategy,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_section": {
            sec: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for sec, v in by_section.items()
        },
        "error_count": len(errors),
        "error_types": dict(err_type_counter),
        "avg_retrieval_cost": (sum(retrieval_costs) / len(retrieval_costs)) if retrieval_costs else 0.0,
        "avg_infer_cost": (sum(infer_costs) / len(infer_costs)) if infer_costs else 0.0,
    }
