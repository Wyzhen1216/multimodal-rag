from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    args = parse_args()
    gold_rows = load_json_array(Path(args.gold))
    prediction_rows = load_jsonl(Path(args.prediction))
    predictions = {str(row.get("index") or row.get("id")): row for row in prediction_rows}

    total = correct = 0
    by_section = defaultdict(lambda: {"total": 0, "correct": 0})
    for gold in gold_rows:
        sample_id = str(gold.get("index") or gold.get("id"))
        section = str(gold.get("section") or "")
        pred_answer = str((predictions.get(sample_id) or {}).get("answer") or "")
        gold_answer = str(gold.get("expected_answer") or "")
        ok = is_answer_match(section, pred_answer, gold_answer)
        total += 1
        correct += int(ok)
        by_section[section]["total"] += 1
        by_section[section]["correct"] += int(ok)

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_section": {
            section: {
                **counts,
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for section, counts in sorted(by_section.items())
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate public examples.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def is_answer_match(section: str, predicted: str, gold: str) -> bool:
    from src.normalization import normalize_answer

    pred_norm = normalize_answer(section, predicted)
    gold_norm = normalize_answer(section, gold) or normalize_answer(section, f"答案：{gold}")
    return bool(pred_norm and gold_norm and pred_norm == gold_norm)


def load_json_array(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

