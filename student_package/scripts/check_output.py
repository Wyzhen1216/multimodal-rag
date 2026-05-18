from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    args = parse_args()
    input_rows = load_json_array(Path(args.input))
    prediction_rows = load_jsonl(Path(args.prediction))

    expected_ids = [str(row.get("index") or row.get("id")) for row in input_rows]
    seen_ids = []
    errors = []
    for row_no, row in enumerate(prediction_rows, start=1):
        sample_id = str(row.get("index") or row.get("id") or "")
        if not sample_id:
            errors.append(f"line {row_no}: missing index")
            continue
        seen_ids.append(sample_id)
        if "answer" not in row:
            errors.append(f"line {row_no}: missing answer")
        if "contexts" in row or "error" in row:
            errors.append(f"line {row_no}: output must not contain contexts or error")
        if "expected_answer" in row or "material" in row or "think" in row:
            errors.append(f"line {row_no}: output must not contain hidden fields")

    duplicate_ids = sorted({sample_id for sample_id in seen_ids if seen_ids.count(sample_id) > 1})
    if duplicate_ids:
        errors.append(f"duplicate predictions: {duplicate_ids[:10]}")
    missing_ids = [sample_id for sample_id in expected_ids if sample_id not in set(seen_ids)]
    extra_ids = [sample_id for sample_id in seen_ids if sample_id not in set(expected_ids)]
    if missing_ids:
        errors.append(f"missing predictions: {missing_ids[:10]} total={len(missing_ids)}")
    if extra_ids:
        errors.append(f"extra predictions: {extra_ids[:10]} total={len(extra_ids)}")

    # --- time_cost.jsonl validation ---
    time_errors = check_time_cost(Path(args.prediction))
    if time_errors:
        errors.extend(time_errors)

    summary = {
        "input_rows": len(input_rows),
        "prediction_rows": len(prediction_rows),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        return 1
    return 0


def check_time_cost(prediction_path: Path) -> list[str]:
    """Validate time_cost.jsonl exists and format is correct."""
    time_cost_path = prediction_path.parent / "time_cost.jsonl"
    errors = []
    if not time_cost_path.exists():
        errors.append(f"missing time_cost file: {time_cost_path}")
        return errors

    time_rows = load_jsonl(time_cost_path)
    pred_rows = load_jsonl(prediction_path)

    if len(time_rows) != len(pred_rows):
        errors.append(
            f"time_cost row count ({len(time_rows)}) != prediction row count ({len(pred_rows)})"
        )

    for row_no, row in enumerate(time_rows, start=1):
        for key in ("index", "retrieval_cost", "infer_cost"):
            if key not in row:
                errors.append(f"time_cost line {row_no}: missing {key}")
        for key in ("retrieval_cost", "infer_cost"):
            if key in row and not isinstance(row[key], (int, float)):
                errors.append(f"time_cost line {row_no}: {key} must be a number")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prediction JSONL format.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction", required=True)
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_no}")
            rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
