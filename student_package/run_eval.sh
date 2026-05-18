#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---- solution directory (edit this for different students) -----------------
export SOLUTION_DIR="${SOLUTION_DIR:-$ROOT_DIR/solution}"

if [[ ! -d "$SOLUTION_DIR" ]]; then
  echo "ERROR: SOLUTION_DIR='$SOLUTION_DIR' does not exist." >&2
  exit 1
fi

SOLUTION_DIR="$(cd "$SOLUTION_DIR" && pwd)"
SOLUTION_PARENT="$(dirname "$SOLUTION_DIR")"
SOLUTION_NAME="$(basename "$SOLUTION_DIR")"
OUTPUT_DIR="$ROOT_DIR/outputs/$(basename "$SOLUTION_PARENT")_${SOLUTION_NAME}"
OUTPUT_FILE="$OUTPUT_DIR/predictions.jsonl"
METRICS_FILE="$OUTPUT_DIR/metrics.json"

# ---- input / gold ---------------------------------------------------------
INPUT_FILE="${1:-$ROOT_DIR/data/public_examples.full.json}"
GOLD_FILE="${GOLD_FILE:-$ROOT_DIR/data/public_examples.full.json}"

# ---- model & vector-db config ----------------------------------------------
# 路径以课题评测机为准（/inspire/qb-ilm2/project/26summer-camp-16/public/...）
# 如果在其他环境跑，外部 export 同名变量即可覆盖
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/inspire/qb-ilm2/project/26summer-camp-16/public/Rag_Based_HistAgent/Models/qwen2.5-VL-7B-Instruct}"
export GME_MODEL_PATH="${GME_MODEL_PATH:-/inspire/qb-ilm2/project/26summer-camp-16/public/Rag_Based_HistAgent/Models/gem-Qwen2-VL-7B-Instruct}"
export DASHVECTOR_ENDPOINT="${DASHVECTOR_ENDPOINT:-vrs-cn-71j49hani00061.dashvector.cn-hangzhou.aliyuncs.com}"
export DASHVECTOR_API_KEY="${DASHVECTOR_API_KEY:-sk-REPLACE-WITH-YOUR-OWN-VECTOR-KEY}"
export DASHVECTOR_COLLECTION="${DASHVECTOR_COLLECTION:-ccv_cb_student_20260508}"
export AGENT_MODE="${AGENT_MODE:-real}"

# ---- python / venv ---------------------------------------------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
fi

if [[ ! -f "$SOLUTION_DIR/requirements.txt" ]]; then
  echo "ERROR: $SOLUTION_DIR/requirements.txt not found." >&2
  exit 1
fi

VENV_DIR="$ROOT_DIR/.venv-$(basename "$SOLUTION_PARENT")_${SOLUTION_NAME}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv for $SOLUTION_DIR (--system-site-packages) ..."
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  echo "==> Installing $SOLUTION_DIR/requirements.txt ..."
  "$VENV_DIR/bin/pip" install -q -r "$SOLUTION_DIR/requirements.txt"
fi
PYTHON_BIN="$VENV_DIR/bin/python"

# ---- run student solution --------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "==> Running $SOLUTION_DIR ..."
cd "$SOLUTION_PARENT"
"$PYTHON_BIN" -m "${SOLUTION_NAME}.main" \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE"
cd "$ROOT_DIR"

# ---- validate & score ------------------------------------------------------
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo ""
echo "==> Validating output format ..."
"$PYTHON_BIN" scripts/check_output.py \
  --input "$INPUT_FILE" \
  --prediction "$OUTPUT_FILE"

echo ""
echo "==> Scoring ..."
if [[ -n "$GOLD_FILE" && -f "$GOLD_FILE" ]]; then
  "$PYTHON_BIN" scripts/evaluate.py \
    --gold "$GOLD_FILE" \
    --prediction "$OUTPUT_FILE" \
    --output "$METRICS_FILE"
  echo ""
  echo "solution   : $SOLUTION_DIR"
  echo "prediction : $OUTPUT_FILE"
  echo "metrics    : $METRICS_FILE"
else
  echo "No gold file found, skipping scoring."
fi
