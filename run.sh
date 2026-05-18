ROOT=/inspire/qb-ilm2/project/26summer-camp-16/26240052/multimodal-rang/student_package
PUB=/inspire/qb-ilm2/project/26summer-camp-16/public/Rag_Based_HistAgent

# ============================================================================
# 必备环境变量(评测机配置)
# ============================================================================
export QWEN_MODEL_PATH=$PUB/Models/qwen2.5-VL-7B-Instruct
export GME_MODEL_PATH=$PUB/Models/gem-Qwen2-VL-7B-Instruct
export DASHVECTOR_ENDPOINT=vrs-cn-71j49hani00061.dashvector.cn-hangzhou.aliyuncs.com
export DASHVECTOR_API_KEY=sk-REPLACE-WITH-YOUR-OWN-VECTOR-KEY
export DASHVECTOR_COLLECTION=ccv_cb_student_20260508
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT:$PYTHONPATH

cd "$ROOT" || exit 1
mkdir -p outputs

# ============================================================================
# [1/7] 当前环境基本信息
# ============================================================================
echo "================================================================"
echo "[1/7] ENVIRONMENT"
echo "================================================================"
echo "Date     : $(date -Iseconds)"
echo "Host     : $(hostname)"
echo "User     : $USER  (cwd=$PWD)"
echo "Python   : $(which python)  ($(python -V 2>&1))"
echo "CUDA     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2 || echo 'no nvidia-smi')"
echo

# ============================================================================
# [2/7] 路径检查(模型权重 / 数据 / solution 入口)
# ============================================================================
echo "================================================================"
echo "[2/7] PATH CHECK"
echo "================================================================"
for p in \
    "$QWEN_MODEL_PATH" \
    "$GME_MODEL_PATH" \
    "$ROOT/data/public_examples.full.json" \
    "$ROOT/data/images" \
    "$ROOT/solution/main.py" \
    "$ROOT/solution/retriever.py" \
    "$ROOT/scripts/evaluate.py"
do
    if [ -e "$p" ]; then
        sz=$(du -sh "$p" 2>/dev/null | awk '{print $1}')
        printf "  ✓  %-70s  %s\n" "$p" "$sz"
    else
        printf "  ✗  %-70s  MISSING\n" "$p"
    fi
done
echo

# ============================================================================
# [3/7] Python 依赖
# ============================================================================
echo "================================================================"
echo "[3/7] PYTHON DEPS"
echo "================================================================"
python - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "qwen_vl_utils", "PIL",
        "dashvector", "dashscope", "openai"]
for m in mods:
    try:
        x = importlib.import_module(m)
        v = getattr(x, "__version__", "?")
        print(f"  ✓  {m:20s}  {v}")
    except Exception as e:
        print(f"  ✗  {m:20s}  {type(e).__name__}: {e}")
print()
import torch
print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    cuda:{i}  {p.name}  {p.total_memory/1e9:.1f}GB")
PY
echo

# ============================================================================
# [4/7] DashVector 网络可达性(3s 超时,不阻塞主流程)
# ============================================================================
echo "================================================================"
echo "[4/7] DASHVECTOR NETWORK"
echo "================================================================"
HOST="$DASHVECTOR_ENDPOINT"
IP=$(getent hosts "$HOST" | awk '{print $1}' | head -1)
echo "  DNS resolve  : $HOST  →  ${IP:-FAILED}"
for port in 80 443 9000; do
    if [ -n "$IP" ]; then
        if timeout 3 bash -c "exec 3<>/dev/tcp/$IP/$port" 2>/dev/null; then
            echo "  TCP $port      : OK"
        else
            echo "  TCP $port      : BLOCKED/TIMEOUT"
        fi
    fi
done
echo "  → 调一次 stats() 看 SDK 返回:"
python - <<'PY' 2>&1 | sed 's/^/  /'
import os
try:
    from dashvector import Client
    c = Client(api_key=os.environ["DASHVECTOR_API_KEY"],
               endpoint=os.environ["DASHVECTOR_ENDPOINT"])
    coll = c.get(os.environ["DASHVECTOR_COLLECTION"])
    s = coll.stats()
    print("stats:", s)
except Exception as e:
    print(f"stats failed: {type(e).__name__}: {e}")
PY
echo

# ============================================================================
# [5/7] 跑 solution.main(--limit 2 快速验证)
# ============================================================================
echo "================================================================"
echo "[5/7] RUN solution.main --limit 2"
echo "================================================================"
START=$(date +%s)
python -m solution.main \
    --input data/public_examples.full.json \
    --output outputs/predictions.jsonl \
    --limit 2
END=$(date +%s)
echo "  elapsed: $((END-START))s"
echo

# ============================================================================
# [6/7] 输出展示
# ============================================================================
echo "================================================================"
echo "[6/7] OUTPUT FILES"
echo "================================================================"
echo "--- predictions.jsonl ---"
cat outputs/predictions.jsonl 2>/dev/null
echo
echo "--- time_cost.jsonl ---"
cat outputs/time_cost.jsonl 2>/dev/null
echo

# ============================================================================
# [7/7] 格式校验 + 评分
# ============================================================================
echo "================================================================"
echo "[7/7] CHECK + EVALUATE"
echo "================================================================"
python scripts/check_output.py \
    --input data/public_examples.full.json \
    --prediction outputs/predictions.jsonl 2>&1 | head -10
echo
python scripts/evaluate.py \
    --gold data/public_examples.full.json \
    --prediction outputs/predictions.jsonl \
    --output outputs/metrics.json 2>&1 | head -20
echo
echo "================================================================"
echo "FINISHED"
echo "================================================================"
