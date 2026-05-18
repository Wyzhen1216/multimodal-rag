# test/ —— 简易测试套件

针对 `solution/` 提供两种模式的对比测试:

- **baseline**: 仅"题型分支 prompt"（关 few-shot、关 VQA 多采样投票）
- **enhanced**: 默认全开（few-shot + 简答VQA 多采样投票）

> 本地无 DashVector / GME 模型，retriever 自动降级为 NullRetriever
> （由 `DISABLE_RETRIEVAL=1` 显式禁用），效果等同闭卷推理。
> 正式评测环境需另外 `export DASHVECTOR_*` + `GME_MODEL_PATH` 启用检索。

每次测试都会在 `test/outputs/<时间戳>_<模式>/` 下生成:

```text
predictions.jsonl       # 模型预测
time_cost.jsonl         # 检索与推理耗时
run.log                 # 完整 stdout/stderr
meta.json               # 命令行/退出码/host/汇总指标
metrics.json            # 准确率 (复用 scripts/evaluate.py)
errors.jsonl            # 错误样例明细
error_analysis.md       # 错因聚类 + 单条错误样例
solution_snapshot/      # 本次跑的 solution/ 代码副本
```

## 用法

```bash
cd student_package

# 仅题型分支 prompt
python test/run_test.py --mode baseline

# 完整 prompt 栈（few-shot + VQA 投票）
python test/run_test.py --mode enhanced

# 调试: 只跑前 5 条
python test/run_test.py --mode enhanced --limit 5

# 一次跑两种模式并生成对比报告
python test/compare_strategies.py
```

## 环境变量

- `QWEN_API_KEY` (必需): DashScope / OpenAI 兼容服务的 API Key
- `QWEN_MODEL_NAME` (可选, 默认 `qwen2.5-vl-7b-instruct`)
- `QWEN_BASE_URL`   (可选, 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `DISABLE_RETRIEVAL=1` (本地默认): 禁用 DashVector 检索，闭卷推理

`run_test.py` / `compare_strategies.py` 已经把题目中提供的 API key 作为默认值塞进 env, 直接跑就行。
