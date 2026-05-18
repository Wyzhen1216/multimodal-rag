# test/ —— 简易测试套件

针对 `solution/` 提供两种策略的对比测试:

- **baseline**: 无任何检索, 直接调用 Qwen2.5-VL
- **rag**: 借鉴 `chuangzhi-summer-camp` 的 TF-IDF + BM25 RRF 融合检索, 在公开样例上做"留一"相似题检索作为 few-shot

每次测试都会在 `test/outputs/<时间戳>_<策略>/` 下生成:

```
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

# 基线 (无检索)
python test/run_test.py --strategy baseline

# 检索策略
python test/run_test.py --strategy rag

# 调试: 只跑前 5 条
python test/run_test.py --strategy baseline --limit 5

# 一次跑两种策略并生成对比报告
python test/compare_strategies.py
```

## 环境变量

- `QWEN_API_KEY` (必需): DashScope / OpenAI 兼容服务的 API Key
- `QWEN_MODEL_NAME` (可选, 默认 `qwen2.5-vl-7b-instruct`)
- `QWEN_BASE_URL`   (可选, 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`)

`run_test.py` / `compare_strategies.py` 已经把题目中提供的 API key 作为默认值塞进 env, 直接跑就行。
