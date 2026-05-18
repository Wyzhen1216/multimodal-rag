# 评分说明

最终评分以隐藏测试集为准。`data/public_examples.full.json` 是公开样例，用于开发过程中自测。

## 评测方式

```text
选择题：归一化选项字母后精确匹配
判断题：归一化"正确 / 错误"后精确匹配
简答题：填空式短答案匹配
```

归一化逻辑见 `src/normalization.py`。

## 自测

```bash
# 1. 设置环境变量（模型路径、向量库凭证等）
export QWEN_MODEL_PATH=/path/to/model
export GME_MODEL_PATH=/path/to/embedding
# ...

# 2. 一键评测
bash run_eval.sh
```

脚本依次执行：运行 `solution.main` → 格式校验（含 `time_cost.jsonl` 校验）→ 评分，结果写入 `outputs/` 目录。

正式评测使用相同流程，仅环境变量由评测机统一配置。

## 时间统计

`time_cost.jsonl` 作为参考指标，统计检索和推理耗时，**不参与答案评分**。格式要求见数据格式文档。
