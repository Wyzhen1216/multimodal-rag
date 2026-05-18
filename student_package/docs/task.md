# 课题任务说明

实现一个面向历史问答场景的多模态 RAG Agent。系统输入为 `question` 和可选 `image_url`，需要结合历史知识库检索结果，调用 Qwen2.5-VL 生成答案。

## 必做内容

1. 读取输入 JSON（`data/public_examples.full.json`）。
2. 使用 `question` 和 `image_url` 构造检索 query。
3. 调用向量库（DashVector）检索历史资料。
4. 组织检索结果，构造 prompt。
5. 调用 Qwen2.5-VL 生成答案。
6. 输出 `predictions.jsonl`。
7. 输出 `time_cost.jsonl`（每题检索与推理耗时，见数据格式文档）。
8. 跑通 `bash run_eval.sh`。

## 说明

- 所有代码写在 `solution/` 目录下。入口为 `solution/main.py`。
- 不得读取或依赖隐藏字段（`expected_answer`、`material`、`think` 等）。正式评测输入不会提供这些字段。
- 不限制具体检索策略，可自行决定检索、生成和中间处理流程。
- 必须在 `solution/requirements.txt` 中声明你的 Python 依赖。评测时会基于该文件创建 venv 并安装。
