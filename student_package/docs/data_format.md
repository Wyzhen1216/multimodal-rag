# 数据格式

## 输入

输入文件是 JSON 数组，每个元素是一道题：

```json
{
  "index": 1,
  "section": "选择QA",
  "question": "问题文本",
  "image_url": ""
}
```

| 字段 | 说明 |
|---|---|
| `index` | 题目 ID |
| `section` | 题型：`选择QA` / `选择VQA` / `判断QA` / `判断VQA` / `简答QA` / `简答VQA`（VQA 表示带图片） |
| `question` | 题目文本 |
| `image_url` | 图片 URL；纯文本题为空字符串。图片已预下载到 `data/images/`，文件名与 URL 末尾一致。输入文件为 `data/public_examples.full.json`（包含 `index`、`section`、`question`、`image_url` 四个公开字段，其余字段不得读取） |

## 输出

输出文件为 JSONL，每行一道题：

```json
{"index": 1, "answer": "B", "raw_answer": "答案：B"}
```

必含字段：

| 字段 | 说明 |
|---|---|
| `index` | 题目 ID，需与输入对应 |
| `answer` | 最终答案，用于评分 |
| `raw_answer` | （建议）模型原始输出，便于排查 |

**禁止**在输出中包含：`contexts`、`error`、`expected_answer`、`material`、`think`。

## time_cost.jsonl

除 `predictions.jsonl` 外，还需输出 `time_cost.jsonl`，记录每题的检索耗时和推理耗时：

```json
{"index": 1, "retrieval_cost": 1.23, "infer_cost": 3.45}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `index` | int | 题目 ID，需与输入对应 |
| `retrieval_cost` | float | 检索耗时（秒），从开始构造检索 query 到获得检索结果 |
| `infer_cost` | float | 推理耗时（秒），从 prompt 送入模型到拿到完整回复 |

计时边界说明：
- `retrieval_cost`：从 query 构造完成、准备发起检索开始，到 DashVector 返回结果结束。包含 embedding 计算+网络请求时间
- `infer_cost`：从组织完 prompt、调用 `model.generate()` 开始，到拿到完整生成结果结束

> 因各组的检索策略和并行方式不同，时间请自行在代码中统计。

## 答案格式

| 题型 | answer 要求 | 示例 |
|---|---|---|
| 选择（QA / VQA） | 选项字母，多选无分隔符 | `A`、`ABD` |
| 判断（QA / VQA） | `正确` 或 `错误` | `正确` |
| 简答（QA / VQA） | 简短答案文本 | `战国中晚期` |
