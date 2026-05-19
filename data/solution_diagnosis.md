# Solution 现状诊断与优化方案

> 对象：[solution/](../solution/) 当前实现 + [test/outputs/20260518_195647_compare/](../test/outputs/20260518_195647_compare/) 跑分结果
> 配套：[data_analysis.md](data_analysis.md)、[task_decomposition.md](task_decomposition.md)
> 写作时间：2026-05-18

## 一、当前跑分基线

| 策略 | 准确率 | 正确数 | 检索耗时 | 推理耗时 |
|---|---|---|---|---|
| baseline | **44.0%** | 22/50 | 0.000s | 0.448s |
| rag (TF-IDF+BM25) | **30.0%** | 15/50 | 0.000s | 0.936s |

**关键发现：RAG 路线反而比 baseline 低 14 个百分点。**

## 二、按题型拆解的准确率

| section | baseline | rag | gold 难点 |
|---------|----------|-----|-----------|
| 判断QA   | 63.6%    | 45.5% | 模型偏"保守负"，6 道把"正确"判成"错误" |
| 判断VQA  | 66.7%    | 22.2% | 同上，RAG 噪声雪上加霜 |
| 选择QA   | 40.0%    | 20.0% | 多选漏选 / 全选误判 |
| 选择VQA  | 33.3%    | 44.4% | 视觉细粒度判断弱 |
| 简答QA   | **0.0%** | **0.0%** | 4 道全错 — 知识盲区 + 检索无效 |
| 简答VQA  | 28.6%    | 28.6% | 识字 / 计数 / 方向，几乎纯视觉 |

## 三、致命缺陷

### 🔴 缺陷 1：评测环境根本跑不起来

```python
# solution/llm_client.py:42-43
if not self.api_key:
    raise RuntimeError("QWEN_API_KEY is not set")
```

- 评测脚本 `run_eval.sh` 设置 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，明确**离线本地推理**
- `configs/env.example` 只提供 `QWEN_MODEL_PATH=/inspire/.../qwen2.5-VL-7B-Instruct`（本地路径），**没有 QWEN_API_KEY**
- 评测一启动就 raise，整个 run 失败 → 0 分

**影响**：致命。即使本地 dev 跑得再好，正式评测也是 0 分。

### 🔴 缺陷 2：RAG 读取了 `expected_answer`，违反任务规则

```python
# solution/prompts.py:43
r_a = str(r.get("expected_answer") or "")
```

- `docs/task.md` 明令禁止读 `expected_answer` / `material` / `think`
- 隐藏测试集中 `expected_answer` 字段**根本不存在** → RAG 上下文里全是空字符串
- 当前 RAG 看起来"低分"，是因为在公开样例上还能侥幸偷到部分答案；一旦上隐藏集，**RAG 路线必然 0 增益甚至负增益**

**影响**：致命。即使评测环境能跑，这个实现的 RAG 没有任何价值。

### 🔴 缺陷 3：根本没有用 DashVector + GME 检索

- `docs/task.md` 第 3 步明确要求 "**调用向量库(DashVector)检索历史资料**"
- 当前实现完全没引入 `dashvector` SDK，没用 `GME_MODEL_PATH` 算 embedding
- 用公开样例 50 题做"互检索"是错误理解 — 课题要求检索的是**外部历史文献语料库**（`DASHVECTOR_COLLECTION=ccv_cb_student_20260508`）

**影响**：致命。这是课题的核心任务，没做就等于跑题。

### 🟠 缺陷 4：Prompt 没有题型分支

`prompts.py` 一套 system prompt 打 6 种题型。错误分布证实了这一点：

| 错因 | 数量 | 案例 |
|---|---|---|
| 选择题数量不一致 | 7 | 题目 5589：gold=`B`，预测=`BC`；题目 5462：gold=`ABCD`，预测=`AC` |
| 判断题正/误颠倒 | 7 | **6 道把"正确"判成"错误"** —— 模型有严重的负偏 |
| 简答题内容不一致 | 9 | `2个` vs `2`（带量词），`铜权` vs `铜钾`（错答） |
| 选择题选项不同 | 5 | 视觉细节看错 |

### 🟠 缺陷 5：time_cost 计时不准确

`retrieval_cost` 当前实际包含的是"本地 TF-IDF + prompt 构造"，平均 0.000s。但 `docs/data_format.md` 定义是：

> "从开始构造检索 query 到获得检索结果。包含 embedding 计算 + 网络请求时间"

`infer_cost` 包含 retry sleep（[llm_client.py:94](../solution/llm_client.py#L94) `time.sleep(1.5 * attempt)`），也不符合定义。

### 🟡 缺陷 6：简答 VQA 处理粗糙

- 简答 VQA 准确率 28.6%，错例典型：
  - 题目 6234："穿孔数量是多少？" gold=`2`，预测=`2个` → 归一化后多了"个"丢分
  - 题目 6164："猴子握持什么？" gold=`链条`，预测=`铁环` → 视觉看错
- 该题型**几乎不需要检索**，强行检索还会引入噪声
- 缺少**多次采样投票**机制，单次推理对视觉细节题不稳定

### 🟡 缺陷 7：判断题"保守偏负"

- 7 道判断题错误中 **6 道**是 gold=正确、模型答错误（即"假阴性"）
- 说明模型默认倾向于"挑陈述里的瑕疵判错误"
- 当前 prompt 没有任何"如果检索资料支持，则倾向于正确"的引导

### 🟡 缺陷 8：选择题"漏选"压倒"全选"

- 多选错误中漏选（gold=ABCD 预测=BD）多于全选误判
- 模型默认保守只选最有把握的
- 当前 prompt 没有"如果几个选项都被检索资料支持，则一起选"的引导

### 🟢 次要 9：图片走 base64 + DashScope 上传

- 当前 [llm_client.py:51-65](../solution/llm_client.py#L51-L65) 用 base64 编码图片走 OpenAI 接口
- 本地 transformers 推理时，应该直接传 PIL Image 或本地路径，免去 base64 开销

### 🟢 次要 10：错误时输出空 raw_answer

- [main.py:106-107](../solution/main.py#L106-L107) 异常时 `raw=""`、`answer=""`
- 但合规校验脚本不接受**缺失** `answer` 字段，**空字符串**虽合规但等于送分 0
- 应该按题型给"兜底默认值"（选择题 `A`、判断题 `正确`、简答 `无`）

---

## 四、优化方案（按 ROI 排序）

### Tier 1：必须改（不改就 0 分）

#### A. 重写 `llm_client.py`：本地 transformers + API 双模式
- 增加 `LocalQwenVL` 类：用 `AutoModelForVision2Seq` + `AutoProcessor` 本地加载
- 保留 `ApiQwenVL` 走 OpenAI 兼容接口（本地 dev 用）
- 通过 `AGENT_MODE` 或环境变量自动选择
  - `QWEN_API_KEY` 在 → API 模式
  - 否则 → 本地模式
- 启动时**一次性加载**，循环复用

#### B. 重写 `retriever.py`：真 DashVector + GME embedding
- 引入 `dashvector` SDK
- 用 `GME_MODEL_PATH` 加载多模态 embedding（也支持 fallback：纯文本 sentence-transformers）
- 提供：`DashVectorRetriever(endpoint, api_key, collection)`
- 检索时同时支持文本和图（多模态 embedding）
- 提供降级开关 `DISABLE_RETRIEVAL=1` 用于消融测试

#### C. 删除所有读取 `expected_answer` 的逻辑
- `prompts.py` 中 `build_rag_user` 改为：拼接的是检索到的**文献片段**（DashVector 返回的 doc text/payload），不是其他题的答案
- 公开样例之间的"互检索"作为 fallback 完全删除

### Tier 2：高 ROI（预计 +10~20 个百分点）

#### D. 重写 `prompts.py`：题型分支 + 反陷阱约束

每个题型一套独立模板，**强约束输出格式**：

**选择题**：
```
你正在做一道多模态多项选择题。
- 题型可能是单选或多选，请仔细判断
- 题干末尾的 A/B/C/D 是所有候选选项
- 输出格式：仅输出大写字母，多选连写无分隔符（例如 A 或 ABD 或 ABCD）
- 不要输出"答案"、"我选"等前缀
- 不要任何解释
[检索资料]：{contexts}
[题目]：{question}
[图]：（如附）
```

**判断题**（重点修正负偏）：
```
你正在做一道判断题。请阅读检索到的资料，判断给定陈述是否符合资料。
- 输出格式：仅输出"正确"或"错误"两个字之一
- 重要原则：若检索资料明确支持该陈述，输出"正确"；只有当资料明确否定或与陈述冲突时才输出"错误"；资料不充分时倾向输出"正确"
- 不要同时使用"正确""错误"两个词
- 不要任何解释
```

**简答题**（重点修正量词、前缀）：
```
你正在做一道简答题。请输出最简短的答案。
- 输出 ≤ 6 个汉字
- 数字答案直接写数字（如"2"而不是"2个"）
- 时期答案直接写朝代（如"战国中晚期"而不是"该城始筑于战国中晚期"）
- 不要"答："、"答案："、"为"等前缀
- 不要句号、引号
- 第一行就是最终答案
```

#### E. 简答 VQA 跳过检索 + 多次采样投票
- `if section == "简答VQA": skip_retrieval = True`
- temperature=0.7，采样 3 次取多数票
- 投票时归一化后再比较

### Tier 3：精打细算

#### F. 修正 time_cost 计时点
- `retrieval_cost`：从"构造好 query → 调用 retriever.search()" 开始计时，到拿到 results
- `infer_cost`：仅围绕 `model.generate()`，排除 retry sleep
- 用 `time.perf_counter()`

#### G. 答案兜底
- 选择题 raw 输出归一化失败 → 默认 `A`
- 判断题归一化失败 → 默认 `正确`（基于公开样例先验）
- 简答题归一化失败 → 用 raw_answer 第一行强行截前 6 字

#### H. 断点续跑
- 跑前先读 predictions.jsonl 已有的 index，跳过已完成的题

---

## 五、改造后的目录结构

```
solution/
├── __init__.py
├── main.py              # 入口：参数解析、题型路由、计时埋点、循环
├── llm_client.py        # 本地 transformers + API 双模式
├── retriever.py         # 真 DashVector + GME，含降级
├── prompts.py           # 6 种题型分支模板 + 反陷阱约束
├── postprocess.py       # 答案归一化预校验 + 兜底
└── requirements.txt     # +dashvector, +sentence-transformers (fallback)
```

## 六、预期收益

| 改动 | 预期 +acc |
|------|-----------|
| 题型分支 prompt（D） | +8~15% |
| 真 DashVector 检索（B/C） | +10~20% |
| 简答 VQA 跳过检索 + 投票（E） | +5% |
| 答案兜底（G） | +1~3% |
| 计时正确（F） | 不影响 acc 但保证评测不 fail |

**乐观估计：从 44% → 65–75%。**
**保底（即使 DashVector 失效）：题型分支单独就能把 baseline 推到 55%+。**
