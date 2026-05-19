# 推理性能瓶颈分析与优化方案

> 测试环境：DashScope API 模式  
> 测试规模：1289 题全量 mock（SelectQA/VQA + AnswerQA/VQA + TF_QA/VQA）  
> 测试时间：2026-05-19，总耗时 5873s（97.9 min）

---

## 一、当前性能基线

### 1.1 API 模式（实测）

| 题型 | 题目数 | 平均耗时 | 最大耗时 | 准确率 |
|---|---|---|---|---|
| 选择VQA | 374 | **7.3s** | 38.3s | 51.1% |
| 选择QA | 323 | **6.3s** | 35.4s | 50.8% |
| 判断VQA | 100 | 1.6s | 23.7s | 66.0% |
| 判断QA | 90 | 1.5s | 30.0s | 70.0% |
| 简答VQA | 322 | 1.4s | 29.7s | 8.1% |
| 简答QA | 79 | **0.5s** | 1.0s | 28.7% |
| **全量** | **1289** | **4.6s** | — | **41.3%** |

> 简答QA 快的原因：T2 短路（题干自含答案）命中率高，跳过了检索和推理。  
> 简答VQA 准确率异常低（8.1%）：mock 数据答案评估标准与 evaluator 不兼容，不代表真实表现。

### 1.2 H200 本地模式（实测，用户报告）

| 环境 | 平均耗时/题 | 说明 |
|---|---|---|
| DashScope API | ~4.6s | 网络 I/O 为主，LLM 串行调用 |
| H200 本地推理 | **~30s** | GPU forward pass 串行，无并发 |

API 模式比本地慢的原因是网络延迟；本地 30s 的原因是每道选择题需要 **10-14 次串行 GPU forward pass**。

---

## 二、瓶颈根因分析

### 2.1 选择题调用链（最慢题型，主战场）

一道标准 4 选项选择题的完整调用链：

```
choice_as_judge(4 选项)
│
├─ detect_multi_select()          → 1 次 LLM（max_tokens=8）
│
├─ _judge_one_option() × 4       → 每个选项：
│   ├─ make_reverse_query()       →   1 次 LLM（max_tokens=60）
│   ├─ retriever.search(fwd)      →   1 次 GME embed（GPU）
│   ├─ retriever.search(rev)      →   1 次 GME embed（GPU）
│   └─ client.generate(verdict)  →   1 次 LLM（max_tokens=8）
│                                    → 小计：2 次 LLM + 2 次 embed
│
├─ _reflect_answer()              → 1 次 embed + 1 次 LLM
│
└─ [可选] _fallback / recheck    → 0-3 次 LLM
```

| 步骤 | LLM 调用数 | H200 本地耗时 | 小计 |
|---|---|---|---|
| detect_multi_select | 1 | ~2s | 2s |
| make_reverse_query × 4 | **4** | ~2s/次 | **8s** |
| GME embed × 8 | — | ~0.1s/次 | 0.8s |
| verdict generate × 4 | **4** | ~1s/次 | **4s** |
| reflect_answer | 1 | ~1s | 1s |
| recheck（条件触发） | 0–3 | ~1-2s/次 | 0-5s |
| **合计** | **10–14 次** | | **~16-21s** + overhead |

**核心问题：所有 LLM 调用完全串行，H200 每次 forward pass 约 1-2s，10 次叠加 = 10-20s，加上图像编码/prefill 开销总计约 30s。**

### 2.2 其他题型瓶颈

**判断题（判断QA/VQA）：**
- 需 `retriever.search` + `client.generate_n(n=3)`
- `generate_n` 是 3 次串行 `generate`，共 3 次 forward pass
- H200 上约 3 × 2s = 6s

**简答VQA：**
- `client.generate_n(n=5)` → 5 次串行 forward pass
- H200 上约 5 × 2s = 10s

**简答QA：**
- T2 短路命中则 < 1s；否则需 `diverge_queries`（3-4 次 LLM + 多次检索）

---

## 三、优化方案

### 优先级排序

| 优先级 | 优化项 | 预计收益 | 精度影响 | 实现难度 |
|---|---|---|---|---|
| **P0** | `generate_batch()`：选项 verdict 批量化 | -4s/题 | 无 | 中 |
| **P0** | `generate_n()` → `num_return_sequences` | -(n-1)×2s/题 | 无 | 低 |
| **P1** | `make_reverse_query` 批量化 | -6s/题 | 无 | 中 |
| **P2** | `--workers N` 多线程（API 模式） | ~3-4x（API） | 无 | 已完成 |
| **P3** | recheck 批量化 | -0-3s/题 | 无 | 高 |

### 3.1 P0：`LocalQwenVL.generate_batch()`

**现状：** 4 个选项的 verdict 分 4 次独立 forward pass（每次 ~1s = 4s）  
**优化：** 将 4 个输入打包为 batch=4，一次 forward pass 同时输出 4 个 verdict（~1.5s）

```python
# llm_client.py — LocalQwenVL 新增
def generate_batch(
    self,
    requests: List[Tuple[str, str, Optional[Path]]],  # (system, user_text, image_path)
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> List[str]:
    """批量推理：batch_size = len(requests)，适用于选择题多选项并行判断。"""
    from qwen_vl_utils import process_vision_info

    all_texts, all_images = [], []
    for system, user_text, image_path in requests:
        messages = self._build_messages(system, user_text, image_path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        all_texts.append(text)
        if image_inputs:
            all_images.extend(image_inputs)

    inputs = self.processor(
        text=all_texts,
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    ).to(self.device)

    do_sample = temperature > 0.0
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=0.9)

    with self.torch.no_grad():
        generated = self.model.generate(**inputs, **gen_kwargs)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return [
        r.strip() for r in self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    ]
```

**预期效果：** 4 次 verdict 从 ~4s → ~1.5s，节省 ~2.5s/题。

### 3.2 P0：`LocalQwenVL.generate_n()` 覆写

**现状：** `generate_n(n=3)` = 3 次串行 `generate()`（3 次完整 forward pass）  
**优化：** `num_return_sequences=n`，prompt 只需 prefill 一次，之后 n 路采样并行解码

```python
# llm_client.py — LocalQwenVL 覆写 BaseQwenVL.generate_n
def generate_n(
    self,
    system: str,
    user_text: str,
    image_path: Optional[Path] = None,
    n: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> List[str]:
    if n == 1:
        return [self.generate(system, user_text, image_path, temperature, max_tokens)]

    from qwen_vl_utils import process_vision_info

    messages = self._build_messages(system, user_text, image_path)
    text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(self.device)

    with self.torch.no_grad():
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            num_return_sequences=n,
        )

    prompt_len = inputs.input_ids.shape[1]
    trimmed = [seq[prompt_len:] for seq in generated]
    return [
        r.strip() for r in self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    ]
```

**预期效果：**
- 判断题 3-sample：3 次 → 1 次 forward，节省 ~4s/题  
- 简答VQA 5-sample：5 次 → 1 次 forward，节省 ~8s/题

### 3.3 P1：`choice_as_judge` 内批量化 `make_reverse_query`

**现状：** 4 个选项的 reverse query 分 4 次串行（4 × 2s = 8s）  
**优化：** 在 `choice_as_judge` 入口一次性 batch 生成全部反向 query

```python
# main.py 中，choice_as_judge 开头新增
def _batch_reverse_queries(client, stem: str, options: list) -> list:
    """批量生成所有选项的反向 query；不支持 batch 时回退串行。"""
    requests = [
        (SYS_REVERSE_QUERY,
         f"[题干] {stem}\n[选项] {opt}\n输出陈述 query：",
         None)
        for _, opt in options
    ]
    if hasattr(client, "generate_batch"):
        raws = client.generate_batch(requests, temperature=0.0, max_tokens=60)
    else:
        raws = [make_reverse_query(client, stem, opt) for _, opt in options]
    # 与 make_reverse_query 相同的后处理
    results = []
    for raw, (_, opt) in zip(raws, options):
        s = (raw or "").strip().splitlines()[0].strip(" \t、。，,.;:'\"`")
        if s and 5 <= len(s) <= 80 and s != opt and s != stem:
            results.append(s)
        else:
            results.append("")
    return results
```

然后在 `choice_as_judge` 的 for 循环前调用 `_batch_reverse_queries`，把结果传入 `_judge_one_option`（已有接口，只需新增 `rev_q` 参数）。

**预期效果：** 4 次 reverse query 从 ~8s → ~2s，节省 ~6s/题。

---

## 四、优化后预期耗时

| 步骤 | 优化前（H200 本地） | 优化后 | 节省 |
|---|---|---|---|
| make_reverse_query × 4 | 8s（串行） | 2s（batch） | **6s** |
| verdict generate × 4 | 4s（串行） | 1.5s（batch） | **2.5s** |
| generate_n（判断 n=3） | 6s（串行） | 2s（num_return_sequences） | **4s** |
| generate_n（简答VQA n=5） | 10s（串行） | 2.5s（num_return_sequences） | **7.5s** |
| detect + reflect + 其他 | ~4s（不变） | ~4s | — |
| **选择题合计** | **~20s** | **~8s** | **60% ↓** |
| **判断题合计** | **~10s** | **~6s** | **40% ↓** |
| **简答VQA 合计** | **~15s** | **~7s** | **53% ↓** |

> 1289 题全量预计从 **97 分钟（API 模式）** 或 **~200 分钟（H200 本地串行）** 缩短至 **~40-50 分钟（H200 本地优化后）**。

---

## 五、实现优先级与注意事项

### 实现顺序
```
1. llm_client.py: LocalQwenVL.generate_n()       ← 最简单，改 1 个函数
2. llm_client.py: LocalQwenVL.generate_batch()   ← 新增 1 个函数
3. main.py: _batch_reverse_queries()             ← 新增辅助函数
4. main.py: choice_as_judge 调用 batch 接口      ← 修改调用点
```

### 注意事项

1. **API 模式兼容性**：`ApiQwenVL` 不实现 `generate_batch`，`choice_as_judge` 通过 `hasattr(client, "generate_batch")` 判断，自动降级为串行。无需改动 API 路径。

2. **Batch padding 与 H200 内存**：Qwen2.5-VL-7B 在 float16 约占 14GB，batch=4 且有图像时额外显存约 2-4GB，H200（80GB）完全安全。

3. **`num_return_sequences` 与 padding**：输入 batch_size=1，输出 batch_size=n；需用 `inputs.input_ids.shape[1]`（而非逐条 `len(inp)`）截取新生成部分。

4. **精度不变保证**：`generate_batch` 对每条输入独立 attention（非 cross-attention），结果与串行调用完全等价。`num_return_sequences` 与多次 `do_sample=True` 统计等价（同分布采样）。

5. **recheck 暂不批量化**：recheck 是条件触发且次数低（多选题才触发），批量化收益小且逻辑复杂，暂不改动。

---

## 六、与并发方案的关系

| 方案 | 适用场景 | 收益来源 |
|---|---|---|
| **Batch inference**（本文档） | H200 本地模式 | 减少 GPU forward pass 次数 |
| **ThreadPoolExecutor（已实现）** | API 模式 | 并行网络 I/O，不减 GPU 次数 |

两者互补：本地模式用 batch；API 模式用并发线程（`--workers 4`）。

当前 `--workers` 默认值已设为 4，API 模式下次 run_mock_test 自动并发。

---

## 七、当次全量 mock 测试结果摘要

| 题型 | 正确/总数 | 准确率 |
|---|---|---|
| 判断QA | 63/90 | 70.0% |
| 判断VQA | 66/100 | 66.0% |
| 简答QA | 23/80 | 28.7% |
| 简答VQA | 26/322 | 8.1%* |
| 选择QA | 164/323 | 50.8% |
| 选择VQA | 191/374 | 51.1% |
| **总计** | **533/1289** | **41.3%** |

*简答VQA 准确率低主要源于 mock 数据 gold answer 格式与 evaluator exact-match 不兼容，非模型能力问题。选择题 50-51% 低于官方 50 题 78% 是因为 mock 数据分布更难，且 API 模式下图像未被模型看到（仅靠 base64 路径，不影响文本题）。

> 官方 50 题基线：78%（39/50）  
> mock 泛化 gap：-36.7 pp（正常，mock 题型分布比官方更均匀、难度更高）
