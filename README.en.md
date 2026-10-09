# Multimodal History QA RAG Agent

An end-to-end multimodal RAG agent for archaeology / history question answering. Given a question with an optional image, the system routes it to a task-specific retrieval strategy, recalls evidence from a history knowledge base, and generates a normalized answer with Qwen2.5-VL. Six question types are supported (judgement / multiple-choice / short-answer, with text-only and image variants).

**Team project note:** I was responsible for the **retrieval layer and knowledge-base tagging** — question-type routing, BM25 hybrid retrieval, HyDE query expansion, IB relevance reranking, entity extraction and the tagging scheme. Prompt engineering, evaluation scripts and the frontend were shared with teammates.

## Highlights

- **Per-question-type retrieval routing**: judgement questions go through BM25 hybrid retrieval, short-answer questions through HyDE + IB rerank, VQA and choice questions through dense retrieval with image context.
- **Hybrid sparse + dense search**: a local BM25 index over 5472 passages fused with remote DashVector embeddings via Reciprocal Rank Fusion (k=60).
- **HyDE query expansion plus LLM-based binary relevance reranking**, improving context quality rather than recall volume.
- **Domain entity extraction and tagging** over the corpus, used to constrain retrieval scope and resolve cross-passage references.

## Results

- Official 50-question benchmark: **80%** accuracy, up from a 76% baseline.
- 120-question stratified mock set: **51.7%**, up from 48.3%; short-answer QA improved from 30% to 55%.
- Cost trade-off: +5% latency on 50 questions, +22% on 120 questions, driven by extra LLM calls on the short-answer path.

## Layout

```
student_package/solution/     core implementation (entry point: main.py)
student_package/src/          answer normalization
student_package/configs/      environment variable templates
student_package/data/         sample data and images
student_package/docs/         task, scoring and data-format docs
student_package/findings/     technical report, error and performance analysis
student_package/test/         self-test and A/B comparison scripts
student_package/frontend/     demo frontend
```

## Running

```bash
export QWEN_API_KEY=sk-your-own-key
export DASHVECTOR_API_KEY=sk-your-own-key
export DASHVECTOR_ENDPOINT=<your-dashvector-endpoint>

bash run_eval.sh
```

Python dependencies are declared in `student_package/solution/requirements.txt`.

