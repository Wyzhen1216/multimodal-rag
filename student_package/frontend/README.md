# 多模态历史 RAG Agent — 演示前端

基于 Flask 的轻量 Web 演示，展示项目的方法论与 50 题官评 80% 的完整推理流程。

## 启动

```bash
# 在 student_package 根目录下（必须，因为前端从 ../data/ 和 ../test/outputs/ 读数据）
cd /root/autodl-fs/multimodal-rang/student_package

# 安装依赖（已装就跳过）
.venv-student_package_solution/bin/pip install flask openai

# 启动（默认 5000 端口，可设 PORT 环境变量改）
.venv-student_package_solution/bin/python -m frontend.app

# 浏览器访问 http://localhost:5000
```

## 功能

| Tab | 功能 |
|-----|------|
| 📊 项目概览 | 50 题官评 80% / 各题型分项 / 系统架构图 |
| 🔬 方法论 | baseline → BM25 → HyDE → Rerank → Router → Prompt v2 全时间线，关键发现卡片 |
| 🎯 交互演示 | 50 题任选，展示题面 + 图片 + 路由 + 检索 log + 预测/正确答案对比 + 耗时 |
| 💬 自由提问 | 输入自定义题目，调用 DashScope qwen2.5-vl-7b-instruct（闭卷推理，不走 RAG） |

## 数据来源

前端**不本地加载模型**，所有 50 题数据来自已落地的预测产物：

| 来源 | 路径 |
|------|------|
| 50 题题目 | `data/public_examples.full.json` |
| v2 预测 | `test/outputs/20260520_081855_enhanced_fusion_AB_v2/predictions.jsonl` |
| 分项准确率 | `test/outputs/20260520_081855_enhanced_fusion_AB_v2/metrics.json` |
| 每题耗时 | `test/outputs/20260520_081855_enhanced_fusion_AB_v2/time_cost.jsonl` |
| 检索 log | `test/outputs/20260520_081855_enhanced_fusion_AB_v2/run.log` |
| 错题列表 | `test/outputs/20260520_081855_enhanced_fusion_AB_v2/errors.jsonl` |
| 题目图片 | `data/images/<sha1>.jpeg` |

## 自由提问模式说明

为了 demo 启动快、不依赖 GPU：

- **演示模式（前端默认）**：调 DashScope OpenAI 兼容 API（qwen2.5-vl-7b-instruct），仅闭卷推理，不调用 BM25/HyDE/IB rerank
- **完整 RAG 路径**：在 student_package 根目录跑 `bash run_eval.sh`，使用本地 GME-Qwen2-VL（embedding）+ Qwen2.5-VL（生成）+ DashVector（向量检索）+ BM25（本地稀疏检索）

题型路由的提示信息（演示模式下也会显示），让用户知道评测时该题型实际会走哪条路径。

## 文件结构

```
frontend/
├── app.py              # Flask 后端，单文件 ~250 行
├── README.md           # 本文件
├── requirements.txt    # 依赖（flask + openai）
├── templates/
│   └── index.html      # 单页应用（顶部 4 tab 切换）
└── static/
    ├── css/styles.css  # 配色：考古/历史风格（牛皮纸+古铜+焦糖）
    └── js/app.js       # AJAX 调 /api/* 拿数据，无外部 JS 依赖
```

## API 端点

| 端点 | 用途 |
|-----|------|
| `GET /` | 主页（单页 SPA） |
| `GET /api/health` | 健康检查（数据加载是否完整） |
| `GET /api/metrics` | 50 题官评 metrics.json |
| `GET /api/timeline` | 优化阶段时间线（baseline → v2） |
| `GET /api/sections` | 6 题型 + 题数 + 路由方法 |
| `GET /api/questions?section=XX` | 题目列表（可过滤题型） |
| `GET /api/question/<idx>` | 单题详情（题面 + 路由 + log + 预测） |
| `POST /api/ask` | 自由提问（演示模式调 DashScope API） |
| `GET /images/<filename>` | 题目图片 |

## 设计原则

1. **零外部依赖**（除 flask + openai）：HTML/CSS/JS 全部本地，断网可用
2. **离线优先**：所有 50 题数据预加载到内存，毫秒级响应
3. **路由透明**：每题展示"评测时走哪条路径"，强化方法论理解
4. **配色主题**：考古/历史风（牛皮纸背景 + 古铜色 + 焦糖突出），呼应选题
