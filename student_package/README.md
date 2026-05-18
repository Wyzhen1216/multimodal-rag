# 多模态历史 RAG Agent

## 目录

```
├── run_eval.sh                 # 一键自测脚本（与正式评测一致）
├── requirements.txt            # 基础环境依赖（已预装，供参考）
├── configs/env.example         # 环境变量模板
├── docs/
│   ├── task.md                 # 任务说明
│   ├── data_format.md          # 输入输出格式
│   └── scoring.md              # 评分规则与自测方法
├── data/
│   └── public_examples.full.json    # 测试数据（含公开字段+标准答案）
│   └── images/                      # VQA 图片（24 张，已下载）
├── scripts/
│   ├── check_output.py              # 输出格式校验
│   └── evaluate.py                  # 评分脚本
├── src/
│   └── normalization.py             # 答案归一化
└── solution/                         # 你的代码写在这里
    ├── __init__.py               # Python 包声明（必须保留）
    ├── main.py                   # 入口（必须）
    └── requirements.txt          # 依赖声明（必须）
```

## 快速开始

1. 配置环境变量（参考 `configs/env.example`）
2. 在 `solution/` 中实现 Agent，入口为 `solution/main.py`
3. 运行自测：

```bash
source /inspire/qb-ilm2/project/26summer-camp-16/public/Rag_Based_HistAgent/miniconda3/bin/activate
bash run_eval.sh
```

详细说明见 `docs/`。
