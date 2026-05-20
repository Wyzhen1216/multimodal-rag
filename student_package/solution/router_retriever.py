"""按 section 路由到不同 retriever 的总分发器。

用法：
    main.py 在每个 section 分支前调用 router.for_section(section) 拿到合适的 retriever。
    若不调 for_section 直接 .search()，走 default retriever。

设计原则：
  - 不替换现有 retriever 接口；调用方按需切换
  - 失败兜底：未注册 section 自动走 default
"""
from __future__ import annotations

from typing import Dict, List, Optional


class RouterRetriever:
    def __init__(self, default, routes: Dict[str, object]):
        self.default = default
        self.routes = routes

    def for_section(self, section: str):
        return self.routes.get(section, self.default)

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        return self.default.search(text=text, image_path=image_path, topk=topk)
