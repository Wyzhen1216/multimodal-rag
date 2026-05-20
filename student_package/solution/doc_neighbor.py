"""
文献多跳/邻接扩展 + metadata 路由工具。

设计依据（schema 探查实测）：
  - DashVector content/type/document/page/seq 全有 INVERT 索引
  - filter="document='X' and seq>=s-N and seq<=s+N" 可取同文邻段
  - filter="type='paragraph'" / "type in ('image','table')" 工作正常
  - content LIKE 不支持子串（INVERT 不是全文索引）

两层能力：
  1. doc_neighbor_expand: 实体 → vector top-1 锚定 (document, seq) → 同文邻段拉取
     适用：选择QA 多跳（[5462] 同文献多墓号）/ 简答QA 同文背景（[2467] 中庸源头）
  2. type filter 路由：
     - 文字 QA → type='paragraph'（屏蔽 image/table 噪声）
     - 简答 VQA 视觉直读 → type IN ('image','table')（优先 VLM 描述）
"""
from __future__ import annotations

from typing import List, Optional, Tuple


# ============================================================================
# Type filter 路由
# ============================================================================

def type_filter_for_section(section: str, visual_direct: bool = False) -> Optional[str]:
    """按题型/题面特征选 type filter。

    visual_direct: 简答VQA 中"识字/数数/方向"等视觉直读题
                   → 偏向 image/table（VLM 描述更可能命中视觉细节）
    """
    if "简答" in section and "VQA" in section:
        if visual_direct:
            return "type in ('image','table')"
        return None  # 知识类 VQA 不做 type 约束
    if "VQA" in section:
        return None  # 选择/判断 VQA 不做 type 约束（图像 + 文字资料都可能有用）
    # 文字题：屏蔽 image/table 噪声
    return "type='paragraph'"


# ============================================================================
# 多跳邻接扩展（核心能力）
# ============================================================================

def _shape_filter(document: str, seq_lo: int, seq_hi: int,
                  type_filter: Optional[str] = None) -> str:
    """构造邻接扩展 filter：同文献内 seq 在 [seq_lo, seq_hi]，可选 type。"""
    # document 字段是 SHA256 编号/字符串，需要单引号；seq 是 INT 直接数字
    parts = [f"document='{document}'", f"seq>={seq_lo}", f"seq<={seq_hi}"]
    if type_filter:
        parts.append(f"({type_filter})")
    return " and ".join(parts)


def doc_neighbor_expand(
    retriever,
    anchor_query: str,
    *,
    image_path: Optional[str] = None,
    neighbor_window: int = 3,
    max_total: int = 6,
    type_filter: Optional[str] = None,
) -> List[str]:
    """anchor_query → vector top-1 取 (document, seq) → seq±window 邻段拉取。

    返回邻段 content list（含锚点本身），失败/无锚点时返回 []。
    要求 retriever 有 search_with_meta；没有则降级 []。
    """
    if not hasattr(retriever, "search_with_meta"):
        return []

    # 1) 取锚点
    try:
        anchors = retriever.search_with_meta(
            text=anchor_query,
            image_path=image_path,
            topk=1,
            filter=type_filter,
        )
    except Exception:
        anchors = []
    if not anchors:
        return []

    anchor = anchors[0]
    document = anchor.get("document")
    seq = anchor.get("seq")
    if document is None or seq is None:
        # 无 metadata 无法做邻接 → 只返回锚点 content
        return [anchor["content"]]

    # 2) 邻段拉取（一次 query 全部取回，DashVector 按相关度排序）
    try:
        seq_i = int(seq)
    except (TypeError, ValueError):
        return [anchor["content"]]

    expand_filter = _shape_filter(
        document=str(document),
        seq_lo=seq_i - neighbor_window,
        seq_hi=seq_i + neighbor_window,
        type_filter=type_filter,
    )
    try:
        nbrs = retriever.search_with_meta(
            text=anchor_query,
            image_path=image_path,
            topk=max_total,
            filter=expand_filter,
        )
    except Exception:
        nbrs = []

    if not nbrs:
        return [anchor["content"]]

    # 3) 按 seq 升序排（保留阅读顺序），去重
    nbrs_sorted = sorted(nbrs, key=lambda d: (d.get("seq") or 0))
    seen: set = set()
    out: List[str] = []
    for d in nbrs_sorted:
        c = d.get("content") or ""
        if c and c not in seen:
            seen.add(c)
            out.append(c)
            if len(out) >= max_total:
                break
    return out


def doc_neighbor_expand_multi(
    retriever,
    anchor_queries: List[str],
    *,
    image_path: Optional[str] = None,
    neighbor_window: int = 2,
    max_per_anchor: int = 4,
    max_total: int = 8,
    type_filter: Optional[str] = None,
) -> List[str]:
    """多个 anchor query（如多个实体）→ 各自取 top-1 锚 → 各邻段拉取 → 合并去重。

    适用场景：题干含多个独立实体（[5462] M16/M28/M29/M32 各自定位 → 同文献定位 → 邻段抓全文）
    """
    if not anchor_queries:
        return []
    seen: set = set()
    out: List[str] = []
    seen_anchors: set = set()  # (document, seq) 去重，避免多个实体命中同一锚点重复 expand

    for q in anchor_queries:
        if not hasattr(retriever, "search_with_meta"):
            return []
        try:
            anchors = retriever.search_with_meta(
                text=q,
                image_path=image_path,
                topk=1,
                filter=type_filter,
            )
        except Exception:
            anchors = []
        if not anchors:
            continue
        a = anchors[0]
        doc_id = a.get("document")
        seq = a.get("seq")
        if doc_id is None or seq is None:
            content = a["content"]
            if content not in seen:
                seen.add(content)
                out.append(content)
                if len(out) >= max_total:
                    return out
            continue
        anchor_key = (str(doc_id), int(seq) if isinstance(seq, (int, float)) else 0)
        if anchor_key in seen_anchors:
            continue
        seen_anchors.add(anchor_key)

        try:
            seq_i = int(seq)
        except (TypeError, ValueError):
            continue
        nbrs = retriever.search_with_meta(
            text=q,
            image_path=image_path,
            topk=max_per_anchor,
            filter=_shape_filter(str(doc_id), seq_i - neighbor_window, seq_i + neighbor_window, type_filter),
        )
        nbrs_sorted = sorted(nbrs, key=lambda d: (d.get("seq") or 0))
        for d in nbrs_sorted:
            c = d.get("content") or ""
            if c and c not in seen:
                seen.add(c)
                out.append(c)
                if len(out) >= max_total:
                    return out
    return out
