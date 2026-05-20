"""
纯正则 + 后缀 + 朝代词典的实体抽取器（零依赖）。

设计依据：基于官方 50 题题干样本归纳，6 类高价值实体：
  - 编号:    M16 / M38:14 / H3:7 / T2 / G5
  - 书名:    《散氏盘》《秦律十八种·仓律》《释"中庸"》
  - 遗址类:  朱家台遗址 / 朱郎埠墓群 / 中堡岛遗址 / 古共城南城墙
  - 考古文化: 龙山文化 / 屈家岭文化 / 仰韶文化
  - 朝代时期: 西周 / 商代晚期 / 春秋早期 / 秦汉时期
  - 引号专名: '传弃之' / "中庸" / "光■凷宇"

输出 list[str]，按优先级排序、去重、长度过滤；为 BM25 短 query 命中专门优化。
"""
from __future__ import annotations

import re
from typing import List


# ============================================================================
# 朝代/时期词典
# ============================================================================

_DYNASTY_BASE = (
    "夏", "商", "周", "西周", "东周", "春秋", "战国",
    "先秦", "秦汉", "汉魏", "魏晋",
    "秦", "汉", "西汉", "东汉", "三国", "魏", "蜀", "吴",
    "晋", "西晋", "东晋", "南北朝", "魏晋南北朝",
    "隋", "唐", "五代", "宋", "北宋", "南宋", "辽", "金", "元",
    "明", "清", "民国",
    "新石器", "旧石器", "新石器时代", "旧石器时代",
)
_DYNASTY_SUFFIX = ("早期", "中期", "晚期", "前期", "后期", "时期", "初期", "末期", "中叶", "代")

# 预编译成长词优先匹配的 pattern：商代晚期 > 商代 > 商
# 注意：base 已排序，正则交替算子在 Python 里是左优先，所以长形式必须在前
_DYNASTY_PATTERN = re.compile(
    "(?:" + "|".join(sorted(set(_DYNASTY_BASE), key=len, reverse=True)) + ")"
    + "(?:" + "|".join(_DYNASTY_SUFFIX) + ")?"
)

# ============================================================================
# 各类实体正则
# ============================================================================

# 考古编号：M16, M38:14, H3:7, T2, G5, Y3 等
_ID_PATTERN = re.compile(r"[MHTGYK]\d{1,4}(?::\d{1,4})?")

# 书名：《xxx》（含中文标点）
_BOOK_PATTERN = re.compile(r"《[^《》]{1,30}》")

# 遗址/墓群类：2-7 个汉字 + 关键后缀
# 用 finditer + 起点剥离，避免粘连介词（在/的/于/和/或/与/到/从）
_SITE_SUFFIX = ("遗址", "墓群", "墓地", "聚落", "城址", "窑址", "石窟", "陵园", "古城", "城墙")
_SITE_PATTERN = re.compile(
    r"[一-龥]{2,8}(?:" + "|".join(_SITE_SUFFIX) + ")"
)
_LEADING_PARTICLES = ("在", "的", "于", "和", "或", "与", "到", "从", "至", "对", "把", "将", "据", "由", "以", "及")

# 考古文化：2-4 个汉字 + "文化"，同样剥离粘连介词
_CULTURE_PATTERN = re.compile(r"[一-龥]{2,4}文化")

# 引号专名：单引号/双引号包裹的 1-10 字短语
# 中文单/双引号 + 英文 ' " 都接
_QUOTE_PATTERN = re.compile(r"[‘“「『\"\'']([^’”」』\"\'']{1,10})[’”」』\"\'']")

# 器物类（青铜器/陶器/玉器命名常带"鼎/盘/盉/簋/卣/觥/钾/铙/匜/甗/罍"等）
_VESSEL_SUFFIX = ("鼎", "盘", "盉", "簋", "卣", "觥", "钾", "铙", "匜", "甗", "罍",
                  "尊", "爵", "斝", "觚", "壶", "盆", "豆", "镜", "剑", "戈",
                  "矛", "戟", "钺", "印", "符")
# 器物名是常见的 2-4 字定语 + 单字后缀
_VESSEL_PATTERN = re.compile(
    r"[一-龥]{1,4}(?:" + "|".join(_VESSEL_SUFFIX) + ")"
)


# ============================================================================
# 实体抽取主函数
# ============================================================================

def extract_entities(text: str, max_entities: int = 6) -> List[str]:
    """从题干抽取高价值实体；优先级从高到低排序、去重、最多返回 max_entities 个。

    优先级（高优先级实体更可能是查询锚点）：
      1. 编号 (M38:14)  — 唯一性极高，BM25 字面命中
      2. 书名 (《散氏盘》) — 唯一性高
      3. 引号专名 ('传弃之') — 唯一性高
      4. 遗址/墓群 (朱家台遗址) — 主题锚点
      5. 考古文化 (龙山文化) — 主题锚点
      6. 朝代时期 (商代晚期) — 区分时段
      7. 器物 (铜钾, 甗) — 兜底类别
    """
    if not text:
        return []

    seen: set = set()
    out: List[str] = []

    def _add_strict(s: str) -> None:
        """引号专名、编号、书名等：内容已经被引号/书名号定界，直接入池不做任何剥离。"""
        s = s.strip().strip("，。、；：「」『』\"'`'\"")
        if not s or len(s) < 2 or len(s) > 15:
            return
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    def _add_pattern(s: str) -> None:
        """site/culture/dynasty/vessel：可能粘连介词前缀；剥离 + 拒绝中段"的"。"""
        s = s.strip().strip("，。、；：「」『』\"'`'\"")
        # 剥离开头粘连字（介词 + 出土残留单字）
        # 不含"中"：中堡岛/中堂这类合法地名
        _STRIPS = set(_LEADING_PARTICLES) | {"土", "出", "为", "制", "用", "其", "该", "此"}
        while s and s[0] in _STRIPS:
            s = s[1:]
        # 中段含"的" → 拒绝（site/culture 不应包含语法连词）
        if "的" in s:
            return
        if not s or len(s) < 2 or len(s) > 15:
            return
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    # 1. 编号
    for m in _ID_PATTERN.findall(text):
        _add_strict(m)

    # 2. 书名：去掉书名号本身后入池
    for m in _BOOK_PATTERN.findall(text):
        inner = m.strip("《》").strip()
        if inner:
            _add_strict(inner)

    # 3. 引号专名
    for m in _QUOTE_PATTERN.findall(text):
        _add_strict(m)

    # 4. 遗址类
    for m in _SITE_PATTERN.findall(text):
        _add_pattern(m)

    # 5. 考古文化
    for m in _CULTURE_PATTERN.findall(text):
        _add_pattern(m)

    # 6. 朝代时期：保留长形式（商代晚期），避免拆成"商代"+"晚期"
    for m in _DYNASTY_PATTERN.finditer(text):
        full = m.group(0)
        if full and full not in ("代", "期"):
            _add_pattern(full)

    # 7. 器物 — 最容易过抓，放最后，且只在前 6 个槽位用完后启用
    if len(out) < max_entities:
        for m in _VESSEL_PATTERN.findall(text):
            if len(m) >= 3:
                _add_pattern(m)
                if len(out) >= max_entities:
                    break

    return out[:max_entities]


def build_entity_query(entities: List[str]) -> str:
    """把抽取出的实体拼成单条 BM25 query。

    空格分隔即可：BM25 词袋模型对空格切分友好；jieba 在 BM25Retriever 内部会再分一遍。
    """
    return " ".join(entities)


# ============================================================================
# 视觉实体抽取（VQA 同模态匹配增强）
# ============================================================================

def extract_visual_entities(
    client, image_path, question: str = "", max_entities: int = 4,
) -> List[str]:
    """让 VLM 从图中抽视觉实体名词（器物/铭文/纹饰），用于 image-type content 同模态匹配。

    严格 prompt 控制：只输出实体名词，禁止描述句/答案。失败/无图 → 返回 []。
    """
    if image_path is None or client is None:
        return []
    try:
        from .prompts import SYS_VISUAL_ENTITY_EXTRACT
    except Exception:
        return []

    user = f"题目: {question}\n请列出图中可作为检索关键词的视觉实体（逗号分隔，≤5 个）："
    try:
        raw = client.generate(
            SYS_VISUAL_ENTITY_EXTRACT, user, image_path,
            temperature=0.0, max_tokens=40,
        )
    except Exception:
        return []
    if not raw:
        return []

    # 取第一行，按中英文逗号/顿号切分
    line = raw.strip().splitlines()[0].strip()
    # 过滤明显的描述句开头（防止 VLM 不守规则）
    bad_starts = ("图中", "图片", "这是", "答案", "答：", "图为", "可以看到")
    if any(line.startswith(s) for s in bad_starts):
        return []

    import re as _re
    parts = _re.split(r"[,，、;；\s]+", line)
    out: List[str] = []
    seen: set = set()
    for p in parts:
        p = p.strip(" \t\n、。，,.；;:'\"`'\"《》【】[]()")
        if not p or len(p) < 2 or len(p) > 8:
            continue
        # 过滤明显的非实体词（动词/形容词常见前缀）
        if p in ("无", "未知", "不确定", "暂无", "无法识别"):
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_entities:
            break
    return out

