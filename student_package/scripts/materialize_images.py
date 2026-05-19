"""
materialize_images.py
=====================
将 mock JSON 文件中 base64 data URI 格式的 image_url 解码为本地图片，
并将 image_url 字段替换为相对路径。

图片存储位置：每个 JSON 文件所在目录的 img/ 子目录。
命名规则：payload 的 SHA1，与 llm_client.py 缓存逻辑一致。

用法：
    # 处理所有 mock VQA JSON
    python scripts/materialize_images.py data/mock/SelectVQA/history_mixed_qa_SelectVQA.json
    python scripts/materialize_images.py data/mock/

    # 预览，不修改文件
    python scripts/materialize_images.py data/mock/ --dry-run

    # 只处理已含 data URI 的字段（默认跳过已是路径的）
    python scripts/materialize_images.py data/mock/ --overwrite
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

# 以脚本所在目录的父级作为包根（scripts/ → student_package/）
_PKG_DIR = Path(__file__).resolve().parent.parent

_DATA_URI_RE = re.compile(
    r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL
)


def _decode(image_url: str) -> tuple[str, bytes] | None:
    """解码 data URI，返回 (ext, raw_bytes)；失败返回 None。"""
    m = _DATA_URI_RE.match(image_url)
    if not m:
        return None
    fmt = m.group(1).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    try:
        raw = base64.b64decode(m.group(2), validate=False)
    except Exception:
        return None
    return (fmt, raw) if raw else None


def _sha1(payload: str) -> str:
    return hashlib.sha1(payload.encode("ascii", errors="ignore")).hexdigest()


def process_file(json_path: Path, dry_run: bool = False, overwrite: bool = False) -> dict:
    """处理单个 JSON 文件，返回统计 dict。"""
    img_dir = json_path.parent / "img"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"  [skip] 非数组: {json_path.name}", file=sys.stderr)
        return {"skipped": 1, "converted": 0, "already_path": 0, "errors": 0}

    stats = {"converted": 0, "already_path": 0, "errors": 0}
    changed = False

    for rec in data:
        url = rec.get("image_url") or ""
        if not url:
            continue

        # 已是相对/绝对路径（不是 data URI）
        if not url.startswith("data:"):
            # 检查是否是旧格式 img/<sha1>.jpeg（需升级为包根相对路径）
            if url.startswith("img/"):
                abs_path = (json_path.parent / url).resolve()
                try:
                    pkg_rel = str(abs_path.relative_to(_PKG_DIR))
                except ValueError:
                    pkg_rel = str(abs_path)
                if pkg_rel != url:
                    rec["image_url"] = pkg_rel
                    stats["converted"] += 1
                    changed = True
                    continue
            stats["already_path"] += 1
            continue

        result = _decode(url)
        if result is None:
            print(f"  [error] idx={rec.get('index')} decode failed", file=sys.stderr)
            stats["errors"] += 1
            continue

        ext, raw = result
        # 用 payload 的 SHA1 命名（与 llm_client 缓存对齐）
        payload = url.split(",", 1)[1]
        digest = _sha1(payload)
        fname = f"{digest}.{ext}"
        abs_path = img_dir / fname
        # 相对于包根的路径，供 resolve_local_image 查找
        try:
            rel_path = str(abs_path.resolve().relative_to(_PKG_DIR))
        except ValueError:
            rel_path = str(abs_path.resolve())

        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            if not abs_path.exists() or overwrite:
                abs_path.write_bytes(raw)

        rec["image_url"] = rel_path
        stats["converted"] += 1
        changed = True

    if changed and not dry_run:
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return stats


def collect_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        root = Path(p)
        if root.is_file() and root.suffix == ".json":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.json")))
        else:
            print(f"[warn] 路径不存在或不是 .json: {p}", file=sys.stderr)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="base64 data URI → 本地图片 + 替换路径")
    ap.add_argument("paths", nargs="+", help="JSON 文件或包含 JSON 的目录")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在图片文件")
    args = ap.parse_args()

    files = collect_json_files(args.paths)
    if not files:
        print("FATAL: 未找到任何 .json 文件", file=sys.stderr)
        return 2

    total = {"converted": 0, "already_path": 0, "errors": 0}
    for f in files:
        print(f"[process] {f}")
        s = process_file(f, dry_run=args.dry_run, overwrite=args.overwrite)
        for k in total:
            total[k] += s.get(k, 0)
        print(f"  converted={s['converted']}  already_path={s['already_path']}  errors={s['errors']}")

    print()
    print(f"[done] 文件数={len(files)}  "
          f"converted={total['converted']}  "
          f"already_path={total['already_path']}  "
          f"errors={total['errors']}")
    if args.dry_run:
        print("[done] dry-run 模式，未写入任何文件")
    return 0 if total["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
