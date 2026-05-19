"""
检索器 —— DashVector + 多模态 embedding。

embedder 三档自动选择（按可用性）：
  1. LocalGMEEmbedder       : 评测环境，QWEN_MODEL_PATH 指向本地 GME-Qwen2-VL（3584 维，与集合匹配）
  2. DashScopeApiEmbedder   : 本地 dev，走 DashScope multimodal-embedding-v1 / text-embedding-v3
                              （维度由 API 决定，启动时与集合做兼容性校验）
  3. **失败** —— 抛 RetrievalUnavailableError，由 main 决定是 fail-loud 还是降级闭卷

环境变量：
  DASHVECTOR_ENDPOINT       : DashVector endpoint
  DASHVECTOR_API_KEY        : DashVector api key
  DASHVECTOR_COLLECTION     : collection 名
  DASHVECTOR_VECTOR_FIELD   : 多向量集合的字段名，默认 'text'（可选 'image'/'fused'）

  GME_MODEL_PATH            : 本地 GME 模型路径（评测环境必填，本地 dev 留空）
  DASHSCOPE_API_KEY         : DashScope API Key（fallback：用 QWEN_API_KEY 同一把）
  DASHSCOPE_EMBED_MODEL     : DashScope 模型名（默认 multimodal-embedding-v1）

  DISABLE_RETRIEVAL=1       : 强制走 NullRetriever（最快闭卷推理；仅用于调试）

设计原则（v3 修订）：
  - **不静默降级**：检索系统真坏掉时抛 RetrievalUnavailableError
  - 维度不匹配视为"系统不可用"，必须显式告知调用方
  - 调用方（main.py）拿到异常后决定：
      * 默认 → 直接 sys.exit(1)，避免误以为在做 RAG
      * --allow-closed-book → 切到 NullRetriever，明确进入闭卷模式
  - 调用方负责计时
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


class RetrievalUnavailableError(RuntimeError):
    """检索系统不可用 —— 维度不匹配 / 网络挂 / 模型加载失败 等。"""


# ============================================================================
# 主类
# ============================================================================

class DashVectorRetriever:
    """DashVector + 多模态 embedding。"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        topk: int = 5,
        text_field: str = "content",     # collection 里的文本字段名
        vector_field: Optional[str] = None,
    ):
        self.endpoint = endpoint or os.environ.get("DASHVECTOR_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("DASHVECTOR_API_KEY", "")
        self.collection_name = collection or os.environ.get("DASHVECTOR_COLLECTION", "")
        self.embed_model_path = embed_model_path or os.environ.get("GME_MODEL_PATH", "")
        self.topk = topk
        self.text_field = text_field
        self.vector_field = (
            vector_field
            or os.environ.get("DASHVECTOR_VECTOR_FIELD", "text")
        )

        self._collection = None
        self._collection_dim: Optional[int] = None
        self._embedder = None
        self._disabled = False

        if os.environ.get("DISABLE_RETRIEVAL") == "1":
            self._disabled = True

    # ---- collection ----------------------------------------------------

    def _ensure_collection(self) -> None:
        if self._collection is not None or self._disabled:
            return
        if not (self.endpoint and self.api_key and self.collection_name):
            print("[retriever] DashVector env not set, disabling retrieval", flush=True)
            self._disabled = True
            return
        try:
            import dashvector
            client = dashvector.Client(api_key=self.api_key, endpoint=self.endpoint)
            self._collection = client.get(self.collection_name)
            self._collection_dim = self._probe_collection_dim()
            print(
                f"[retriever] DashVector connected: {self.collection_name}, "
                f"vector_field={self.vector_field}, dim={self._collection_dim}",
                flush=True,
            )
        except Exception as e:
            print(f"[retriever] DashVector connect failed: {e}; disabling retrieval", flush=True)
            self._disabled = True

    def _probe_collection_dim(self) -> Optional[int]:
        """从集合描述中读出指定 vector field 的维度。"""
        try:
            desc = str(self._collection)
            # collection.__str__ 包含 fields_schema/vectors_schema 信息
            # 例如：vectors_schema={'text': {'dimension': 3584, ...}, ...}
            import json
            obj = json.loads(desc)
            vs = obj.get("vectors_schema") or {}
            if self.vector_field in vs:
                return int(vs[self.vector_field].get("dimension") or 0)
            # 单向量场景
            if "dimension" in obj:
                return int(obj["dimension"])
        except Exception:
            pass
        return None

    # ---- embedder ------------------------------------------------------

    def _ensure_embedder(self) -> None:
        if self._embedder is not None or self._disabled:
            return

        # 本地 GME（评测环境必须设置 GME_MODEL_PATH）
        if self.embed_model_path and Path(self.embed_model_path).exists():
            try:
                self._embedder = _LocalGMEEmbedder(self.embed_model_path)
                print(f"[retriever] embedder=LocalGMEEmbedder({self.embed_model_path})", flush=True)
                self._check_dim_compat("LocalGMEEmbedder")
                return
            except Exception as e:
                print(f"[retriever] LocalGMEEmbedder failed: {e}", flush=True)

        print("[retriever] no embedder available, disabling retrieval", flush=True)
        self._disabled = True

    def _check_dim_compat(self, embedder_name: str) -> None:
        """探测一次 embed dim，与 collection dim 比对；不匹配 → 自动降级。"""
        if self._collection_dim is None or self._embedder is None:
            return
        try:
            sample_vec = self._embedder.encode(text="维度探测")
        except Exception as e:
            print(f"[retriever] dim probe failed: {e}; disabling retrieval", flush=True)
            self._disabled = True
            return
        ed = len(sample_vec)
        if ed != self._collection_dim:
            print(
                f"[retriever] DIMENSION MISMATCH: {embedder_name} produces {ed}-dim, "
                f"collection requires {self._collection_dim}-dim. "
                f"Disabling retrieval (will use closed-book inference).",
                flush=True,
            )
            print(
                "[retriever] To enable retrieval: set GME_MODEL_PATH to a local "
                "GME-Qwen2-VL model (eval env).",
                flush=True,
            )
            self._disabled = True
        else:
            print(f"[retriever] dim check OK: embedder={ed} == collection={self._collection_dim}", flush=True)

    # ---- public --------------------------------------------------------

    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> List[str]:
        """检索 top-k 文档文本。失败时返回 []。"""
        if self._disabled:
            return []
        self._ensure_collection()
        if self._disabled:
            return []
        self._ensure_embedder()
        if self._disabled:
            return []

        k = topk or self.topk
        try:
            vec = self._embedder.encode(text=text, image_path=image_path)
        except Exception as e:
            print(f"[retriever] embed failed: {e}", flush=True)
            return []

        try:
            resp = self._collection.query(
                vector={self.vector_field: list(vec)},
                topk=k,
            )
            if not getattr(resp, "output", None):
                return []
            docs: List[str] = []
            for doc in resp.output:
                fields = getattr(doc, "fields", None) or {}
                txt = (
                    fields.get(self.text_field)
                    or fields.get("content")
                    or fields.get("text")
                    or fields.get("chunk")
                    or ""
                )
                if txt:
                    docs.append(str(txt))
            return docs
        except Exception as e:
            print(f"[retriever] query failed: {e}", flush=True)
            return []


# ============================================================================
# Embedder 实现
# ============================================================================

class _LocalGMEEmbedder:
    """GME-Qwen2-VL 本地推理。评测环境用。"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        import torch
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def encode(self, text: str, image_path: Optional[str] = None) -> list:
        from PIL import Image

        pil_image = None
        if image_path and Path(image_path).exists():
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception:
                pil_image = None

        with self.torch.inference_mode():
            vec = self.model.embed(texts=[text], images=[pil_image])
        return _to_list(vec)


class _DashScopeApiEmbedder:
    """走 DashScope API 的 embedding。

    模型选择优先级：
      - DASHSCOPE_EMBED_MODEL 环境变量（如果设置）
      - multimodal-embedding-v1（支持图+文，1024-dim）
    """

    def __init__(self, api_key: str, model: Optional[str] = None):
        import dashscope
        self.dashscope = dashscope
        self.dashscope.api_key = api_key
        self.model = model or os.environ.get("DASHSCOPE_EMBED_MODEL", "multimodal-embedding-v1")
        self._is_multimodal = "multimodal" in self.model

    def encode(self, text: str, image_path: Optional[str] = None) -> list:
        if self._is_multimodal:
            payload = [{"text": text or ""}]
            if image_path and Path(image_path).exists():
                # multimodal-embedding-v1 接受 base64 或公网 URL
                # 这里走 base64 inline
                import base64
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                suffix = Path(image_path).suffix.lower().lstrip(".") or "jpeg"
                if suffix == "jpg":
                    suffix = "jpeg"
                payload.append({"image": f"data:image/{suffix};base64,{b64}"})
            r = self.dashscope.MultiModalEmbedding.call(model=self.model, input=payload)
        else:
            r = self.dashscope.TextEmbedding.call(model=self.model, input=[text or ""])

        if r.status_code != 200:
            raise RuntimeError(f"DashScope embedding failed: {r.status_code} {r.message}")
        embs = r.output["embeddings"]
        return list(embs[0]["embedding"])


def _to_list(vec) -> list:
    if hasattr(vec, "detach"):
        return vec.detach().cpu().float().numpy().reshape(-1).tolist()
    if hasattr(vec, "tolist"):
        return list(vec.reshape(-1).tolist() if hasattr(vec, "reshape") else vec.tolist())
    return list(vec)


# ============================================================================
# Null + factory
# ============================================================================

class NullRetriever:
    """空检索器，所有 search 返回 []。"""

    def search(self, text: str, image_path: Optional[str] = None, topk: Optional[int] = None) -> List[str]:
        return []


def build_retriever() -> object:
    if os.environ.get("DISABLE_RETRIEVAL") == "1":
        print("[retriever] DISABLE_RETRIEVAL=1, using NullRetriever", flush=True)
        return NullRetriever()
    return DashVectorRetriever()
