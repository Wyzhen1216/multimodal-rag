"""
LLM 客户端 —— 本地 transformers + DashScope API 双模式。

优先级：
  - QWEN_MODEL_PATH 存在 → LocalQwenVL（默认，评测/本地均优先）
  - 显式 --llm api 或 prefer="api" → ApiQwenVL（需 QWEN_API_KEY）
  - auto 模式：本地模型优先，无 QWEN_MODEL_PATH 才考虑 API

环境变量：
  QWEN_MODEL_PATH  : 本地模型路径（优先，有此路径默认走本地）
  QWEN_API_KEY     : DashScope LLM API Key（仅 --llm api 时使用）
  QWEN_BASE_URL    : 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  QWEN_MODEL_NAME  : API 模式下模型名，默认 qwen2.5-vl-7b-instruct

注意：DASHSCOPE_API_KEY / DASHVECTOR_API_KEY 是向量库/embedding 专用 key，
      不用于 LLM 推理选择。
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple


_PKG_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_IMG_DIR = _PKG_DIR / "data" / "images"
_DATA_URI_CACHE_DIR = _PKG_DIR / "data" / "images_cache"

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_MODEL = "qwen2.5-vl-7b-instruct"

_DATA_URI_RE = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)


# ============================================================================
# 工具：URL / base64 data URI → 本地路径
# ============================================================================

def _materialize_data_uri(image_url: str, cache_dir: Path = _DATA_URI_CACHE_DIR) -> Optional[Path]:
    """把 data:image/<fmt>;base64,<payload> 解码落盘，返回缓存路径。

    用 payload 的 sha1 命名，幂等：同一图片重复出现不会重复写。
    解码失败返回 None。
    """
    m = _DATA_URI_RE.match(image_url)
    if not m:
        return None
    fmt, payload = m.group(1).lower(), m.group(2)
    if fmt == "jpg":
        fmt = "jpeg"
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception:
        return None
    if not raw:
        return None
    digest = hashlib.sha1(payload.encode("ascii", errors="ignore")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{digest}.{fmt}"
    if not out.exists():
        out.write_bytes(raw)
    return out


def resolve_local_image(image_url: str, image_dir: Path = _DEFAULT_IMG_DIR) -> Optional[Path]:
    if not image_url:
        return None
    if image_url.startswith("data:image/"):
        return _materialize_data_uri(image_url)
    p = Path(image_url)
    if p.is_absolute():
        return p if p.exists() else None
    # 包根相对路径（e.g. data/mock/SelectVQA/img/<sha1>.jpeg）
    pkg_rel = _PKG_DIR / p
    if pkg_rel.exists():
        return pkg_rel
    # 兜底：只取文件名，在 image_dir 里找
    fname = image_url.rstrip("/").split("/")[-1]
    local = image_dir / fname
    return local if local.exists() else None


# ============================================================================
# 基类
# ============================================================================

class BaseQwenVL:
    """统一接口：generate(system, user_text, image_path, temperature) -> raw_text"""

    name = "base"

    def generate(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        raise NotImplementedError

    def generate_batch(
        self,
        requests: List[Tuple[str, str, Optional[Path]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """批量推理：[(system, user_text, image_path), ...]  -> [str, ...]
        默认实现：串行调 generate（子类可覆写为真正的 batch forward）。
        """
        return [self.generate(s, u, p, temperature=temperature, max_tokens=max_tokens)
                for s, u, p in requests]

    def generate_n(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """多次采样（用于 self-consistency）。"""
        return [
            self.generate(system, user_text, image_path, temperature=temperature, max_tokens=max_tokens)
            for _ in range(n)
        ]


# ============================================================================
# API 模式 —— DashScope OpenAI 兼容
# ============================================================================

class ApiQwenVL(BaseQwenVL):
    name = "api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        image_dir: Optional[Path] = None,
        retries: int = 2,
    ):
        self.api_key = (
            api_key
            or os.environ.get("QWEN_API_KEY")
            or ""
        )
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_API_BASE)
        self.model = model or os.environ.get("QWEN_MODEL_NAME", DEFAULT_API_MODEL)
        self.image_dir = Path(image_dir) if image_dir else _DEFAULT_IMG_DIR
        self.retries = retries
        if not self.api_key:
            raise RuntimeError("ApiQwenVL: QWEN_API_KEY / DASHSCOPE_API_KEY is not set")
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _img_payload(self, image_path: Optional[Path]) -> Optional[dict]:
        if image_path is None or not image_path.exists():
            return None
        suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{b64}"}}

    def _build_messages(self, system: str, user_text: str, image_path: Optional[Path]) -> list:
        user_content: list = [{"type": "text", "text": user_text}]
        img = self._img_payload(image_path)
        if img is not None:
            user_content.append(img)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self._ensure()
        messages = self._build_messages(system, user_text, image_path)
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(1.0 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def generate_batch(
        self,
        requests: List[Tuple[str, str, Optional[Path]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """并发发送多个独立请求，保序返回。"""
        if not requests:
            return []
        if len(requests) == 1:
            s, u, p = requests[0]
            return [self.generate(s, u, p, temperature=temperature, max_tokens=max_tokens)]
        results = [""] * len(requests)
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = {
                pool.submit(self.generate, s, u, p, temperature=temperature, max_tokens=max_tokens): i
                for i, (s, u, p) in enumerate(requests)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = ""
        return results

    def generate_n(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """n 个请求并发发送。"""
        if n == 1:
            return [self.generate(system, user_text, image_path, temperature=temperature, max_tokens=max_tokens)]
        return self.generate_batch(
            [(system, user_text, image_path)] * n,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ============================================================================
# 本地模式 —— transformers + Qwen-VL 系列
#
# 设计：基类 LocalQwenVLBase 持有共享推理逻辑（messages / batch / generate_n），
#       子类 LocalQwen2VL / LocalQwen2_5VL 各自只声明自己的 model + processor 类，
#       互不耦合。factory `LocalQwenVL()` 按 config.json 的 model_type 派发。
# ============================================================================

class LocalQwenVLBase(BaseQwenVL):
    """Qwen-VL 系列本地客户端的共享基类。

    子类需在 `_load_classes()` 里返回 (ModelCls, ProcessorCls)。
    """
    name = "local"

    def __init__(self, model_path: Optional[str] = None, image_dir: Optional[Path] = None):
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", "")
        if not self.model_path or not Path(self.model_path).exists():
            raise RuntimeError(
                f"{type(self).__name__}: QWEN_MODEL_PATH invalid: {self.model_path!r}"
            )
        self.image_dir = Path(image_dir) if image_dir else _DEFAULT_IMG_DIR

        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[local-vl] loading {self.model_path} on {self.device} "
              f"via {type(self).__name__} ...", flush=True)

        ModelCls, ProcessorCls = self._load_classes()
        self.processor = ProcessorCls.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = ModelCls.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()
        print(f"[local-vl] ready ({type(self).__name__}).", flush=True)

    # ---- 子类必须实现 ----
    def _load_classes(self):
        """返回 (ModelCls, ProcessorCls)，子类各自 import 自己的类。"""
        raise NotImplementedError

    # ---- 共享推理逻辑 ----
    def _build_messages(self, system: str, user_text: str, image_path: Optional[Path]) -> list:
        user_content: list = []
        if image_path is not None and image_path.exists():
            user_content.append({"type": "image", "image": f"file://{image_path.as_posix()}"})
        user_content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        results = self.generate_batch(
            [(system, user_text, image_path)],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return results[0] if results else ""

    def generate_batch(
        self,
        requests: List[Tuple[str, str, Optional[Path]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> List[str]:
        """真 batch forward：多条 prompt 一次 model.generate，GPU 利用率最高。"""
        import torch
        from qwen_vl_utils import process_vision_info

        if not requests:
            return []

        texts = []
        all_image_inputs = []
        all_video_inputs = []
        for system, user_text, image_path in requests:
            messages = self._build_messages(system, user_text, image_path)
            t = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_in, vid_in = process_vision_info(messages)
            texts.append(t)
            all_image_inputs.extend(img_in or [])
            all_video_inputs.extend(vid_in or [])

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        do_sample = temperature > 0.0
        gen_kwargs: dict = dict(max_new_tokens=max_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        text_out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [t.strip() for t in text_out]

    def generate_n(
        self,
        system: str,
        user_text: str,
        image_path: Optional[Path] = None,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """num_return_sequences=n 一次 forward 返回 n 个采样。"""
        if n == 1:
            return self.generate_batch(
                [(system, user_text, image_path)],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        import torch
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(system, user_text, image_path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=img_in if img_in else None,
            videos=vid_in if vid_in else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.9,
                num_return_sequences=n,
            )
        inp_len = inputs.input_ids.shape[1]
        trimmed = [seq[inp_len:] for seq in generated]
        text_out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [t.strip() for t in text_out]


class LocalQwen2VL(LocalQwenVLBase):
    """Qwen2-VL 7B / GME-Qwen2-VL（model_type=qwen2_vl）。"""

    def _load_classes(self):
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        return Qwen2VLForConditionalGeneration, Qwen2VLProcessor


class LocalQwen2_5VL(LocalQwenVLBase):
    """Qwen2.5-VL 7B（model_type=qwen2_5_vl）。"""

    def _load_classes(self):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )
        return Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def _read_model_type(model_path: str) -> str:
    """读 config.json 的 model_type；失败返回空串。"""
    import json
    try:
        with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
            return (json.load(f).get("model_type") or "").lower()
    except Exception as e:
        print(f"[local-vl] warn: read config.json failed: {e}", flush=True)
        return ""


def LocalQwenVL(
    model_path: Optional[str] = None,
    image_dir: Optional[Path] = None,
) -> BaseQwenVL:
    """Factory：按 config.json 的 model_type 选具体的本地客户端类。

    保留旧名 `LocalQwenVL` 用作工厂函数，兼容 `LocalQwenVL()` 调用方式。
    """
    path = model_path or os.environ.get("QWEN_MODEL_PATH", "")
    if not path or not Path(path).exists():
        raise RuntimeError(f"LocalQwenVL: QWEN_MODEL_PATH invalid: {path!r}")

    mt = _read_model_type(path)
    if mt == "qwen2_5_vl":
        return LocalQwen2_5VL(model_path=path, image_dir=image_dir)
    if mt == "qwen2_vl":
        return LocalQwen2VL(model_path=path, image_dir=image_dir)

    # config.json 读取失败或 model_type 未知时，从路径名启发式推断
    # 防止 Qwen2.5-VL 被误用 Qwen2VLForConditionalGeneration（hidden_size 不匹配会崩溃）
    path_lower = Path(path).name.lower()
    if "2.5" in path_lower or "2_5" in path_lower or "qwen2-5" in path_lower:
        print(f"[local-vl] warn: unknown model_type={mt!r}, "
              f"path heuristic → Qwen2.5-VL", flush=True)
        return LocalQwen2_5VL(model_path=path, image_dir=image_dir)

    print(f"[local-vl] warn: unknown model_type={mt!r}, "
          f"fallback to LocalQwen2VL", flush=True)
    return LocalQwen2VL(model_path=path, image_dir=image_dir)


# ============================================================================
# Factory
# ============================================================================

def build_client(prefer: Optional[str] = None) -> BaseQwenVL:
    """
    选择 LLM 客户端：
      - prefer="local"  强制本地（默认行为）
      - prefer="api"    强制 API（需 QWEN_API_KEY）
      - prefer=None/auto：本地模型优先；无 QWEN_MODEL_PATH 时才走 API
    """
    if prefer == "api":
        return ApiQwenVL()
    if prefer == "local":
        return LocalQwenVL()

    # auto 模式：本地模型优先
    local_path = os.environ.get("QWEN_MODEL_PATH", "")
    if local_path and Path(local_path).exists():
        return LocalQwenVL()

    # 无本地模型 → 尝试 API（仅 QWEN_API_KEY，不用 DashScope/DashVector key）
    if os.environ.get("QWEN_API_KEY"):
        try:
            return ApiQwenVL()
        except Exception as e:
            print(f"[llm] ApiQwenVL init failed: {e}, falling back to local", flush=True)
    return LocalQwenVL()


# 兼容旧引用
QwenVLClient = ApiQwenVL
