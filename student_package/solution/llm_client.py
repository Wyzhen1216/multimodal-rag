"""
LLM 客户端 —— 本地 transformers + DashScope API 双模式。

自动选择：
  - 设置 QWEN_API_KEY → ApiQwenVL（开发期方便）
  - 否则 → LocalQwenVL（评测环境，离线本地推理）

环境变量：
  QWEN_API_KEY     : DashScope API Key（dev 用，可选）
  QWEN_BASE_URL    : 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  QWEN_MODEL_NAME  : API 模式下模型名，默认 qwen2.5-vl-7b-instruct
  QWEN_MODEL_PATH  : 本地模型路径，评测环境必填
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
            or os.environ.get("DASHSCOPE_API_KEY")
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
# 本地模式 —— transformers + Qwen2.5-VL
# ============================================================================

class LocalQwenVL(BaseQwenVL):
    name = "local"

    def __init__(self, model_path: Optional[str] = None, image_dir: Optional[Path] = None):
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", "")
        if not self.model_path or not Path(self.model_path).exists():
            raise RuntimeError(
                f"LocalQwenVL: QWEN_MODEL_PATH invalid: {self.model_path!r}"
            )
        self.image_dir = Path(image_dir) if image_dir else _DEFAULT_IMG_DIR

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[local-vl] loading {self.model_path} on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()
        print(f"[local-vl] ready.", flush=True)

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


# ============================================================================
# Factory
# ============================================================================

def build_client(prefer: Optional[str] = None) -> BaseQwenVL:
    """
    自动选择客户端：
      - prefer="api" 强制走 API
      - prefer="local" 强制走本地
      - prefer=None：QWEN_API_KEY 优先 API，否则本地
    """
    if prefer == "api":
        return ApiQwenVL()
    if prefer == "local":
        return LocalQwenVL()

    # 有任意一个 API key → 走 API 做生成；GME 模型只用于 embedding（retriever.py）
    if (
        os.environ.get("QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    ):
        try:
            return ApiQwenVL()
        except Exception as e:
            print(f"[llm] ApiQwenVL init failed: {e}, falling back to local", flush=True)
    return LocalQwenVL()


# 兼容旧引用
QwenVLClient = ApiQwenVL
