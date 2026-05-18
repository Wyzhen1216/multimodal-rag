"""
LLM 客户端: 调用 Qwen2.5-VL-7B-Instruct (DashScope OpenAI 兼容接口)。

环境变量
--------
  QWEN_API_KEY    : 必填, DashScope / OpenAI 兼容服务的 API Key
  QWEN_BASE_URL   : 可选, 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  QWEN_MODEL_NAME : 可选, 默认 qwen2.5-vl-7b-instruct
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen2.5-vl-7b-instruct"

_DEFAULT_IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


class QwenVLClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        image_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 2,
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("QWEN_MODEL_NAME", DEFAULT_MODEL)
        self.image_dir = Path(image_dir) if image_dir else _DEFAULT_IMG_DIR
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not set")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _local_image_payload(self, image_url: str) -> Optional[dict]:
        if not image_url:
            return None
        fname = image_url.rstrip("/").split("/")[-1]
        local = self.image_dir / fname
        if not local.exists():
            return None
        suffix = local.suffix.lower().lstrip(".") or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(local.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{suffix};base64,{b64}"},
        }

    def build_messages(self, system: str, user_text: str, image_url: str = "") -> list:
        content: list = [{"type": "text", "text": user_text}]
        if image_url:
            payload = self._local_image_payload(image_url)
            if payload is None:
                payload = {"type": "image_url", "image_url": {"url": image_url}}
            content.append(payload)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def chat(self, messages: list) -> str:
        self._ensure_client()
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]
