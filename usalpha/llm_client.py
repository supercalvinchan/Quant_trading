from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from urllib import error, request


@dataclass
class GLMClientConfig:
    model: str = "glm-5"
    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 4096
    thinking_enabled: bool = True
    timeout_sec: int = 90
    retry_times: int = 5


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def glm_chat_completion(
    messages: list[dict[str, str]],
    api_key: str,
    cfg: GLMClientConfig | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("GLM api key is required")

    conf = cfg or GLMClientConfig()
    body: dict[str, Any] = {
        "model": conf.model,
        "messages": messages,
        "temperature": conf.temperature,
        "max_tokens": conf.max_tokens,
    }
    if conf.thinking_enabled:
        body["thinking"] = {"type": "enabled"}

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Exception | None = None
    for attempt in range(conf.retry_times + 1):
        try:
            req = request.Request(conf.endpoint, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=conf.timeout_sec) as resp:
                text = resp.read().decode("utf-8")
            payload = json.loads(text)
            return {
                "request": body,
                "response": payload,
                "content": _extract_content(payload),
            }
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt >= conf.retry_times:
                break
            sleep_sec = 1.5 * (attempt + 1)
            if isinstance(exc, error.HTTPError) and exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_sec = float(retry_after)
                    except ValueError:
                        sleep_sec = max(sleep_sec, 8.0 * (attempt + 1))
                else:
                    sleep_sec = max(sleep_sec, 8.0 * (attempt + 1))
            time.sleep(sleep_sec)

    raise RuntimeError(f"GLM request failed: {last_err}")
