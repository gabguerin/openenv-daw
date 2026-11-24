"""OpenAI-compatible chat client for stepping DAW Studio with a local or HF-hosted model.

No Gradio dependency: pure request/response logic so it can be unit-tested offline.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx
from openenv_daw.server.tools import load_instructions

HF_ROUTER_URL = "https://router.huggingface.co/v1"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_MODEL = "default_model"
DEFAULT_SPACE_MODEL = "gabguerin/openenv-daw-lfm3-1.7b-sft"
DEFAULT_MAX_TOKENS = 4096

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def is_space() -> bool:
    """True when running inside a Hugging Face Space."""
    return bool(os.getenv("SPACE_ID"))


def _env_max_tokens() -> int:
    raw = os.getenv("OPENENV_DAW_MODEL_MAX_TOKENS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_MAX_TOKENS


@dataclass
class ModelConfig:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: float = 120.0


@dataclass
class ModelReply:
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    raw: dict[str, Any] | None = None
    error: str | None = None


def default_config() -> ModelConfig:
    """Default endpoint configuration from ``SPACE_ID`` and ``OPENENV_DAW_MODEL_*`` env vars."""
    max_tokens = _env_max_tokens()
    if is_space():
        return ModelConfig(
            base_url=HF_ROUTER_URL,
            model=os.getenv("OPENENV_DAW_MODEL") or DEFAULT_SPACE_MODEL,
            api_key="",
            max_tokens=max_tokens,
        )
    return ModelConfig(
        base_url=os.getenv("OPENENV_DAW_MODEL_URL") or DEFAULT_LOCAL_URL,
        model=os.getenv("OPENENV_DAW_MODEL") or DEFAULT_LOCAL_MODEL,
        api_key=os.getenv("OPENENV_DAW_MODEL_KEY") or "",
        max_tokens=max_tokens,
    )


def openai_tools(tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert MCP ``ListToolsAction`` tool objects into OpenAI function-calling tool specs."""
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def build_messages(intent: str, step_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build chat messages from the system instructions, task intent, and episode step log."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_instructions()},
        {"role": "user", "content": intent},
    ]
    for i, entry in enumerate(step_log, start=1):
        call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": entry["tool"],
                            "arguments": json.dumps(entry.get("arguments") or {}),
                        },
                    }
                ],
            }
        )
        payload = (
            entry["result"]
            if entry.get("result") is not None
            else {"ok": False, "error": entry.get("error")}
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(payload),
            }
        )
    return messages


def _parse_args(raw_args: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw_args is None or raw_args == "":
        return {}, None
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except ValueError as e:
            return None, f"Invalid tool arguments JSON: {e}"
        if not isinstance(parsed, dict):
            return None, "Tool arguments must be a JSON object."
        return parsed, None
    return None, f"Unsupported tool arguments type: {type(raw_args).__name__}"


def _extract_text_tool_call(text: str) -> tuple[str | None, dict[str, Any], str, str | None]:
    """Fallback parser for servers that emit ``<tool_call>{...}</tool_call>`` or bare JSON."""
    candidates: list[tuple[str, tuple[int, int]]] = []
    m = _TOOL_CALL_TAG.search(text)
    if m:
        candidates.append((m.group(1), m.span()))
    else:
        f = _CODE_FENCE.search(text)
        if f:
            candidates.append((f.group(1), f.span()))
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append((stripped, (0, len(text))))

    for blob, span in candidates:
        try:
            obj = json.loads(blob)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "arguments" in obj:
            args, err = _parse_args(obj["arguments"])
            if err is not None:
                return None, {}, text, err
            assert args is not None
            remaining = (text[: span[0]] + text[span[1] :]).strip()
            return obj["name"], args, remaining, None
    return None, {}, text, None


def _http_error_message(resp: httpx.Response) -> str:
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err)
            elif err:
                detail = str(err)
            elif body.get("message"):
                detail = str(body["message"])
    except ValueError:
        detail = resp.text.strip()[:300]
    suffix = f": {detail}" if detail else ""
    return f"Model request failed (HTTP {resp.status_code}){suffix}"


async def next_call(
    cfg: ModelConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelReply:
    """Request the next tool call from an OpenAI-compatible ``/chat/completions`` endpoint."""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_s, transport=transport) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        return ModelReply(error=f"Model request failed: {e}")

    if resp.status_code >= 400:
        return ModelReply(error=_http_error_message(resp))

    try:
        data = resp.json()
    except ValueError as e:
        return ModelReply(error=f"Invalid JSON from model server: {e}")

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ModelReply(raw=data if isinstance(data, dict) else None, error="Model returned no choices.")

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ModelReply(raw=data, error="Model response missing message object.")

    text = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        fn = first.get("function") if isinstance(first, dict) else None
        if not isinstance(fn, dict):
            return ModelReply(text=text, raw=data, error="Model returned a malformed tool call.")
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            return ModelReply(text=text, raw=data, error="Model returned a tool call without a name.")
        args, err = _parse_args(fn.get("arguments"))
        if err is not None:
            return ModelReply(text=text, raw=data, error=err)
        assert args is not None
        return ModelReply(tool=name, args=args, text=text, raw=data)

    name, args, remaining, err = _extract_text_tool_call(text)
    if err is not None:
        return ModelReply(text=text, raw=data, error=err)
    if name is not None:
        return ModelReply(tool=name, args=args, text=remaining, raw=data)
    if choices[0].get("finish_reason") == "length":
        return ModelReply(
            tool=None,
            args={},
            text=text,
            raw=data,
            error=(
                f"The model hit the max_tokens limit ({cfg.max_tokens}) before finishing "
                "its tool call; raise Max tokens."
            ),
        )
    return ModelReply(tool=None, args={}, text=text, raw=data)
