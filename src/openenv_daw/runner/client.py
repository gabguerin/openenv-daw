"""Provider-agnostic OpenAI-compatible chat policy, configured only through environment
variables: OPENAI_BASE_URL, OPENAI_API_KEY and OPENENV_DAW_MODEL."""

from __future__ import annotations

import os
from typing import Any, Protocol


class Policy(Protocol):
    name: str

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return the next assistant message in OpenAI chat format."""


class OpenAIPolicy:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        from openai import AsyncOpenAI  # optional extra: openenv-daw[runner]

        self.name = model or os.environ.get("OPENENV_DAW_MODEL", "")
        if not self.name:
            raise ValueError("set OPENENV_DAW_MODEL (or pass model=)")
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = AsyncOpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"),
        )

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        resp = await self._client.chat.completions.create(
            model=self.name,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        m = resp.choices[0].message.model_dump(exclude_none=True)
        msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in m["tool_calls"]
            ]
        return msg
