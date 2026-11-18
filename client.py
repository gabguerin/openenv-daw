"""WebSocket client for the DAW environment (exposes reward/done, unlike MCPToolClient)."""

from __future__ import annotations

from typing import Any

import requests
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_daw.models import DawObservation, DawState

ENV_NAME = "openenv_daw"  # the ``env_name`` given to ``create_app``


def extract_data(result: Any) -> dict[str, Any] | None:
    """Pull the tool's JSON dict out of a serialized CallToolResult."""
    if not isinstance(result, dict):
        return None
    for key in ("data", "structured_content", "structuredContent"):
        v = result.get(key)
        if isinstance(v, dict):
            return v.get("result", v) if isinstance(v.get("result"), dict) else v
    return None


class DawEnv(EnvClient[CallToolAction, DawObservation, DawState]):
    def _step_payload(self, action: Any) -> dict[str, Any]:
        if isinstance(action, ListToolsAction):
            return {"type": "list_tools"}
        return {"type": "call_tool", "tool_name": action.tool_name, "arguments": action.arguments}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DawObservation]:
        o = payload.get("observation", {})
        meta = payload.get("metadata") or o.get("metadata") or {}
        obs = DawObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=meta,
            task_id=o.get("task_id", ""),
            intent=o.get("intent", ""),
            tool_name=o.get("tool_name", ""),
            data=extract_data(o.get("result")),
            error=o.get("error"),
            tools=[t if isinstance(t, dict) else dict(t) for t in o.get("tools", [])],
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=meta,
        )

    def _parse_state(self, payload: dict[str, Any]) -> DawState:
        return DawState.model_validate(payload)

    # Task API (plain HTTP, served next to the WebSocket session)

    def _task_url(self, path: str) -> str:
        base = self.base_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
        return f"{base.rstrip('/')}/{ENV_NAME}/{path}"

    def list_splits(self) -> list[str]:
        r = requests.get(self._task_url("splits"), timeout=30)
        r.raise_for_status()
        return [s["name"] for s in r.json()]

    def num_tasks(self, split: str) -> int:
        r = requests.post(self._task_url("num_tasks"), json={"split": split}, timeout=30)
        r.raise_for_status()
        return int(r.json()["num_tasks"])

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        body = {"split": split, "start": start, "stop": stop}
        r = requests.post(self._task_url("task_range"), json=body, timeout=60)
        r.raise_for_status()
        return r.json()["tasks"]

    async def list_tools(self) -> list[dict[str, Any]]:
        res = await self.step(ListToolsAction())
        return res.observation.tools

    async def call(self, tool: str, **arguments: Any) -> StepResult[DawObservation]:
        return await self.step(CallToolAction(tool_name=tool, arguments=arguments))
