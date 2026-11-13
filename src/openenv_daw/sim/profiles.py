"""Capability profiles: which tools exist for a given target DAW family."""

from __future__ import annotations

from .errors import unsupported

CORE_TOOLS = (
    "get_session",
    "set_transport",
    "create_track",
    "delete_track",
    "rename_track",
    "create_midi_clip",
    "delete_clip",
    "insert_midi_notes",
    "remove_midi_notes",
    "set_mixer_channel",
    "set_send",
    "list_devices",
    "insert_device",
    "get_device_params",
    "set_device_param",
)
MIXER_TOOLS = (
    "get_session",
    "set_transport",
    "set_mixer_channel",
    "set_send",
    "list_devices",
    "get_device_params",
    "set_device_param",
)

PROFILES: dict[str, dict[str, frozenset[str]]] = {
    # Ableton Live 12, REAPER 7, Bitwig 5/6, Ardour 9 can all do these through scripting.
    "portable": {"tools": frozenset(CORE_TOOLS), "capabilities": frozenset()},
    # Logic and Cubase over a control surface; FL Studio's MIDI scripting (mixer, transport and
    # plugin parameters only).
    "mixer_only": {"tools": frozenset(MIXER_TOOLS), "capabilities": frozenset()},
}
# Capability tags beyond tools that a task may require: group_tracks, automation,
# third_party_plugin, render. No profile offers them yet (an audio backend would add "render").
DEFAULT_PROFILE = "portable"
ENV_ONLY_TOOLS = ("submit",)


def tools_for(profile: str) -> frozenset[str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    return PROFILES[profile]["tools"]


def capabilities_for(profile: str) -> frozenset[str]:
    tools_for(profile)
    return PROFILES[profile]["capabilities"]


def covers(profile: str, requires: list[str]) -> bool:
    """Whether every required tool or capability tag exists in ``profile``."""
    have = tools_for(profile) | capabilities_for(profile) | {"submit"}
    return all(r in have for r in requires)


def check_allowed(profile: str, tool: str) -> None:
    if tool not in tools_for(profile):
        raise unsupported(f"{tool} is not available in the {profile!r} profile")
