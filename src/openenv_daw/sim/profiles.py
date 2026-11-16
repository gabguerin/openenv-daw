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
    # The portable tools on a DAW that can also load third-party plugins by name: library plugins
    # (Serum, ...) can be inserted; in every other profile they are unavailable ("blocked").
    "portable_plugins": {
        "tools": frozenset(CORE_TOOLS),
        "capabilities": frozenset({"third_party_plugin"}),
    },
    # Logic and Cubase over a control surface; FL Studio's MIDI scripting (mixer, transport and
    # plugin parameters only).
    "mixer_only": {"tools": frozenset(MIXER_TOOLS), "capabilities": frozenset()},
    # The parameter-gap slice of heldout_v3: the portable tools, but reverb.mix is read-only. Every
    # reverb_send task needs it, so the correct answer there is submit("blocked").
    "portable_gap": {
        "tools": frozenset(CORE_TOOLS),
        "capabilities": frozenset(),
        "unsupported_params": frozenset({"reverb.mix"}),
    },
}
# Capability tags beyond tools that a task may require: group_tracks, automation,
# third_party_plugin, render. Only ``portable_plugins`` offers third_party_plugin; an audio
# backend would add "render".
DEFAULT_PROFILE = "portable"


def register_profile(
    name: str,
    tools: set[str] | frozenset[str],
    capabilities: set[str] | frozenset[str] = frozenset(),
    unsupported_params: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Register a capability profile (for example from a DAW adapter package).

    ``unsupported_params`` holds ``"<device>.<param>"`` entries the target DAW cannot set."""
    unknown = set(tools) - set(CORE_TOOLS)
    if unknown:
        raise ValueError(f"unknown tools {sorted(unknown)}; the toolset is fixed")
    if name in PROFILES:
        raise ValueError(f"profile {name!r} is already registered")
    PROFILES[name] = {
        "tools": frozenset(tools),
        "capabilities": frozenset(capabilities),
        "unsupported_params": frozenset(unsupported_params),
    }


def unsupported_params_for(profile: str) -> frozenset[str]:
    tools_for(profile)
    return PROFILES[profile].get("unsupported_params", frozenset())


def tools_for(profile: str) -> frozenset[str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    return PROFILES[profile]["tools"]


def capabilities_for(profile: str) -> frozenset[str]:
    tools_for(profile)
    return PROFILES[profile]["capabilities"]


def covers(profile: str, requires: list[str], mask: frozenset[str] = frozenset()) -> bool:
    """Whether every required tool, capability tag and ``param:<device>.<param>`` entry is
    available in ``profile`` (``mask`` adds per-episode unsupported parameters)."""
    have = tools_for(profile) | capabilities_for(profile) | {"submit"}
    missing_params = unsupported_params_for(profile) | mask
    return all(
        (r[6:] not in missing_params) if r.startswith("param:") else r in have for r in requires
    )


def check_allowed(profile: str, tool: str) -> None:
    if tool not in tools_for(profile):
        raise unsupported(f"{tool} is not available in the {profile!r} profile")
