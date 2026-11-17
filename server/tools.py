"""The single source of tool definitions: a FastMCP server over a Backend.

The same registry serves ``env.step(CallToolAction)`` during training, and could be served as a
standalone MCP server later. Tools return plain JSON dicts (``ok`` plus the affected object, or
``ok: false`` plus a typed error)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from typing import Any, Literal

from fastmcp import FastMCP
from openenv_daw.server.backends import Backend
from openenv_daw.server.sim.session import GRID_DIVISIONS
from pydantic import BaseModel, Field

Result = dict[str, Any]


class NoteIn(BaseModel):
    pitch: int = Field(description="MIDI pitch 0-127 (60 = middle C)")
    start_beat: float = Field(description="Start in beats, relative to the start of the clip")
    duration_beats: float = Field(description="Length in beats")
    velocity: int = Field(default=100, description="Velocity 1-127")


def load_instructions() -> str:
    """The system prompt. Training and deployment both call this function."""
    text = resources.files("openenv_daw.server").joinpath("instructions.md").read_text()
    return text.replace("{GRID}", str(GRID_DIVISIONS)).rstrip() + "\n"


def build_server(
    get_backend: Callable[[], Backend],
    on_submit: Callable[[str, str], Result] | None = None,
) -> FastMCP:
    """Build the FastMCP server. ``get_backend`` is resolved per call so an environment can swap
    its backend on reset. ``on_submit`` registers the env-only ``submit`` tool."""
    mcp = FastMCP("openenv-daw", instructions=load_instructions())

    def call(tool: str, **kw: Any) -> Result:
        return get_backend().call(tool, {k: v for k, v in kw.items()})

    def tool(fn: Callable[..., Result]) -> Callable[..., Result]:
        mcp.tool(fn)
        return fn

    @tool
    def get_session(track: str | None = None, include_notes: bool = False) -> Result:
        """Read the project: tempo, time signature, and tracks with their mixer state, devices
        and clips. Pass a track (id or name) for full detail on one track, including device
        parameters.

        Args:
            track: Track id or name for a detailed view of a single track.
            include_notes: Include the MIDI notes of every clip.
        """
        return call("get_session", track=track, include_notes=include_notes)

    @tool
    def set_transport(
        tempo_bpm: float | None = None, time_signature: list[int] | None = None
    ) -> Result:
        """Set the project tempo and/or time signature.

        Args:
            tempo_bpm: Tempo in beats per minute (20-300).
            time_signature: [numerator, denominator], for example [3, 4].
        """
        return call("set_transport", tempo_bpm=tempo_bpm, time_signature=time_signature)

    @tool
    def create_track(name: str, kind: Literal["midi", "audio", "return"]) -> Result:
        """Create a track. Track names must be unique. Return tracks host shared effects.

        Args:
            name: Unique track name.
            kind: "midi" (holds MIDI clips and an instrument), "audio" or "return".
        """
        return call("create_track", name=name, kind=kind)

    @tool
    def delete_track(track: str) -> Result:
        """Delete a track and everything on it.

        Args:
            track: Track id or name.
        """
        return call("delete_track", track=track)

    @tool
    def rename_track(track: str, name: str) -> Result:
        """Rename a track.

        Args:
            track: Track id or name.
            name: New unique name.
        """
        return call("rename_track", track=track, name=name)

    @tool
    def create_midi_clip(
        track: str, start_beat: float, length_beats: float, name: str | None = None
    ) -> Result:
        """Create an empty MIDI clip on a midi track's timeline. Clips on one track cannot overlap.

        Args:
            track: Track id or name (must be a midi track).
            start_beat: Clip start on the timeline, in beats from 0.
            length_beats: Clip length in beats (one 4/4 bar = 4 beats).
            name: Optional clip name.
        """
        return call(
            "create_midi_clip",
            track=track,
            start_beat=start_beat,
            length_beats=length_beats,
            name=name,
        )

    @tool
    def delete_clip(clip: str) -> Result:
        """Delete a clip and its notes.

        Args:
            clip: Clip id or unique name.
        """
        return call("delete_clip", clip=clip)

    @tool
    def update_midi_clip(
        clip: str,
        name: str | None = None,
        start_beat: float | None = None,
        length_beats: float | None = None,
    ) -> Result:
        """Rename, move and/or resize a MIDI clip. Omitted fields stay unchanged. The clip cannot
        overlap another clip, and cannot become shorter than its last note (remove those notes
        first).

        Args:
            clip: Clip id or unique name.
            name: New clip name.
            start_beat: New start on the timeline, in beats from 0.
            length_beats: New length in beats.
        """
        return call(
            "update_midi_clip",
            clip=clip,
            name=name,
            start_beat=start_beat,
            length_beats=length_beats,
        )

    @tool
    def insert_midi_notes(clip: str, notes: list[NoteIn]) -> Result:
        """Add MIDI notes to a clip. Notes must fit inside the clip, and notes of the same pitch
        cannot overlap. All notes are validated first; if any is invalid nothing is added.

        Args:
            clip: Clip id or unique name.
            notes: Notes to add (at most 512 per call).
        """
        return call("insert_midi_notes", clip=clip, notes=[_dump(n) for n in notes])

    @tool
    def remove_midi_notes(
        clip: str,
        pitch: int | None = None,
        start_beat: float | None = None,
        end_beat: float | None = None,
    ) -> Result:
        """Remove notes from a clip that match all given filters. With no filters, removes all.

        Args:
            clip: Clip id or unique name.
            pitch: Only notes with this MIDI pitch.
            start_beat: Only notes starting at or after this beat (clip-relative).
            end_beat: Only notes starting before this beat (clip-relative).
        """
        return call(
            "remove_midi_notes", clip=clip, pitch=pitch, start_beat=start_beat, end_beat=end_beat
        )

    @tool
    def set_mixer_channel(
        track: str,
        volume_db: float | None = None,
        pan: float | None = None,
        mute: bool | None = None,
        solo: bool | None = None,
    ) -> Result:
        """Set a track's fader, pan, mute and/or solo. Omitted fields stay unchanged.

        Args:
            track: Track id or name.
            volume_db: Fader level in dB (-70 to +6, 0 = unity).
            pan: -1 (left) to 1 (right), 0 = center.
            mute: Mute the track.
            solo: Solo the track (non-exclusive).
        """
        return call(
            "set_mixer_channel", track=track, volume_db=volume_db, pan=pan, mute=mute, solo=solo
        )

    @tool
    def set_send(track: str, return_track: str, level_db: float) -> Result:
        """Set how much of a track is sent to a return track. -70 turns the send off.

        Args:
            track: Source track id or name (not a return track).
            return_track: Destination return track id or name.
            level_db: Send level in dB (-70 to +6).
        """
        return call("set_send", track=track, return_track=return_track, level_db=level_db)

    @tool
    def list_devices(
        role: Literal["instrument", "audio_fx"] | None = None, query: str | None = None
    ) -> Result:
        """List the devices that can be inserted, with their parameter names. Pass a query to
        search named presets, sounds, drum kits and plugins (for example "sub bass").

        Args:
            role: Only instruments or only audio effects.
            query: Words to search for; each hit's "device" is what insert_device takes.
        """
        return call("list_devices", role=role, query=query)

    @tool
    def insert_device(track: str, device: str, position: int | None = None) -> Result:
        """Insert a device on a track. Instruments go on midi tracks (one per track, always
        first in the chain); effects go after the instrument.

        Args:
            track: Track id or name.
            device: Device id from list_devices, for example "simple_synth", "reverb" or a
                preset or plugin id found with list_devices(query=...).
            position: Chain position for effects (default: end of chain).
        """
        return call("insert_device", track=track, device=device, position=position)

    @tool
    def delete_device(track: str, device: str) -> Result:
        """Remove a device from a track's chain. Racks and plugins cannot be removed.

        Args:
            track: Track id or name.
            device: Device instance id (for example dev_1) or its unique device name.
        """
        return call("delete_device", track=track, device=device)

    @tool
    def get_device_params(track: str, device: str, query: str | None = None) -> Result:
        """Read a device's parameters: normalized value (0..1), display value, unit, and the
        min, max and scale that map one to the other (stepped parameters list their steps).

        Args:
            track: Track id or name.
            device: Device instance id (for example dev_1) or its unique device name.
            query: Only parameters whose name contains this text.
        """
        return call("get_device_params", track=track, device=device, query=query)

    @tool
    def set_device_param(
        track: str,
        device: str,
        param: str,
        value: float | None = None,
        display_value: float | str | None = None,
    ) -> Result:
        """Set one device parameter. Give exactly one of display_value (in the parameter's own
        unit, for example 800 for 800 Hz, or "sine" for a stepped parameter) or value (normalized,
        0 to 1).

        Args:
            track: Track id or name.
            device: Device instance id or its unique device name.
            param: Parameter name from get_device_params.
            value: Normalized value, 0 to 1.
            display_value: Value in the parameter's unit within its min..max range; a step name
                for stepped parameters.
        """
        return call(
            "set_device_param",
            track=track,
            device=device,
            param=param,
            value=value,
            display_value=display_value,
        )

    if on_submit is not None:

        @mcp.tool
        def submit(status: Literal["done", "blocked"] = "done", reason: str = "") -> Result:
            """Finish the task. Call with "done" when the request is complete, or "blocked" (with
            a short reason) when it cannot be done with the available tools.

            Args:
                status: "done" or "blocked".
                reason: Short explanation, mainly for "blocked".
            """
            return on_submit(status, reason)

    return mcp


def _dump(n: NoteIn | dict[str, Any]) -> dict[str, Any]:
    return n.model_dump() if isinstance(n, BaseModel) else dict(n)
