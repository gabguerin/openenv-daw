"""Stable canonical ids (``trk_N``, ``clp_N``, ``dev_N``) over a DAW's own object keys."""

from __future__ import annotations

from openenv_daw.server.sim.session import Session

PREFIXES = ("trk", "clp", "dev", "fdv", "fcl")


class IdMap:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {p: 0 for p in PREFIXES}
        self._n2c: dict[tuple[str, str], str] = {}
        self._c2n: dict[str, str] = {}

    @staticmethod
    def prefix(canonical: str) -> str:
        return canonical.split("_", 1)[0]

    def reset(self) -> None:
        self._n2c.clear()
        self._c2n.clear()

    def bind(self, canonical: str, native: str) -> None:
        prefix = self.prefix(canonical)
        self._n2c[(prefix, native)] = canonical
        self._c2n[canonical] = native
        self.counters[prefix] = max(self.counters.get(prefix, 0), int(canonical.rsplit("_", 1)[1]))

    def native(self, canonical: str) -> str:
        try:
            return self._c2n[canonical]
        except KeyError:
            raise KeyError(f"{canonical!r} is not bound to a native object") from None

    def canonical(self, prefix: str, native: str) -> str:
        key = (prefix, native)
        if key not in self._n2c:
            self.counters[prefix] = self.counters.get(prefix, 0) + 1
            self.bind(f"{prefix}_{self.counters[prefix]}", native)
        return self._n2c[key]

    def canonicalize(self, native: Session) -> Session:
        """A copy of a driver-read session with native keys replaced by canonical ids."""
        s = Session.restore(native.snapshot())
        tmap = {t.id: self.canonical("trk", t.id) for t in s.tracks}
        for t in s.tracks:
            t.id = tmap[t.id]
            t.sends = {tmap.get(k, k): v for k, v in t.sends.items()}
            for c in t.clips:
                c.id = self.canonical("clp", c.id)
            for d in t.devices:
                d.id = self.canonical("dev", d.id)
            for fd in t.foreign_devices:
                fd.id = self.canonical("fdv", fd.id)
            for fc in t.foreign_clips:
                fc.id = self.canonical("fcl", fc.id)
        s.counters = {k: v for k, v in self.counters.items() if v or k in ("trk", "clp", "dev")}
        return s
