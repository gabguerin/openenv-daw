"""In-memory limits for the owner-token fallback of "Run agent" on a public Space.

Counters reset when the process restarts; they are a safety net, not accounting."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

SESSION_RUNS_PER_HOUR = 3
SESSION_MAX_TURNS = 20
DEFAULT_DAILY_RUNS = 200


class Quota:
    def __init__(
        self,
        runs_per_hour: int = SESSION_RUNS_PER_HOUR,
        max_turns: int = SESSION_MAX_TURNS,
        daily_runs: int | None = None,
        clock=time.time,
    ) -> None:
        self.runs_per_hour, self.max_turns = runs_per_hour, max_turns
        self.daily_runs = (
            daily_runs
            if daily_runs is not None
            else int(os.getenv("OPENENV_DAW_UI_DAILY_RUNS", str(DEFAULT_DAILY_RUNS)))
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._session: dict[str, deque[float]] = defaultdict(deque)
        self._daily: deque[float] = deque()

    def acquire(self, session: str, oauth: bool) -> str | None:
        """Reserve a run. Returns a refusal message, or None when allowed (OAuth users are
        exempt: they spend their own credits)."""
        if oauth:
            return None
        now = self._clock()
        with self._lock:
            runs = self._session[session]
            while runs and now - runs[0] > 3600:
                runs.popleft()
            while self._daily and now - self._daily[0] > 86400:
                self._daily.popleft()
            if len(runs) >= self.runs_per_hour:
                return f"Limit reached ({self.runs_per_hour} runs per hour without signing in)."
            if len(self._daily) >= self.daily_runs:
                return "The daily limit for anonymous runs is reached; sign in to continue."
            runs.append(now)
            self._daily.append(now)
        return None

    def turns(self, oauth: bool, requested: int) -> int:
        return requested if oauth else min(requested, self.max_turns)


QUOTA = Quota()
