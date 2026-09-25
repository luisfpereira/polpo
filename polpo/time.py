"""Utilities for tracking execution time."""

import time
from contextlib import contextmanager
from datetime import datetime, timezone


def utc_now():
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Timer:
    """Track execution times for named events within a run."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all run metadata and recorded events.

        Returns
        -------
        self : Timer
            Reset timer.
        """
        self.events = {}
        self._start = None
        self._end = None
        self.started_at = None
        self.finished_at = None
        return self

    def start(self, key=None):
        """Start timing an event.

        Parameters
        ----------
        key : hashable
            Event identifier.
        """
        if key is None:
            self.reset()
            self.started_at = utc_now()
            self._start = time.perf_counter()
            return self

        if key in self.events and "started_at" in self.events[key]:
            raise RuntimeError(f"{key} already started")

        self.events[key] = {
            "start": time.perf_counter(),
            "started_at": utc_now(),
        }

        return self

    def stop(self, key=None):
        """Stop timing an event.

        Parameters
        ----------
        key : hashable
            Event identifier.
        """
        if key is None:
            self.finished_at = utc_now()
            self._end = time.perf_counter()
            return self

        if key not in self.events or "start" not in self.events[key]:
            raise RuntimeError(f"{key} not started")

        event = self.events[key]
        if "end" in event:
            raise RuntimeError(f"{key} already stopped")

        event["end"] = time.perf_counter()
        event["finished_at"] = utc_now()

        return self

    def duration(self, key=None):
        """Return the duration of a completed event.

        Parameters
        ----------
        key : hashable
            Event identifier.

        Returns
        -------
        duration : float
            Elapsed time in seconds.
        """
        if key is None:
            if self._start is None:
                raise RuntimeError("run not started")
            if self._end is None:
                raise RuntimeError("run not stopped")
            return self._end - self._start

        if key not in self.events:
            raise RuntimeError(f"{key} unknown")

        event = self.events[key]
        if "end" not in event:
            raise RuntimeError(f"{key} not stopped")

        return event["end"] - event["start"]

    def as_dict(self, key=None):
        """Return run metadata and event timings.

        Parameters
        ----------
        key : hashable
            Event identifier.

        Returns
        -------
        data : dict
            Run timestamps and timing information for each event.
        """
        if key is None:
            return {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration": self.duration(),
                "events": {key: self.as_dict(key) for key in self.events},
            }

        event = self.events[key]
        return {
            "started_at": event["started_at"],
            "finished_at": event["finished_at"],
            "duration": self.duration(key),
        }

    @contextmanager
    def __call__(self, key):
        """Time an event within a context manager.

        Parameters
        ----------
        key : hashable
            Event identifier.
        """
        # use e.g. ```with timer("simulation"):```
        self._start(key)
        try:
            yield
        finally:
            self.stop(key)
