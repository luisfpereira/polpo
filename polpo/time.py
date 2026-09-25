"""Utilities for tracking execution time."""

import time
from contextlib import contextmanager
from datetime import datetime, timezone


def utc_now():
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class _Timing:
    """Track the timing of a single interval."""

    def __init__(self):
        self._start_tick = None
        self._end_tick = None
        self.started_at = None
        self.finished_at = None

    def start(self):
        """Start timing the interval.

        Returns
        -------
        self : _Timing
            Started timing.
        """
        if self._start_tick is not None:
            raise RuntimeError("already started")

        self._start_tick = time.perf_counter()
        self.started_at = utc_now()

        return self

    def stop(self):
        """Stop timing the interval.

        Returns
        -------
        self : _Timing
            Stopped timing.
        """
        if self._start_tick is None:
            raise RuntimeError("not started")

        if self._end_tick is not None:
            raise RuntimeError("already stopped")

        self._end_tick = time.perf_counter()
        self.finished_at = utc_now()

        return self

    def duration(self):
        """Return the interval duration.

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        if self._start_tick is None:
            raise RuntimeError("not started")

        if self._end_tick is None:
            raise RuntimeError("not stopped")

        return self._end_tick - self._start_tick

    def as_dict(self):
        """Return timing information as a dictionary.

        Returns
        -------
        data : dict
            Start and finish timestamps and duration.
        """
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration(),
        }


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
        self._run = _Timing()
        return self

    def _get_timing(self, key=None):
        """Return the timing associated with a run or event."""
        if key is None:
            return self._run

        if key not in self.events:
            raise RuntimeError(f"{key} unknown")

        return self.events[key]

    def start(self, key=None):
        """Start timing a run or event.

        Parameters
        ----------
        key : hashable
            Event identifier. If None, start a new run.

        Returns
        -------
        self : Timer
            Timer with the run or event started.
        """
        if key is None:
            self.reset()
            self._run.start()
            return self

        if key in self.events:
            raise RuntimeError(f"{key} already started")

        self.events[key] = _Timing().start()

        return self

    def stop(self, key=None):
        """Stop timing a run or event.

        Parameters
        ----------
        key : hashable
            Event identifier. If None, stop the current run.

        Returns
        -------
        self : Timer
            Timer with the run or event stopped.
        """
        self._get_timing(key).stop()

        return self

    def duration(self, key=None):
        """Return the duration of a completed run or event.

        Parameters
        ----------
        key : hashable
            Event identifier. If None, return the run duration.

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        return self._get_timing(key).duration()

    def as_dict(self, key=None):
        """Return timing information for a run or event.

        Parameters
        ----------
        key : hashable
            Event identifier. If None, return the full run information.

        Returns
        -------
        data : dict
            Timing information.
        """
        if key is None:
            return {
                **self._run.as_dict(),
                "events": {key: self.as_dict(key) for key in self.events},
            }

        return self._get_timing(key).as_dict()

    @contextmanager
    def __call__(self, key=None):
        """Time an event within a context manager.

        Parameters
        ----------
        key : hashable
            Event identifier.
        """
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)
