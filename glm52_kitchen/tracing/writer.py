"""Async, bounded trace writer.

The inference/callback thread never performs file I/O.  It only builds a
compact :class:`~glm52_kitchen.tracing.schema.MoeRoutingRecord` and pushes it to a
bounded :class:`queue.Queue`; a dedicated writer thread drains the queue and
appends JSONL, flushing periodically.  Backpressure is configurable so tracing
can preserve inference speed (the documented default ``sample``).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .schema import RunMetadata, MoeRoutingRecord, SUPPORTED_BACKPRESSURE, TraceSchemaError

_SENTINEL = object()
_DEFAULT_QUEUE_SIZE = 8192
_DEFAULT_FLUSH_EVERY = 64
_DEFAULT_FLUSH_INTERVAL_SEC = 0.5


class TraceWriter:
    """Bounded async JSONL writer with configurable backpressure.

    Usage::

        md = RunMetadata(run_id="...", model="...", ...)
        w = TraceWriter("traces/run.jsonl", md, backpressure="sample")
        for rec in records:
            w.push(rec)
        md = w.close()  # adds counters + timing, writes <trace>.meta.json
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        metadata: RunMetadata,
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        backpressure: str = "sample",
        flush_every: int = _DEFAULT_FLUSH_EVERY,
        flush_interval_sec: float = _DEFAULT_FLUSH_INTERVAL_SEC,
    ) -> None:
        if backpressure not in SUPPORTED_BACKPRESSURE:
            raise ValueError(
                f"backpressure must be one of {SUPPORTED_BACKPRESSURE}, got {backpressure!r}"
            )
        if metadata.queue_size and metadata.queue_size != queue_size:
            # Metadata may carry the run-configured queue size; honor it if set,
            # but if caller passed an explicit queue_size keep that consistent.
            queue_size = metadata.queue_size
        self.path = Path(path)
        self.metadata = metadata
        self.backpressure = backpressure
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_sec = max(0.01, float(flush_interval_sec))
        self._q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._lock = threading.Lock()
        self._records_written = 0
        self._records_dropped = 0
        self._records_sampled = 0
        self._sample_keep = 0  # adaptive sample: keep 1 of every (2**_sample_skip)
        self._sample_skip = 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fh = None
        self._start_monotonic = time.monotonic()
        self._start_wall = time.time()

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> "TraceWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # Documented behavior: do not silently overwrite; require append flag explicitly.
            # Callers that want append must construct a fresh run_id and delete/rotate first.
            # We refuse to truncate an existing trace file.
            raise FileExistsError(
                f"trace file already exists: {self.path} (rotate or pass a fresh run_id)"
            )
        self._fh = open(self.path, "w", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._run, name=f"trace-writer-{self.metadata.run_id}", daemon=True
        )
        self._thread.start()
        return self

    def push(self, record: MoeRoutingRecord) -> bool:
        """Push a routing record. Returns True if accepted into the queue.

        Backpressure:
          * ``block``  – block up to a short timeout, then the record is dropped
                         (keeps inference moving while still favoring exactness).
          * ``drop``   – never block; drop immediately when full.
          * ``sample`` – under pressure, keep an adaptively shrinking fraction.
        """
        if self._stop.is_set():
            return False
        if self.backpressure == "sample":
            accepted = self._push_sample(record)
        elif self.backpressure == "drop":
            accepted = self._push_drop(record)
        else:  # block
            accepted = self._push_block(record)
        if accepted:
            # n_expert_used is part of the record already; nothing extra to do.
            pass
        return accepted

    def close(self) -> RunMetadata:
        """Drain remaining records, write metadata sidecar, join writer thread."""
        if self._fh is None:
            return self.metadata
        self._stop.set()
        # Push a sentinel so the writer wakes even if the queue is full.
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
        now_wall = time.time()
        md = self.metadata
        with self._lock:
            md.records_written = self._records_written
            md.records_dropped = self._records_dropped
            md.records_sampled = self._records_sampled
            md.queue_size = self._q.maxsize
            md.backpressure = self.backpressure
            md.started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self._start_wall))
            md.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now_wall))
            md.wall_seconds = round(now_wall - self._start_wall, 3)
        meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(md.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        return md

    # ------------------------------------------------------------------ internals

    def _push_block(self, record: MoeRoutingRecord) -> bool:
        try:
            self._q.put(record, block=True, timeout=0.05)
            return True
        except queue.Full:
            with self._lock:
                self._records_dropped += 1
            return False

    def _push_drop(self, record: MoeRoutingRecord) -> bool:
        try:
            self._q.put_nowait(record)
            return True
        except queue.Full:
            with self._lock:
                self._records_dropped += 1
            return False

    def _push_sample(self, record: MoeRoutingRecord) -> bool:
        fill_ratio = self._q.qsize() / max(1, self._q.maxsize)
        if fill_ratio > 0.9:
            # Pressure: keep 1 of every _sample_skip records to shed load.
            with self._lock:
                self._sample_keep += 1
                if self._sample_keep % self._sample_skip == 0:
                    self._sample_keep = 0
                    self._sample_skip = min(self._sample_skip * 2, 64)
                    self._records_sampled += 1
                    return False
        try:
            self._q.put_nowait(record)
            # When pressure subsides, gently relax sampling.
            if fill_ratio < 0.3 and self._sample_skip > 1:
                with self._lock:
                    self._sample_skip = max(1, self._sample_skip // 2)
            return True
        except queue.Full:
            with self._lock:
                self._records_dropped += 1
            return False

    def _run(self) -> None:
        assert self._fh is not None
        count = 0
        last_flush = time.monotonic()
        while True:
            timeout = self.flush_interval_sec
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                if self._stop.is_set() and self._q.empty():
                    break
                continue
            if item is _SENTINEL:
                # Drain anything still queued after we were asked to stop.
                while True:
                    try:
                        item = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if item is _SENTINEL:
                        continue
                    self._emit(item)
                break
            self._emit(item)
            count += 1
            now = time.monotonic()
            if count % self.flush_every == 0 or (now - last_flush) > self.flush_interval_sec:
                self._fh.flush()
                last_flush = now

    def _emit(self, record: MoeRoutingRecord) -> None:
        assert self._fh is not None
        try:
            line = json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)
            self._fh.write(line)
            self._fh.write("\n")
            with self._lock:
                self._records_written += 1
        except (TraceSchemaError, TypeError, ValueError):
            # Never let a single bad record kill the writer thread.
            with self._lock:
                self._records_dropped += 1


__all__ = ["TraceWriter"]
