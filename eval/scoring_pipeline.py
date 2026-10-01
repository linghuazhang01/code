"""Bounded shard finalization with isolated, concurrent Code scoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class ScoringPipeline:
    """Keep GPU work on the caller thread and bound retained shard outputs."""

    def __init__(self, workers: int, max_pending: int) -> None:
        if workers < 1 or max_pending < 1:
            raise ValueError("Scoring workers and pending shards must be positive")
        self.max_pending = max_pending
        self.scorers = ThreadPoolExecutor(max_workers=workers)
        self.finalizers = ThreadPoolExecutor(max_workers=1)
        self.pending: deque[Future[None]] = deque()

    def make_room(self) -> None:
        """Surface failures before generating more; block at the shard limit."""
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if len(self.pending) >= self.max_pending:
            self.pending.popleft().result()

    def submit(self, finalize: Callable[[], None]) -> None:
        self.make_room()
        self.pending.append(self.finalizers.submit(finalize))

    def drain(self) -> None:
        failure: Exception | None = None
        while self.pending:
            try:
                self.pending.popleft().result()
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    def __enter__(self) -> ScoringPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.drain()
        finally:
            self.finalizers.shutdown(wait=True, cancel_futures=True)
            self.scorers.shutdown(wait=True, cancel_futures=True)
