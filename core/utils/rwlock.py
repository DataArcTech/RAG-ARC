import threading
from contextlib import contextmanager
from typing import Iterator


class RWLock:
    """
    A small readers-writer lock.

    - Multiple readers can enter concurrently.
    - Writers get exclusive access.
    - Writer-preferring to avoid writer starvation.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()

