"""IO reference helpers.

We use a lightweight scheme prefix to avoid confusing IO-managed objects with local filesystem paths.
"""
from dataclasses import dataclass


IO_REF_PREFIX = "io://"


@dataclass(frozen=True)
class IORef:
    """Opaque reference to an object stored via IOManager."""

    key: str

    def __str__(self) -> str:
        token = str(self.key or "").lstrip("/")
        return f"{IO_REF_PREFIX}{token}"


def is_io_ref(value: object) -> bool:
    return isinstance(value, str) and value.strip().startswith(IO_REF_PREFIX)


def to_io_ref(key: str) -> str:
    return str(IORef(key=str(key or "").strip().lstrip("/")))


def from_io_ref(value: str) -> str:
    """Return the underlying key for an io:// ref; return input as-is when not a ref."""

    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    if not text.startswith(IO_REF_PREFIX):
        return text
    return text[len(IO_REF_PREFIX) :].lstrip("/")

