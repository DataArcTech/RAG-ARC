"""Benchmark/experiment mode toggle.

This module centralizes the env-driven switch used by benchmarking runs (e.g. GraphRAG-Benchmark)
to reduce product-layer behaviors that can confound algorithm comparisons.

Source of truth:
- `.env` / environment variable `bench_mode=0|1` (lowercase to match existing docs request).

Compatibility aliases (read-only):
- `BENCH_MODE`
- `BENCHMARK_MODE`
"""
import os
from dataclasses import dataclass
from typing import Final, Mapping

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off", ""})


def _coerce_bool(value: str | None) -> bool:
    token = str(value or "").strip().lower()
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    # Fail-safe: unknown values default to disabled.
    return False


def benchmark_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True when benchmark mode is enabled by env."""

    env_map = env or os.environ
    raw = env_map.get("bench_mode")
    if raw is None:
        raw = env_map.get("BENCH_MODE")
    if raw is None:
        raw = env_map.get("BENCHMARK_MODE")
    return _coerce_bool(raw)


@dataclass(frozen=True, slots=True)
class BenchmarkMode:
    enabled: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "BenchmarkMode":
        return cls(enabled=benchmark_mode_enabled(env))

