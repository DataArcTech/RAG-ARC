from config.benchmark_mode import benchmark_mode_enabled, BenchmarkMode


def test_benchmark_mode_enabled_parses_values():
    assert benchmark_mode_enabled({"bench_mode": "1"}) is True
    assert benchmark_mode_enabled({"bench_mode": "true"}) is True
    assert benchmark_mode_enabled({"bench_mode": "yes"}) is True
    assert benchmark_mode_enabled({"bench_mode": "0"}) is False
    assert benchmark_mode_enabled({"bench_mode": ""}) is False
    assert benchmark_mode_enabled({"bench_mode": "unexpected"}) is False


def test_benchmark_mode_aliases():
    assert benchmark_mode_enabled({"BENCH_MODE": "1"}) is True
    assert benchmark_mode_enabled({"BENCHMARK_MODE": "1"}) is True


def test_benchmark_mode_dataclass():
    mode = BenchmarkMode.from_env({"bench_mode": "1"})
    assert mode.enabled is True

