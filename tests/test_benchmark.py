"""Unit tests for rl/benchmark.py pure helpers (region-of-attraction boundary).

Pure Python — runs on CI. The sweep functions that drive the sim are
orchestration (verified by running `python -m rl.benchmark`), not unit-tested.
"""

from rl.benchmark import max_recoverable


def test_returns_last_surviving_before_first_failure():
    assert max_recoverable([1, 2, 3], [True, True, False]) == 2


def test_all_survive_returns_largest():
    assert max_recoverable([1, 2, 3], [True, True, True]) == 3


def test_none_survive_returns_zero():
    assert max_recoverable([1, 2, 3], [False, False, False]) == 0.0


def test_stops_at_first_failure_ignoring_later_survivors():
    # Region-of-attraction boundary: once it fails, later "survivals" don't count.
    assert max_recoverable([1, 2, 3, 4], [True, False, True, True]) == 1


def test_empty_returns_zero():
    assert max_recoverable([], []) == 0.0
