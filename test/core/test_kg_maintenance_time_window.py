from core.knowledge_graph.maintenance.time_window import TimeWindow


def test_time_window_overlaps_unknown_when_both_missing() -> None:
    assert TimeWindow.from_strings().overlaps(TimeWindow.from_strings()) is None


def test_time_window_overlaps_false_when_disjoint() -> None:
    a = TimeWindow.from_strings(valid_from="2020-01-01", valid_to="2020-12-31")
    b = TimeWindow.from_strings(valid_from="2021-01-01", valid_to="2021-12-31")
    assert a.overlaps(b) is False


def test_time_window_open_interval_can_overlap() -> None:
    a = TimeWindow.from_strings(valid_from="2020-01-01", valid_to="")
    b = TimeWindow.from_strings(valid_from="2021-01-01", valid_to="2021-12-31")
    assert a.overlaps(b) is True

