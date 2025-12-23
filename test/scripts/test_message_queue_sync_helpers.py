from scripts.message_queue_sync import _read_stream_batch


def test_read_stream_batch_block_zero_does_not_pass_block_kwarg():
    calls: list[dict] = []

    class _Client:
        def xread(self, *args, **kwargs):
            calls.append(dict(kwargs))
            return []

    last_id, entries = _read_stream_batch(_Client(), stream="s", last_id="0-0", count=10, block_ms=0)
    assert last_id == "0-0"
    assert entries == []
    assert calls
    assert "block" not in calls[0]

