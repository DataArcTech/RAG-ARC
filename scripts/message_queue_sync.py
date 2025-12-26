"""Backward-compatible import shim for message queue sync helpers.

Some tests/modules import `scripts.message_queue_sync` while the implementation lives under
`scripts.mq_tools.message_queue_sync`.
"""

from scripts.mq_tools.message_queue_sync import _read_stream_batch

__all__ = ["_read_stream_batch"]

