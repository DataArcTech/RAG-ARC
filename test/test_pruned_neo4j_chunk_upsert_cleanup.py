from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from encapsulation.database.graph_db.pruned_hipporag_neo4j_chunk_upsert_cleanup import (
    run_chunk_replace_cleanup,
)


@dataclass
class _FakeTx:
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, query: str, params: Mapping[str, Any] | None = None) -> None:
        self.calls.append({"query": str(query), "params": dict(params or {})})


def _chunk_keys(payload: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    return [{"chunk_id": str(p.get("chunk_id") or ""), "owner_id": str(p.get("owner_id") or "")} for p in payload]


def test_run_chunk_replace_cleanup_noops_on_empty_payload() -> None:
    tx = _FakeTx()
    run_chunk_replace_cleanup(tx, chunk_keys=[])
    run_chunk_replace_cleanup(tx, chunk_keys=_chunk_keys([{"chunk_id": "", "owner_id": "o"}]))
    run_chunk_replace_cleanup(tx, chunk_keys=_chunk_keys([{"chunk_id": "c", "owner_id": ""}]))
    assert tx.calls == []


def test_run_chunk_replace_cleanup_emits_mentions_and_facts_queries() -> None:
    tx = _FakeTx()
    keys = _chunk_keys([{"chunk_id": "chunk-1", "owner_id": "__GLOBAL__"}])
    run_chunk_replace_cleanup(tx, chunk_keys=keys)

    assert len(tx.calls) == 2
    assert "MENTIONS" in tx.calls[0]["query"]
    assert "RELATES_TO" in tx.calls[1]["query"]
    assert tx.calls[0]["params"]["chunk_keys"] == keys
    assert tx.calls[1]["params"]["chunk_keys"] == keys

