"""In-process evidence memory bank for DeepSearch."""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.deepsearch.utils.compression import focused_truncate_text


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: Optional[str]
    content: str
    score: Optional[float] = None
    provenance: Dict[str, Any] | None = None

    def summary(self, *, max_chars: int = 240) -> str:
        text = (self.content or "").strip().replace("\n", " ")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."


class EvidenceBank:
    """A tiny in-process evidence store with deterministic selection helpers.

    Core design goals:
    - stable lookup by evidence_id (chunk_id)
    - explicit, bounded prompt materialization (index + retrieved snippets)
    - recency retention (retain the last K used evidence ids)
    """

    def __init__(self) -> None:
        self._records: Dict[str, EvidenceRecord] = {}
        self._order: List[str] = []

    def add_many(self, evidences: Iterable[Dict[str, Any]]) -> None:
        for raw in evidences or []:
            if not isinstance(raw, dict):
                continue
            evidence_id = str(raw.get("chunk_id") or raw.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            if evidence_id in self._records:
                continue
            record = EvidenceRecord(
                evidence_id=evidence_id,
                source=str(raw.get("source") or "").strip() or None,
                content=str(raw.get("content") or ""),
                score=raw.get("score"),
                provenance=raw.get("provenance") if isinstance(raw.get("provenance"), dict) else None,
            )
            self._records[evidence_id] = record
            self._order.append(evidence_id)

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        token = str(evidence_id or "").strip()
        if not token:
            return None
        return self._records.get(token)

    def has(self, evidence_id: str) -> bool:
        return self.get(evidence_id) is not None

    def ids(self) -> List[str]:
        return list(self._order)

    def index_for_prompt(self, *, max_items: Optional[int] = None, max_summary_chars: int = 240) -> List[Dict[str, Any]]:
        subset = self._order if max_items is None else self._order[: max(0, int(max_items))]
        payload: List[Dict[str, Any]] = []
        for evidence_id in subset:
            record = self._records.get(evidence_id)
            if record is None:
                continue
            payload.append(
                {
                    # Keep both keys:
                    # - `chunk_id`: the canonical identifier used by inline citations and most prompts.
                    # - `evidence_id`: backward-compatible alias (some call sites use this name).
                    "chunk_id": record.evidence_id,
                    "evidence_id": record.evidence_id,
                    "source": record.source,
                    "score": record.score,
                    "summary": record.summary(max_chars=max_summary_chars),
                }
            )
        return payload

    def select_evidences(
        self,
        evidence_ids: Sequence[str],
        *,
        max_chars: int = 900,
        question: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Materialize a bounded evidence list for prompts.

        - Only include explicitly provided evidence_ids (in given order).
        """

        ordered: List[str] = []
        seen: set[str] = set()

        def _add(eid: str) -> None:
            token = str(eid or "").strip()
            if not token or token in seen:
                return
            if token not in self._records:
                return
            seen.add(token)
            ordered.append(token)

        if not evidence_ids:
            raise ValueError("evidence_ids must be a non-empty list")
        for eid in evidence_ids:
            _add(str(eid))

        payload: List[Dict[str, Any]] = []
        for eid in ordered:
            record = self._records[eid]
            content = (record.content or "").strip()
            if max_chars > 0 and len(content) > max_chars:
                if question and str(question).strip():
                    content = focused_truncate_text(
                        content,
                        max_chars=max_chars,
                        question=str(question),
                        extra=None,
                    )
                else:
                    content = content[: max(0, max_chars - 3)].rstrip() + "..."
            payload.append(
                {
                    "chunk_id": record.evidence_id,
                    "source": record.source,
                    "content": content,
                    "score": record.score,
                }
            )
        return payload

    @staticmethod
    def update_recency(
        recent: List[str],
        *,
        used_ids: Sequence[str],
        retain_k: int,
    ) -> List[str]:
        """Return an updated recency list containing unique ids (most-recent last)."""

        if retain_k <= 0:
            return []
        base: List[str] = [str(x).strip() for x in (recent or []) if str(x).strip()]
        seen: set[str] = set(base)
        for eid in used_ids or []:
            token = str(eid or "").strip()
            if not token:
                continue
            if token in seen:
                # move to the end
                base = [x for x in base if x != token]
            seen.add(token)
            base.append(token)
        if len(base) > retain_k:
            base = base[-retain_k:]
        return base

    @staticmethod
    def normalize_evidence_ids(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            ids = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            ids = [value.strip()] if value.strip() else []
        else:
            ids = [str(value).strip()] if str(value).strip() else []
        # keep order, unique
        seen: set[str] = set()
        ordered: List[str] = []
        for eid in ids:
            if eid in seen:
                continue
            seen.add(eid)
            ordered.append(eid)
        return ordered
