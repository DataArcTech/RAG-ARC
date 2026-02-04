"""In-process evidence memory bank for DeepSearch.

Design:
- Treat the EvidencePool/EvidenceBank as "external memory": it stores full evidence content.
- Tools (think/explore/code) should consume metadata-only *cards* rather than full text.
- The report stage materializes full evidence text from this bank .
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: Optional[str]
    content: str
    score: Optional[float] = None
    provenance: Dict[str, Any] | None = None


class EvidenceBank:
    """A tiny in-process evidence store with deterministic selection helpers ."""

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
            self._records[evidence_id] = EvidenceRecord(
                evidence_id=evidence_id,
                source=str(raw.get("source") or "").strip() or None,
                content=str(raw.get("content") or ""),
                score=raw.get("score"),
                provenance=raw.get("provenance") if isinstance(raw.get("provenance"), dict) else None,
            )
            self._order.append(evidence_id)

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        token = str(evidence_id or "").strip()
        if not token:
            return None
        return self._records.get(token)

    def ids(self) -> List[str]:
        return list(self._order)

    @staticmethod
    def _extract_provenance_hint(provenance: Any) -> str | None:
        """Return a compact provenance hint string for prompts (best-effort)."""

        if not isinstance(provenance, dict):
            return None

        def _pick(*values: Any) -> str | None:
            for v in values:
                token = str(v or "").strip()
                if token:
                    return token
            return None

        meta = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
        filename = _pick(meta.get("filename"), meta.get("source_file_name"), meta.get("path"))
        # Support both normalized page_start/page_end and legacy metadata keys.
        page_start = provenance.get("page_start")
        page_end = provenance.get("page_end")
        if page_start is None:
            page_start = meta.get("page_start")
        if page_end is None:
            page_end = meta.get("page_end")
        if page_start is None:
            page_start = meta.get("page_number")
        if page_start is None:
            page_start = meta.get("page")
        if page_end is None:
            page_end = meta.get("page_number_end")
        if page_end is None:
            page_end = meta.get("page_end")
        if filename and page_start is not None:
            page = f"page={page_start}" if page_end in (None, page_start) else f"pages={page_start}-{page_end}"
            return f"filename={filename}, {page}"
        if filename:
            return f"filename={filename}"
        if page_start is not None:
            page = f"page={page_start}" if page_end in (None, page_start) else f"pages={page_start}-{page_end}"
            return page
        return None

    def index_for_prompt(self, *, max_items: int | None = None) -> List[Dict[str, Any]]:
        subset = self._order if max_items is None else self._order[: max(0, int(max_items))]
        payload: List[Dict[str, Any]] = []
        for evidence_id in subset:
            rec = self._records.get(evidence_id)
            if rec is None:
                continue
            hint = self._extract_provenance_hint(rec.provenance)
            payload.append(
                {
                    "chunk_id": rec.evidence_id,
                    "evidence_id": rec.evidence_id,
                    "source": rec.source,
                    "score": rec.score,
                    "provenance_hint": hint,
                }
            )
        return payload

    def evidence_pack_for_prompt(
        self,
        evidence_ids: Sequence[str],
        *,
        source_key_map: Mapping[str, str] | None = None,
        title: str = "Evidence Pack",
    ) -> str:
        """Materialize a human-readable evidence pack ."""

        source_key_map = dict(source_key_map or {})
        ordered: List[str] = []
        ordered_keys: List[int] = []

        if source_key_map:
            for key, ev_id in source_key_map.items():
                try:
                    key_num = int(str(key).strip())
                except Exception:  # noqa: BLE001
                    continue
                token = str(ev_id or "").strip()
                if not token or token not in self._records:
                    continue
                ordered_keys.append(key_num)
            ordered_keys = sorted(set(ordered_keys))
            for key_num in ordered_keys:
                token = str(source_key_map.get(str(key_num)) or "").strip()
                if token and token not in ordered and token in self._records:
                    ordered.append(token)
        else:
            if not evidence_ids:
                raise ValueError("evidence_ids must be a non-empty list")
            seen: set[str] = set()
            for eid in evidence_ids:
                token = str(eid or "").strip()
                if not token or token in seen or token not in self._records:
                    continue
                seen.add(token)
                ordered.append(token)
            source_key_map = self.source_key_map_for_prompt(ordered)
            ordered_keys = sorted(int(k) for k in source_key_map.keys())

        allowlist = ", ".join(str(k) for k in ordered_keys)
        lines: List[str] = [str(title or "Evidence Pack").strip(), ""]
        if allowlist:
            lines.extend(["Citable Source key allowlist:", allowlist, ""])

        lines.append("Index (Source key -> id + provenance hint):")
        for key_num in ordered_keys:
            eid = str(source_key_map.get(str(key_num)) or "").strip()
            if not eid:
                continue
            rec = self._records[eid]
            hint = self._extract_provenance_hint(rec.provenance)
            hint_str = f" ({hint})" if hint else ""
            src = f" source={rec.source}" if rec.source else ""
            lines.append(f"- Source key={key_num} id={rec.evidence_id}{src}{hint_str}")
        lines.append("")

        lines.append("Evidence details (use ONLY these as authoritative sources):")
        for key_num in ordered_keys:
            eid = str(source_key_map.get(str(key_num)) or "").strip()
            if not eid:
                continue
            rec = self._records[eid]
            lines.append("")
            header = f"Source key={key_num} id={rec.evidence_id}"
            if rec.source:
                header = f"{header} source={rec.source}"
            hint = self._extract_provenance_hint(rec.provenance)
            if hint:
                header = f"{header} ({hint})"
            lines.append(header)
            lines.append(str(rec.content or ""))

        return "\n".join(lines).strip()

    @staticmethod
    def source_key_map_for_prompt(
        evidence_ids: Sequence[str],
        *,
        max_items: int | None = None,
    ) -> Dict[str, str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for eid in evidence_ids or []:
            token = str(eid or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        if max_items is not None:
            ordered = ordered[: max(0, int(max_items))]
        return {str(idx): eid for idx, eid in enumerate(ordered, start=1)}

    def select_evidences(self, evidence_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Materialize full evidence dicts for a given id list ."""

        ordered: List[str] = []
        seen: set[str] = set()
        for eid in evidence_ids or []:
            token = str(eid or "").strip()
            if not token or token in seen or token not in self._records:
                continue
            seen.add(token)
            ordered.append(token)

        payload: List[Dict[str, Any]] = []
        for eid in ordered:
            rec = self._records[eid]
            payload.append(
                {
                    "chunk_id": rec.evidence_id,
                    "source": rec.source,
                    "content": rec.content,
                    "score": rec.score,
                    "provenance": rec.provenance,
                }
            )
        return payload

    @staticmethod
    def update_recency(recent: List[str], *, used_ids: Sequence[str], retain_k: int) -> List[str]:
        if retain_k <= 0:
            return []
        base: List[str] = [str(x).strip() for x in (recent or []) if str(x).strip()]
        seen: set[str] = set(base)
        for eid in used_ids or []:
            token = str(eid or "").strip()
            if not token:
                continue
            if token in seen:
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
        seen: set[str] = set()
        ordered: List[str] = []
        for eid in ids:
            if eid in seen:
                continue
            seen.add(eid)
            ordered.append(eid)
        return ordered


__all__ = ["EvidenceBank", "EvidenceRecord"]
