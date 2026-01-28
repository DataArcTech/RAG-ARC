"""In-process evidence memory bank for DeepSearch."""
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.deepsearch.utils.compression import focused_truncate_text, truncate_text


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
        return truncate_text(text, max_chars=max_chars)


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
        # Cache key: (evidence_id, max_chars, question_key) -> excerpt text
        self._excerpt_cache: Dict[Tuple[str, int, str], str] = {}

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

    @staticmethod
    def _question_key(question: str | None) -> str:
        normalized = " ".join(str(question or "").split())
        if not normalized:
            return "no_question"
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_provenance_hint(provenance: Any) -> str | None:
        """Return a compact provenance hint string for prompts (best-effort)."""

        if not isinstance(provenance, dict):
            return None

        candidates: list[str] = []

        def _add(label: str, value: Any) -> None:
            token = str(value or "").strip()
            if not token:
                return
            candidates.append(f"{label}={token}")

        # Common patterns seen in chunk metadata and evidence wrappers.
        meta = provenance.get("metadata")
        if isinstance(meta, dict):
            _add("filename", meta.get("filename") or meta.get("source_file_name") or meta.get("source_file") or meta.get("path"))
            _add("page", meta.get("page") or meta.get("page_number"))

            chunk_meta = meta.get("chunk_metadata")
            if isinstance(chunk_meta, dict):
                _add(
                    "filename",
                    chunk_meta.get("filename")
                    or chunk_meta.get("source_file_name")
                    or chunk_meta.get("source_file")
                    or chunk_meta.get("path"),
                )
                _add("page", chunk_meta.get("page") or chunk_meta.get("page_number"))

        raw_chunk = provenance.get("raw_chunk")
        if isinstance(raw_chunk, dict):
            raw_meta = raw_chunk.get("metadata")
            if isinstance(raw_meta, dict):
                _add(
                    "filename",
                    raw_meta.get("filename")
                    or raw_meta.get("source_file_name")
                    or raw_meta.get("source_file")
                    or raw_meta.get("path"),
                )
                _add("page", raw_meta.get("page") or raw_meta.get("page_number"))

        if not candidates:
            return None
        # Keep it compact and deterministic.
        return ", ".join(dict.fromkeys(candidates))

    def excerpt_for_prompt(
        self,
        evidence_id: str,
        *,
        max_chars: int,
        question: str | None = None,
    ) -> str:
        record = self.get(evidence_id)
        if record is None:
            return ""
        content = (record.content or "").strip()
        if max_chars <= 0 or len(content) <= max_chars:
            return content

        qkey = self._question_key(question)
        cache_key = (record.evidence_id, int(max_chars), qkey)
        cached = self._excerpt_cache.get(cache_key)
        if isinstance(cached, str) and cached:
            return cached

        if question and str(question).strip():
            excerpt = focused_truncate_text(
                content,
                max_chars=max_chars,
                question=str(question),
                extra=None,
            )
        else:
            excerpt = truncate_text(content, max_chars=max_chars)
        self._excerpt_cache[cache_key] = excerpt
        return excerpt

    def evidence_pack_for_prompt(
        self,
        evidence_ids: Sequence[str],
        *,
        source_key_map: Mapping[str, str] | None = None,
        question: str | None = None,
        max_items: int | None = None,
        max_chars_per_evidence: int = 900,
        max_summary_chars: int = 240,
        title: str = "Evidence Pack",
    ) -> str:
        """Materialize a bounded, human-readable evidence pack for LLM prompting.

        This is intentionally "document-like" (index + per-evidence sections) so models can:
        - navigate sources via the index,
        - quote/extract only the relevant portions from each evidence section,
        - cite using stable Source keys in <sup>k</sup> format.
        """

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
            if max_items is not None:
                ordered_keys = ordered_keys[: max(0, int(max_items))]
                allowed_ids = {str(source_key_map.get(str(k)) or "").strip() for k in ordered_keys}
                ordered = [eid for eid in ordered if eid in allowed_ids]
        else:
            ordered = []
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
            if max_items is not None:
                ordered = ordered[: max(0, int(max_items))]
            source_key_map = self.source_key_map_for_prompt(ordered)
            ordered_keys = sorted(int(k) for k in source_key_map.keys())

        allowlist = ", ".join(str(k) for k in ordered_keys)
        lines: List[str] = [str(title or "Evidence Pack").strip(), ""]
        if allowlist:
            lines.extend(["Citable Source key allowlist:", allowlist, ""])

        lines.append("Index (Source key -> short summary):")
        for key_num in ordered_keys:
            eid = str(source_key_map.get(str(key_num)) or "").strip()
            if not eid:
                continue
            rec = self._records[eid]
            src = f" | source={rec.source}" if rec.source else ""
            hint = self._extract_provenance_hint(rec.provenance)
            hint_str = f" | {hint}" if hint else ""
            lines.append(
                f"- Source key={key_num} id={rec.evidence_id}{src}{hint_str} :: {rec.summary(max_chars=max_summary_chars)}"
            )
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
            lines.append(self.excerpt_for_prompt(rec.evidence_id, max_chars=max_chars_per_evidence, question=question))

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
            content = self.excerpt_for_prompt(record.evidence_id, max_chars=max_chars, question=question)
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
