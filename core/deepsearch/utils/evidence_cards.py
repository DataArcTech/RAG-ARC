"""Evidence-card helpers for DeepSearch.

We treat the EvidencePool/EvidenceBank as "external memory" that stores full evidence content.
To keep LLM tool calls stable and avoid context blow-ups, tools receive *cards* (metadata-only)
instead of full page text. Full evidence is only materialized in the report stage (or via read.pages).
"""
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from encapsulation.data_model.deepsearch import EvidenceChunk


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def evidence_card(raw: EvidenceChunk | Mapping[str, Any]) -> Dict[str, Any]:
    """Build a metadata-only evidence card.

    Contract:
    - Must include `chunk_id` so later stages can reference the evidence in the evidence pool.
    - Must NOT include full `content` (external memory stores it).
    """

    if isinstance(raw, EvidenceChunk):
        payload = raw.model_dump(exclude_none=True)
    else:
        payload = dict(raw)

    chunk_id = str(payload.get("chunk_id") or payload.get("evidence_id") or "").strip()
    prov = _safe_dict(payload.get("provenance"))
    meta = _safe_dict(prov.get("metadata"))

    # Keep only stable, low-entropy fields that help the LLM decide next actions.
    out: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "evidence_id": chunk_id,  # alias
        "source": str(payload.get("source") or "").strip() or None,
        "kind": str(payload.get("kind") or "").strip() or None,
        "score": payload.get("score"),
        "provenance": {
            "source_file_id": prov.get("source_file_id") or meta.get("source_file_id"),
            "page_start": prov.get("page_start") if prov.get("page_start") is not None else meta.get("page_start"),
            "page_end": prov.get("page_end") if prov.get("page_end") is not None else meta.get("page_end"),
            "node_type": prov.get("node_type"),
            "node_types": prov.get("node_types"),
            "evidence_class": prov.get("evidence_class"),
            "metadata": {
                # file hints
                "filename": meta.get("filename"),
                # multimodal hints (kept, but do not inline image bytes/content)
                "image_urls": meta.get("image_urls") if isinstance(meta.get("image_urls"), list) else None,
            },
        },
        "content_in_evidence_pool": True,
        "content_len": len(str(payload.get("content") or "")),
    }
    return out


def evidence_cards(evidences: Sequence[EvidenceChunk] | Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [evidence_card(ev) for ev in (evidences or []) if ev is not None]


def limit_cards(cards: Iterable[Dict[str, Any]], *, max_items: int) -> List[Dict[str, Any]]:
    if max_items is None:
        return list(cards)
    limit = max(0, int(max_items))
    if not limit:
        return []
    out: List[Dict[str, Any]] = []
    for card in cards:
        if len(out) >= limit:
            break
        if isinstance(card, dict):
            out.append(card)
    return out


__all__ = ["evidence_card", "evidence_cards", "limit_cards"]

