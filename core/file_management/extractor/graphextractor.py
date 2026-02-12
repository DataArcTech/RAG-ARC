"""GraphExtractor (JSON-only).

This extractor is used by NetworkXGraphIndexer. We intentionally remove TSV/multi-round/cleaning
logic and switch to the same two-stage structured extraction as HippoRAG2Extractor:
1) Entities (JSON; Pydantic-validated)
2) Edges referencing entity IDs (JSON; Pydantic-validated)
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core.file_management.extractor.base import ExtractorBase
from core.knowledge_graph.extraction_models import ExtractedEntities, ExtractedEdges, ExtractedEntity
from core.knowledge_graph.schema import load_schema_from_yaml, normalize_relation_token
from core.prompts.hipporag2_extractor_prompt_json import (
    HIPPORAG2_EDGE_JSON_PROMPT_EN,
    HIPPORAG2_EDGE_JSON_PROMPT_ZH,
    HIPPORAG2_EDGE_JSON_SYSTEM_EN,
    HIPPORAG2_EDGE_JSON_SYSTEM_ZH,
    HIPPORAG2_EDGE_SCHEMA_HINT_EN,
    HIPPORAG2_EDGE_SCHEMA_HINT_ZH,
    HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_EN,
    HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_ZH,
    HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_EN,
    HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_ZH,
    HIPPORAG2_NER_JSON_PROMPT_EN,
    HIPPORAG2_NER_JSON_PROMPT_ZH,
    HIPPORAG2_NER_JSON_SYSTEM_EN,
    HIPPORAG2_NER_JSON_SYSTEM_ZH,
)
from core.utils.structured_output import StructuredOutputError, call_pydantic_json_with_retry
from encapsulation.data_model.schema import Chunk, GraphData

if TYPE_CHECKING:
    from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig

logger = logging.getLogger(__name__)


def _detect_language(text: str, *, chinese_ratio_threshold: float, default_language: str) -> str:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    total_chars = len(re.sub(r"\s", "", text or ""))
    if total_chars == 0:
        return default_language
    return "zh" if (chinese_chars / total_chars) > chinese_ratio_threshold else "en"


def detect_language_from_text(
    text: str,
    *,
    chinese_ratio_threshold: float,
    default_language: str = "zh",
) -> str:
    """Backward-compatible helper used by tests/config validation."""
    return _detect_language(text, chinese_ratio_threshold=chinese_ratio_threshold, default_language=default_language)


class GraphExtractor(ExtractorBase):
    """NetworkX-path graph extractor (JSON-only)."""

    def __init__(self, config: "GraphExtractorConfig"):
        super().__init__(config)
        self._kg_schema = None
        self._kg_domain = None
        self._kg_domain_schema = None
        self._load_kg_schema()

    def _load_kg_schema(self) -> None:
        cfg_path = str(getattr(self.config, "kg_schema_path", "") or "").strip()
        env_path = os.getenv("KG_SCHEMA_PATH", "").strip()
        schema_path = cfg_path or env_path
        if not schema_path:
            return
        try:
            schema = load_schema_from_yaml(schema_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load KG schema for GraphExtractor: %s", exc, exc_info=True)
            return
        domain_override = str(getattr(self.config, "schema_prompt_domain", "") or "").strip()
        domain = domain_override or getattr(schema, "default_domain", None) or "default"
        try:
            domain_schema = schema.for_domain(domain)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resolve KG schema domain '%s' for GraphExtractor: %s", domain, exc, exc_info=True)
            return
        self._kg_schema = schema
        self._kg_domain = str(domain)
        self._kg_domain_schema = domain_schema

    def detect_language(self, text: str) -> str:
        threshold = float(getattr(self.config, "language_detection_chinese_ratio_threshold", 0.1) or 0.1)
        threshold = max(0.0, min(1.0, threshold))
        default_language = str(getattr(self.config, "language_detection_default_language", "zh") or "zh")
        default_language = default_language if default_language in {"zh", "en"} else "zh"
        return _detect_language(text, chinese_ratio_threshold=threshold, default_language=default_language)

    def _edge_reference_time(self) -> str:
        override = str(getattr(self.config, "edge_reference_time_override", "") or "").strip()
        if override:
            return override
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    async def extract(self, chunk: Chunk) -> GraphData:
        if not chunk.content:
            return GraphData()

        entities_model = await self.extract_entities_json(chunk.content)
        edges_model: ExtractedEdges | None = None
        edge_reference_time: str | None = None
        if entities_model.extracted_entities:
            edges_model, edge_reference_time = await self.extract_edges_json(chunk.content, entities_model)
        else:
            logger.info("No entities extracted (json), skipping edge extraction")

        graph = self.build_graph_data_from_structured(
            entities_model,
            edges_model,
            edge_reference_time=edge_reference_time,
        )
        return graph

    async def extract_entities_json(self, text: str) -> ExtractedEntities:
        language = self.detect_language(text)
        type_constraint = ""
        entity_types = getattr(self.config, "entity_types", None)
        if isinstance(entity_types, list) and entity_types:
            joined = ", ".join([str(t).strip() for t in entity_types if str(t).strip()])
            if joined:
                type_constraint = (
                    f"\nOnly extract entities of the following types: {joined}\n"
                    if language != "zh"
                    else f"\n仅抽取以下类型的实体：{joined}\n"
                )
        prompt = (HIPPORAG2_NER_JSON_PROMPT_ZH if language == "zh" else HIPPORAG2_NER_JSON_PROMPT_EN).format(
            system=((HIPPORAG2_NER_JSON_SYSTEM_ZH if language == "zh" else HIPPORAG2_NER_JSON_SYSTEM_EN) + type_constraint).strip(),
            example_input=HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_ZH if language == "zh" else HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_EN,
            example_output=HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_ZH if language == "zh" else HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_EN,
            passage=text,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            return await call_pydantic_json_with_retry(
                llm_connector=self.llm,
                messages=messages,
                model_cls=ExtractedEntities,
            )
        except StructuredOutputError as exc:
            raise ValueError(f"GraphExtractor NER(JSON) structured output invalid: {exc}") from exc

    async def extract_edges_json(self, text: str, entities: ExtractedEntities) -> tuple[ExtractedEdges, str]:
        language = self.detect_language(text)
        entities_payload = [{"id": e.id, "name": e.name, "entity_type": e.entity_type} for e in entities.extracted_entities]
        reference_time = self._edge_reference_time()

        prompt = (HIPPORAG2_EDGE_JSON_PROMPT_ZH if language == "zh" else HIPPORAG2_EDGE_JSON_PROMPT_EN).format(
            system=HIPPORAG2_EDGE_JSON_SYSTEM_ZH if language == "zh" else HIPPORAG2_EDGE_JSON_SYSTEM_EN,
            edge_types=self._schema_hint_for_edges(language=language) or "",
            entities_json=json.dumps(entities_payload, ensure_ascii=False, separators=(",", ":"), default=str),
            reference_time=reference_time,
            passage=text,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            model = await call_pydantic_json_with_retry(
                llm_connector=self.llm,
                messages=messages,
                model_cls=ExtractedEdges,
            )
        except StructuredOutputError as exc:
            raise ValueError(f"GraphExtractor edges(JSON) structured output invalid: {exc}") from exc
        return model, reference_time

    def _schema_hint_for_edges(self, *, language: str) -> str:
        domain_schema = getattr(self, "_kg_domain_schema", None)
        if domain_schema is None:
            return ""
        max_allowed = int(getattr(self.config, "schema_prompt_max_allowed_relations", 80) or 0)
        max_aliases = int(getattr(self.config, "schema_prompt_max_relation_aliases", 120) or 0)

        allowed = sorted({str(item).strip() for item in (domain_schema.allowed_relations or set()) if str(item).strip()})
        if max_allowed > 0:
            allowed = allowed[:max_allowed]
        allowed_block = "\n".join(allowed) if allowed else "(none)"

        alias_map: dict[str, str] = {}
        if max_aliases != 0 and isinstance(domain_schema.relation_aliases, dict):
            items: list[tuple[str, str]] = []
            for raw_key, raw_value in domain_schema.relation_aliases.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                canonical = normalize_relation_token(str(raw_value or ""))
                if domain_schema.allowed_relations and canonical not in domain_schema.allowed_relations:
                    continue
                items.append((key, canonical))
            items.sort(key=lambda pair: pair[0])
            if max_aliases > 0:
                items = items[:max_aliases]
            alias_map = {k: v for k, v in items}
        alias_json = json.dumps(alias_map, ensure_ascii=False, separators=(",", ":"), default=str) if alias_map else "{}"

        template = HIPPORAG2_EDGE_SCHEMA_HINT_ZH if language == "zh" else HIPPORAG2_EDGE_SCHEMA_HINT_EN
        return template.format(allowed_relations=allowed_block, relation_aliases_json=alias_json).strip()

    def build_graph_data_from_structured(
        self, entities: ExtractedEntities, edges: ExtractedEdges | None, *, edge_reference_time: str | None
    ) -> GraphData:
        entity_list: list[dict] = []
        id_to_entity: dict[int, ExtractedEntity] = {}
        for ent in sorted(entities.extracted_entities, key=lambda e: e.id):
            id_to_entity[ent.id] = ent
            entity_list.append(
                {
                    "id": f"e{ent.id}",
                    "entity_name": ent.name,
                    "entity_type": ent.entity_type,
                    "attributes": {},
                }
            )

        domain_schema = getattr(self, "_kg_domain_schema", None)

        relations: list[list] = []
        dropped_invalid = 0
        dropped_self = 0
        dropped_schema = 0
        if edges is not None:
            for edge in edges.edges:
                if edge.source_entity_id == edge.target_entity_id:
                    dropped_self += 1
                    continue
                src = id_to_entity.get(edge.source_entity_id)
                dst = id_to_entity.get(edge.target_entity_id)
                if src is None or dst is None:
                    dropped_invalid += 1
                    continue

                raw_rel = edge.relation_type
                if domain_schema is not None:
                    norm = domain_schema.normalize_predicate_with_meta(raw_rel)
                    if norm.canonical_predicate is None:
                        dropped_schema += 1
                        continue
                    rel_type = norm.canonical_predicate
                else:
                    rel_type = normalize_relation_token(raw_rel)

                rel: list = [src.name, rel_type, dst.name]
                if edge.valid_at is not None or edge.invalid_at is not None or edge.fact is not None:
                    rel.extend([edge.valid_at, edge.invalid_at, edge.fact])
                relations.append(rel)

        graph = GraphData(entities=entity_list, relations=relations, metadata={})
        graph.metadata["graph_output_format"] = "json"
        if edge_reference_time:
            graph.metadata["edge_reference_time"] = edge_reference_time
        if dropped_invalid or dropped_self or dropped_schema:
            graph.metadata["edge_filter_stats"] = {
                "dropped_invalid_entity_ids": dropped_invalid,
                "dropped_self_loops": dropped_self,
                "dropped_schema_rejected": dropped_schema,
            }
        return graph
