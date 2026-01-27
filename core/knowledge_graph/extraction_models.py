"""
Pydantic schemas for graph extraction structured outputs.

We validate LLM responses before converting them into RAG-ARC's internal GraphData.
"""
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator


_SCREAMING_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ExtractedEntity(BaseModel):
    id: int = Field(..., ge=1, description="1-based stable id for referencing this entity in later steps")
    name: str = Field(..., min_length=1, description="Surface form (resolved; no pronouns)")
    entity_type: str = Field(..., min_length=1, description="Entity type token (e.g., PERSON, ORG, PRODUCT)")

    @field_validator("name", "entity_type")
    @classmethod
    def _strip_non_empty(cls, v: str) -> str:
        v = str(v or "").strip()
        if not v:
            raise ValueError("must be non-empty")
        return v


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)

    @field_validator("extracted_entities")
    @classmethod
    def _unique_ids(cls, v: list[ExtractedEntity]) -> list[ExtractedEntity]:
        ids = [e.id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate entity ids")
        return v


class ExtractedEdge(BaseModel):
    relation_type: str = Field(..., description="FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE")
    source_entity_id: int = Field(..., ge=1)
    target_entity_id: int = Field(..., ge=1)
    fact: Optional[str] = Field(
        default=None,
        description="Optional natural language paraphrase of the relationship; used for debugging/provenance.",
    )
    valid_at: Optional[str] = Field(default=None, description="ISO-8601 datetime (UTC) when the fact became true")
    invalid_at: Optional[str] = Field(default=None, description="ISO-8601 datetime (UTC) when the fact stopped being true")

    @field_validator("relation_type")
    @classmethod
    def _relation_type_style(cls, v: str) -> str:
        token = str(v or "").strip().upper()
        if not token:
            raise ValueError("relation_type must be non-empty")
        # Keep this strict to control drift/noise. Aliases are handled downstream via KG schema.
        if not _SCREAMING_SNAKE_RE.match(token):
            raise ValueError("relation_type must be SCREAMING_SNAKE_CASE (A-Z0-9_)")
        return token


class ExtractedEdges(BaseModel):
    edges: list[ExtractedEdge] = Field(default_factory=list)

