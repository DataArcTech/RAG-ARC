import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set

import yaml

from core.utils.text_processing import text_processing


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_relation_token(value: str) -> str:
    """
    Normalize a predicate token into a stable, schema-governed identifier.

    - Strips punctuation via `text_processing` (lowercase + keep alnum/CJK/spaces)
    - Collapses whitespace to underscores
    - Uppercases for canonical storage
    """
    processed = text_processing(value or "")
    collapsed = _WHITESPACE_RE.sub("_", processed).strip("_")
    return collapsed.upper()


@dataclass(frozen=True)
class PredicateNormalizationResult:
    raw_predicate: str
    raw_predicate_normalized: Optional[str]
    canonical_predicate: Optional[str]
    normalization_strategy: str  # empty | alias | keep | collapse | reject
    allowlist_rejected: bool = False


@dataclass(frozen=True)
class DomainSchema:
    allowed_relations: Set[str] = field(default_factory=set)
    relation_aliases: Dict[str, str] = field(default_factory=dict)
    direction_sensitive_relations: Set[str] = field(default_factory=set)
    direction_policy: str = "whitelist"  # whitelist | blacklist
    direction_insensitive_relations: Set[str] = field(default_factory=set)
    entity_aliases: Dict[str, str] = field(default_factory=dict)
    entity_suffixes_to_strip: List[str] = field(default_factory=list)
    unknown_predicate_policy: str = "collapse"  # reject | keep | collapse
    unknown_predicate_fallback: str = "RELATES_TO"

    def normalize_predicate(self, raw_predicate: str) -> Optional[str]:
        return self.normalize_predicate_with_meta(raw_predicate).canonical_predicate

    def normalize_predicate_with_meta(self, raw_predicate: str) -> PredicateNormalizationResult:
        token = text_processing(raw_predicate or "")
        if not token:
            return PredicateNormalizationResult(
                raw_predicate=str(raw_predicate or ""),
                raw_predicate_normalized=None,
                canonical_predicate=None,
                normalization_strategy="empty",
                allowlist_rejected=False,
            )

        alias_hit = self.relation_aliases.get(token)
        if alias_hit:
            canonical = normalize_relation_token(alias_hit)
            strategy = "alias"
        else:
            policy = (self.unknown_predicate_policy or "collapse").strip().lower()
            if policy == "reject":
                return PredicateNormalizationResult(
                    raw_predicate=str(raw_predicate or ""),
                    raw_predicate_normalized=token,
                    canonical_predicate=None,
                    normalization_strategy="reject",
                    allowlist_rejected=False,
                )
            if policy == "keep":
                canonical = normalize_relation_token(token)
                strategy = "keep"
            else:
                canonical = normalize_relation_token(self.unknown_predicate_fallback or "RELATED_TO")
                strategy = "collapse"

        if self.allowed_relations and canonical not in self.allowed_relations:
            return PredicateNormalizationResult(
                raw_predicate=str(raw_predicate or ""),
                raw_predicate_normalized=token,
                canonical_predicate=None,
                normalization_strategy=strategy,
                allowlist_rejected=True,
            )
        return PredicateNormalizationResult(
            raw_predicate=str(raw_predicate or ""),
            raw_predicate_normalized=token,
            canonical_predicate=canonical,
            normalization_strategy=strategy,
            allowlist_rejected=False,
        )

    def is_direction_sensitive(self, predicate: str) -> bool:
        canonical = normalize_relation_token(predicate or "")
        policy = (self.direction_policy or "whitelist").strip().lower()
        if policy == "blacklist":
            return canonical not in self.direction_insensitive_relations
        return canonical in self.direction_sensitive_relations

    def is_direction_sensitive_from_normalization(self, normalization: PredicateNormalizationResult) -> bool:
        """
        Decide direction sensitivity using both schema configuration and normalization outcome.

        Safety rule (P0): when an unknown predicate is collapsed to a fallback (e.g. RELATED_TO),
        preserve direction by default to avoid head/tail swaps caused by incomplete schema configuration.
        """
        if (normalization.normalization_strategy or "").strip().lower() == "collapse":
            return True
        return self.is_direction_sensitive(normalization.canonical_predicate or "")

    def canonicalize_entity_name(self, raw_entity_name: str) -> Optional[str]:
        token = text_processing(raw_entity_name or "")
        if not token:
            return None

        alias = self.entity_aliases.get(token)
        if alias:
            token = text_processing(alias) or token

        if self.entity_suffixes_to_strip:
            token = self._strip_suffixes(token, self.entity_suffixes_to_strip)

        return _WHITESPACE_RE.sub(" ", token).strip() or None

    @staticmethod
    def _strip_suffixes(value: str, suffixes: List[str]) -> str:
        token = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
        if not token:
            return token
        normalized_suffixes: List[str] = []
        for suf in suffixes:
            processed = _WHITESPACE_RE.sub(" ", text_processing(str(suf or ""))).strip()
            if processed:
                normalized_suffixes.append(processed)
        if not normalized_suffixes:
            return token

        for suf in sorted(set(normalized_suffixes), key=len, reverse=True):
            if token == suf:
                return ""
            if token.endswith(" " + suf):
                token = token[: -(len(suf) + 1)].strip()
                continue
            # CJK/company suffixes are often written without spaces (e.g., "股份有限公司").
            if token.endswith(suf):
                token = token[: -len(suf)].strip()
        return token


@dataclass(frozen=True)
class KGSchemaConfig:
    version: str = "v1"
    default_domain: str = "default"
    domains: Dict[str, DomainSchema] = field(default_factory=dict)

    def for_domain(self, domain: Optional[str]) -> DomainSchema:
        key = (domain or "").strip() or self.default_domain
        return self.domains.get(key) or self.domains.get(self.default_domain) or DomainSchema()

    def direction_sensitive_relations_all(self) -> Set[str]:
        merged: Set[str] = set()
        for domain in self.domains.values():
            merged.update(domain.direction_sensitive_relations)
        return merged

    def direction_insensitive_relations_all(self) -> Set[str]:
        merged: Set[str] = set()
        for domain in self.domains.values():
            merged.update(domain.direction_insensitive_relations)
        return merged

    def direction_policy_all(self) -> str:
        """
        Compute an aggregate direction policy for multi-domain deployments.

        - If any domain declares `blacklist`, treat the overall schema as blacklist
          (safer default: most predicates are direction-sensitive).
        - Otherwise default to `whitelist`.
        """
        policies = {(domain.direction_policy or "whitelist").strip().lower() for domain in self.domains.values()}
        return "blacklist" if "blacklist" in policies else "whitelist"


def _coerce_set(values: Any) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, (set, frozenset)):
        return {normalize_relation_token(str(v)) for v in values if str(v).strip()}
    if isinstance(values, list):
        return {normalize_relation_token(str(v)) for v in values if str(v).strip()}
    raise TypeError(f"Expected list/set for relation set, got {type(values).__name__}")


def _coerce_aliases(values: Any) -> Dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"Expected mapping for relation_aliases, got {type(values).__name__}")
    out: Dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = text_processing(str(raw_key or ""))
        if not key:
            continue
        val = str(raw_value or "").strip()
        if not val:
            continue
        out[key] = val
    return out


def _coerce_entity_aliases(values: Any) -> Dict[str, str]:
    return _coerce_aliases(values)


def _coerce_suffixes(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list):
        out: List[str] = []
        for raw in values:
            token = str(raw or "").strip()
            if token:
                out.append(token)
        return out
    raise TypeError(f"Expected list/str for entity_suffixes_to_strip, got {type(values).__name__}")


def schema_from_dict(payload: Mapping[str, Any]) -> KGSchemaConfig:
    data: MutableMapping[str, Any] = dict(payload or {})
    version = str(data.get("version") or "v1").strip() or "v1"
    default_domain = str(data.get("default_domain") or "default").strip() or "default"

    domains_raw = data.get("domains") or {}
    if not isinstance(domains_raw, Mapping):
        raise TypeError("schema.domains must be a mapping of domain -> domain_schema")

    domains: Dict[str, DomainSchema] = {}
    for domain_name, domain_payload in domains_raw.items():
        domain_key = str(domain_name or "").strip()
        if not domain_key:
            continue
        if not isinstance(domain_payload, Mapping):
            raise TypeError(f"domains.{domain_key} must be a mapping")

        allowed_relations = _coerce_set(domain_payload.get("allowed_relations"))
        direction_sensitive = _coerce_set(domain_payload.get("direction_sensitive_relations"))
        direction_policy = str(domain_payload.get("direction_policy") or "whitelist").strip() or "whitelist"
        direction_insensitive = _coerce_set(domain_payload.get("direction_insensitive_relations"))
        aliases = _coerce_aliases(domain_payload.get("relation_aliases"))
        entity_aliases = _coerce_entity_aliases(domain_payload.get("entity_aliases"))
        suffixes = _coerce_suffixes(domain_payload.get("entity_suffixes_to_strip"))
        policy = str(domain_payload.get("unknown_predicate_policy") or "collapse").strip()
        fallback = str(domain_payload.get("unknown_predicate_fallback") or "RELATES_TO").strip()

        domains[domain_key] = DomainSchema(
            allowed_relations=allowed_relations,
            relation_aliases=aliases,
            direction_sensitive_relations=direction_sensitive,
            direction_policy=direction_policy,
            direction_insensitive_relations=direction_insensitive,
            entity_aliases=entity_aliases,
            entity_suffixes_to_strip=suffixes,
            unknown_predicate_policy=policy,
            unknown_predicate_fallback=fallback,
        )

    return KGSchemaConfig(version=version, default_domain=default_domain, domains=domains)


def load_schema_from_yaml(path: str | Path) -> KGSchemaConfig:
    schema_path = Path(path)
    raw = schema_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"KG schema YAML root must be a mapping, got {type(payload).__name__}")
    return schema_from_dict(payload)
