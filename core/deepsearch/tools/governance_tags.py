"""Stable strategy tags for DeepSearch tool governance.

These tags are included in ToolDescriptor.strategy_tags and are consumed by:
- planner tool catalogs (high-signal, low-token metadata)
- governance docs / audits
"""

EVIDENCE_PRIMARY = "evidence_primary"
EVIDENCE_DERIVED = "evidence_derived"

SCOPE_OWNER = "scope_owner"
SCOPE_FILE = "scope_file"

REQUIRES_CYPHER = "requires_cypher"
REQUIRES_CHAIN_TRAVERSE = "requires_chain_traverse"
REQUIRES_LLM = "requires_llm"

