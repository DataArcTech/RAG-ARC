"""Config defaults for DeepSearch report composition (non-LLM markdown + metadata shaping)."""
DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MIN = 2
DEFAULT_EVIDENCE_RANK_ASCII_ANCHOR_MAX = 10
DEFAULT_EVIDENCE_RANK_CJK_ANCHOR_MIN = 2
DEFAULT_EVIDENCE_RANK_CJK_ANCHOR_MAX = 6
DEFAULT_EVIDENCE_RANK_MAX_ANCHORS = 24

DEFAULT_DIAGNOSTICS_MAX_KEYS = 24
DEFAULT_DIAGNOSTICS_MAX_VALUE_CHARS = 240
DEFAULT_DIAGNOSTICS_MAX_SMALL_LIST_ITEMS = 8
DEFAULT_DIAGNOSTICS_LIST_ITEM_PREVIEW_CHARS = 80

DEFAULT_CITATION_TOKEN_MAX_CHARS = 64
DEFAULT_ALIAS_SAMPLE_SIZE = 10
DEFAULT_UNKNOWN_ALIAS_SAMPLE_SIZE = 20

DEFAULT_GRAPH_EVIDENCE_SEED_ENTITIES_MAX = 12
DEFAULT_GRAPH_CHAIN_PREVIEW_MAX = 40

DEFAULT_EVIDENCE_PREVIEW_MAX_ITEMS = 40

DEFAULT_EXTERNAL_EVIDENCE_MIN_ITEMS = 2
DEFAULT_EXTERNAL_EVIDENCE_FRACTION_DIVISOR = 3

DEFAULT_PROVENANCE_CHUNK_ATTACH_MAX = 10

# Evidence diversification (report prompting): when evidence spans multiple files,
# keep at least a small amount of page evidence per top file to avoid "single-file drift"
# in comparison-like questions. This is a prompt-shaping policy (not a retrieval policy).
DEFAULT_REPORT_EVIDENCE_DIVERSIFY_BY_FILE = True
DEFAULT_REPORT_EVIDENCE_MAX_FILES = 3
DEFAULT_REPORT_EVIDENCE_MIN_ITEMS_PER_FILE = 1

TOOLISH_CHUNK_ID_PREFIXES: tuple[str, ...] = ("tool:", "think")
TOOLISH_SOURCE_NAMES: frozenset[str] = frozenset(
    {
        "think",
    }
)

# Authoritative evidence sources (DeepSearch): only these sources are allowed to be citable
# (plus graph inference evidences explicitly tagged with GRAPH_INFERENCE).
# Keep this list minimal to enforce read.pages as the primary evidence path for answers.
PRIMARY_EVIDENCE_SOURCES: tuple[str, ...] = ("read.pages",)
