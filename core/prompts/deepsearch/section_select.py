"""Prompts for LLM-backed tree search + section selection."""

SECTION_TREE_SEARCH_SYSTEM_PROMPT = (
    "You are a tree search planner for long documents. "
    "Use the provided document tree and hints to identify nodes likely to contain the answer. "
    "Return ONLY valid JSON with the required keys."
)

SECTION_TREE_SEARCH_USER_PROMPT = (
    "Question:\n"
    "{question}\n\n"
    "File:\n"
    "{file_id}\n\n"
    "Document tree (JSON; nodes include section_id/title/path/summary/node_types/page_start/page_end):\n"
    "{tree_json}\n\n"
    "Candidate hints (optional; may include section_index/value_node_score):\n"
    "{candidates_json}\n\n"
    "Instructions:\n"
    "- Select section_id values likely to contain the answer.\n"
    "- Prefer concise lists (ordered by priority).\n"
    "- Hints are for routing only; they are NOT evidence.\n"
    "- Only choose section_id values that appear in the tree.\n"
    "- Output JSON with keys: node_list (array), thinking (string).\n"
)

SECTION_SELECT_CONSUMER_SYSTEM_PROMPT = (
    "You are a retrieval consumer for long documents. "
    "Given a batch of tree nodes, decide whether you already have enough routing information to proceed to read.pages. "
    "Return ONLY valid JSON with the required keys."
)

SECTION_SELECT_CONSUMER_USER_PROMPT = (
    "Question:\n"
    "{question}\n\n"
    "Selected primary so far:\n"
    "{primary_ids}\n\n"
    "Selected supplementary so far:\n"
    "{supplementary_ids}\n\n"
    "Current node batch (JSON; use section_id only):\n"
    "{batch_json}\n\n"
    "Instructions:\n"
    "- Choose primary sections that directly answer the question.\n"
    "- Choose supplementary sections that provide required context, definitions, tables, or images.\n"
    "- Use only the metadata provided (title/path/summary/node_types/page range) for routing.\n"
    "- Set enough_info=true only when you can already pick the pages/sections to read via read.pages.\n"
    "- Output JSON with keys: primary_section_ids (array), supplementary_section_ids (array), enough_info (boolean), explanation (string).\n"
)
