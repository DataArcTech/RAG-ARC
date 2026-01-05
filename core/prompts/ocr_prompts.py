"""Prompt templates for OCR (VLM OCR + DotsOCR)."""

VLM_OCR_MARKDOWN_PROMPT = (
    "You are an OCR engine.\n"
    "Extract ALL visible text content from the image.\n"
    "Output ONLY GitHub Flavored Markdown.\n"
    "\n"
    "General rules:\n"
    "- Do not translate.\n"
    "- Do not add explanations, summaries, or comments.\n"
    "- Preserve reading order (top-to-bottom, left-to-right).\n"
    "- Keep numbers, dates, currencies, units, punctuation, and symbols exactly as shown.\n"
    "- Use Markdown headings/lists/emphasis ONLY when clearly present.\n"
    "\n"
    "Tables:\n"
    "- If the image contains any tabular content, render it as GitHub Flavored Markdown tables using `|` pipes.\n"
    "- Each table must be its own block; do not merge different tables.\n"
    "- Use a header separator row, e.g. `| --- | --- |`.\n"
    "- Preserve row/column order and cell text verbatim.\n"
    "- If merged cells exist, approximate by repeating the cell text in spanned cells; do not invent values.\n"
    "- Do NOT output HTML tables.\n"
    "\n"
    "Output constraints:\n"
    "- Do NOT wrap the output in triple backticks.\n"
)


DOTS_OCR_PROMPTS: dict[str, str] = {
    # prompt_layout_all_en: parse layout + text in JSON format.
    "prompt_layout_all_en": """Please extract layout elements from this PDF image and return them as a JSON array.

Each array item is an object with:
- bbox: [x1, y1, x2, y2]
- category: one of ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
- text: the text content inside the bbox (omit this field ONLY for category 'Picture')

Text formatting rules for the "text" field:
- Picture: omit "text".
- Formula: output LaTeX (no surrounding triple backticks).
- Table: output GitHub Flavored Markdown table(s) using `|` pipes and a header separator row like `| --- | --- |`.
- All others: output Markdown (keep line breaks when present).

Constraints:
- The output text must be the original text from the image, with no translation.
- Keep numbers/currencies/units exactly as shown.
- Sort layout elements in human reading order.
- Return ONLY valid JSON (a single JSON array), no extra commentary.
""",

    # prompt_layout_only_en: layout detection only
    "prompt_layout_only_en": """Please detect layout elements in this PDF image and return them as a JSON array.

Each array item is an object with:
- bbox: [x1, y1, x2, y2]
- category: one of ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

Constraints:
- Sort layout elements in human reading order.
- Return ONLY valid JSON (a single JSON array), no extra commentary.
""",

    # prompt_ocr: plain OCR text
    "prompt_ocr": """Extract all visible text content from this image. Output ONLY GitHub Flavored Markdown (no triple backticks). If there is any table, convert it to a Markdown table using `|` pipes.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract the text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",
}

