"""
Prompt templates used by MinerU server.

Centralized here to avoid scattering prompt strings across business logic.
"""
def build_caption_system_prompt(*, language: str, attempt: int, fixed_context: str = "") -> str:
    """
    Build the system prompt for multimodal captioning.

    Notes:
    - `fixed_context` is optional user-provided context prepended to the instruction.
    - Strings are intentionally kept stable for reproducibility.
    """
    language = (language or "").strip().lower() or "en"
    attempt = int(attempt or 1)

    if language == "zh":
        if attempt <= 1:
            instruction = (
                "你是文档解析助手。请根据图片内容生成一个用于检索的简短中文图注。"
                "要求：1) 只输出一行；2) 8-16 个汉字左右；3) 不要包含文件名或哈希；4) 不要加引号或 Markdown。"
            )
        else:
            instruction = (
                "你是文档解析助手。请生成一个具体、可检索的中文图注。"
                "要求：1) 只输出一行；2) 10-18 个汉字；3) 必须描述画面中的具体对象/关系（例如“注意力偏置矩阵示例”“表格结构偏置示意”“公式推导步骤”）；"
                "4) 禁止输出“示意图/图片/图/表格/公式”等过于泛化的单词；5) 不要包含文件名或哈希；6) 不要加引号或 Markdown。"
            )
    else:
        if attempt <= 1:
            instruction = (
                "You are a document parsing assistant. Generate a short image caption for retrieval."
                " Requirements: (1) output a single line; (2) 3-10 words; (3) no filename/hash; (4) no quotes or Markdown."
            )
        else:
            instruction = (
                "You are a document parsing assistant. Generate a specific, searchable image caption."
                " Requirements: (1) output a single line; (2) 4-12 words; (3) must mention a concrete subject (e.g., 'attention bias matrix', 'table structure example', 'equation derivation');"
                " (4) do not output generic text like 'image', 'figure', 'table', or 'test caption'; (5) no filename/hash; (6) no quotes or Markdown."
            )

    ctx = (fixed_context or "").strip()
    if not ctx:
        return instruction
    if language == "zh":
        return f"{ctx}\n\n【生成要求】\n{instruction}"
    return f"{ctx}\n\n[Generation Rules]\n{instruction}"

