from config.core.file_management.chunker.chunker_config import TokenChunkerConfig


def _assert_math_atomic(text: str, math_block: str, marker: str) -> None:
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    chunks = chunker.chunk_text(text)
    math_chunks = [c for c in chunks if marker in (c.get("content") or "")]
    assert math_chunks, "expected at least one math chunk"
    assert len(math_chunks) == 1
    assert math_block in (math_chunks[0].get("content") or "")


def test_token_chunker_keeps_inline_math_atomic_dollar():
    inline_math = "$" + " ".join(["x"] * 60) + "$"
    text = ("prefix " * 20) + inline_math + (" suffix " * 20)
    _assert_math_atomic(text, inline_math, "$")


def test_token_chunker_keeps_inline_math_atomic_paren():
    inline_math = "\\(" + " ".join(["x"] * 60) + "\\)"
    text = ("prefix " * 20) + inline_math + (" suffix " * 20)
    _assert_math_atomic(text, inline_math, "\\(")


def test_token_chunker_keeps_display_math_block_atomic_dollars():
    block_math = "$$\n" + " + ".join(["x"] * 60) + "\n$$"
    text = ("prefix " * 20) + block_math + (" suffix " * 20)
    _assert_math_atomic(text, block_math, "$$")
