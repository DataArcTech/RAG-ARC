import fitz

from config.core.file_management.chunker.chunker_config import TokenChunkerConfig


def test_token_chunker_does_not_introduce_unicode_replacement_chars_from_pdf():
    doc = fitz.open("test/test_pdf.pdf")
    try:
        text = "".join(page.get_text("text") for page in doc)
    finally:
        doc.close()

    assert "\ufffd" not in text

    # Intentionally tiny chunk size to force token-slice boundaries that can split multi-byte UTF-8 sequences.
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    chunks = chunker.chunk_text(text)

    assert chunks
    assert all("\ufffd" not in chunk["content"] for chunk in chunks)

