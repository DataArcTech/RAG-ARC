import os
import asyncio
from pathlib import Path

import pytest

from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig


@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_VLM_OCR_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="Set RUN_RAGARC_VLM_OCR_TESTS=1 to run VLM OCR integration tests.",
)
def test_vlm_ocr_parser_smoke(tmp_path: Path):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_OCR_MODEL") or os.getenv("OCR_MODEL_NAME")
    if not api_key or not model_name:
        pytest.skip("Missing OPENAI_API_KEY / OPENAI_OCR_MODEL (or OCR_MODEL_NAME).")

    config = VLMOcrParserConfig(
        output_dir=str(tmp_path),
        vlm_ocr=VLMOcrConfig(
            loading_method="openai",
            model_name=model_name,
            openai_api_key=api_key,
            openai_base_url=base_url,
        ),
    )
    parser = config.build()

    test_pdf_path = Path(__file__).resolve().parents[1] / "test_pdf.pdf"
    if not test_pdf_path.exists():
        pytest.skip("Missing test PDF fixture.")

    file_data = test_pdf_path.read_bytes()
    results = asyncio.run(parser.parse_file(file_data, test_pdf_path.name))
    assert results is not None

