import pytest

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor


class _StubLLM:
    async def achat(self, _messages):  # noqa: ANN001
        raise AssertionError("LLM should not be called in prompt rendering tests")


class _StubLLMConfig:
    def build(self):
        return _StubLLM()


class _StubConfig:
    max_concurrent = 1
    entity_types = None
    error_policy = "raise"

    ner_prompt = None
    triple_prompt = None
    temporal_prompt = None
    sdf_hs_prompt = None

    ner_prompt_path = None
    triple_prompt_path = None
    temporal_prompt_path = None
    sdf_hs_prompt_path = None

    def __init__(self):
        self.llm_config = _StubLLMConfig()


def test_temporal_prompt_path_overrides_default(tmp_path) -> None:
    prompt_path = tmp_path / "temporal.txt"
    prompt_path.write_text("TEMPORAL({language}):: {passage}", encoding="utf-8")

    cfg = _StubConfig()
    cfg.temporal_prompt_path = str(prompt_path)
    extractor = HippoRAG2Extractor(cfg)

    rendered = extractor.build_temporal_prompt("Effective from 2024-01-01.")
    assert "TEMPORAL(en):: Effective from 2024-01-01." in rendered


def test_sdf_prompt_path_overrides_default(tmp_path) -> None:
    prompt_path = tmp_path / "sdf.txt"
    prompt_path.write_text("SDF({language}):: {passage}", encoding="utf-8")

    cfg = _StubConfig()
    cfg.sdf_hs_prompt_path = str(prompt_path)
    extractor = HippoRAG2Extractor(cfg)

    rendered = extractor.build_sdf_hs_prompt("Premium payment term is 5 years.")
    assert "SDF(en):: Premium payment term is 5 years." in rendered

