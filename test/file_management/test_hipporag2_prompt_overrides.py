from types import SimpleNamespace

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor


def _make_extractor(*, ner_prompt: str | None = None, triple_prompt: str | None = None) -> HippoRAG2Extractor:
    extractor = HippoRAG2Extractor.__new__(HippoRAG2Extractor)
    extractor.config = SimpleNamespace(ner_prompt=ner_prompt, triple_prompt=triple_prompt)
    extractor.entity_types = None
    return extractor


def test_custom_ner_prompt_is_used_when_configured() -> None:
    extractor = _make_extractor(ner_prompt="CUSTOM_NER {passage}")
    prompt = extractor.build_ner_prompt("hello")
    assert "CUSTOM_NER" in prompt
    assert "hello" in prompt


def test_custom_triple_prompt_is_used_when_configured() -> None:
    extractor = _make_extractor(triple_prompt="CUSTOM_TRIPLE {entities} {passage}")
    prompt = extractor.build_triple_prompt("hello", ["A", "B"])
    assert "CUSTOM_TRIPLE" in prompt
    assert "A" in prompt
    assert "hello" in prompt

