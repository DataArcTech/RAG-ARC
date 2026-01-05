from types import SimpleNamespace

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor


def _make_extractor() -> HippoRAG2Extractor:
    extractor = HippoRAG2Extractor.__new__(HippoRAG2Extractor)
    extractor.config = SimpleNamespace()
    extractor.entity_types = None
    return extractor


def test_parse_triple_response_accepts_case_insensitive_header() -> None:
    extractor = _make_extractor()
    raw = """### Triples
Radio City\tlocated in\tIndia
Radio City\tstarted on\t3 July 2001
"""
    triples = extractor.parse_triple_response(raw)
    assert ("Radio City", "located in", "India") in triples
    assert ("Radio City", "started on", "3 July 2001") in triples


def test_parse_triple_response_accepts_headerless_tsv_rows() -> None:
    extractor = _make_extractor()
    raw = """Radio City\tlocated in\tIndia
PlanetRadiocity.com\tlaunched in\tMay 2008
"""
    triples = extractor.parse_triple_response(raw)
    assert triples == [
        ("Radio City", "located in", "India"),
        ("PlanetRadiocity.com", "launched in", "May 2008"),
    ]


def test_parse_triple_response_ignores_non_triple_tsv_rows() -> None:
    extractor = _make_extractor()
    raw = """### ENTITIES
Radio City\tORGANIZATION
India\tLOCATION
"""
    assert extractor.parse_triple_response(raw) == []


def test_parse_triple_response_accepts_bulleted_rows_and_code_fences() -> None:
    extractor = _make_extractor()
    raw = """```tsv
### TRIPLES
- Radio City\tlocated in\tIndia
* Radio City\tstarted on\t3 July 2001
```"""
    triples = extractor.parse_triple_response(raw)
    assert triples == [
        ("Radio City", "located in", "India"),
        ("Radio City", "started on", "3 July 2001"),
    ]

