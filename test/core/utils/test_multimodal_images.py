from pathlib import Path

import pytest

from core.utils.multimodal_images import collect_image_paths_from_deepsearch_evidences


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_collect_images_from_deepsearch_evidence_provenance_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARSER_OUTPUT_DIR", str(tmp_path))
    file_id = "1f396b17-f355-45f6-8acd-0eda1292cb77"
    rel = "images/example.jpg"
    _touch(tmp_path / "mineru" / file_id / rel)

    evidences = [
        {
            "chunk_id": "chunk-1",
            "content": f"![图](%s)" % rel,
            "provenance": {
                "metadata": {
                    "filename": "RAG-ARC/demo.pdf",
                    "source_file_id": file_id,
                    "image_urls": [rel],
                }
            },
        }
    ]

    paths = collect_image_paths_from_deepsearch_evidences(evidences, max_images=10)
    assert [p.resolve() for p in paths] == [(tmp_path / "mineru" / file_id / rel).resolve()]


def test_collect_images_from_deepsearch_evidence_chunk_metadata_backcompat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PARSER_OUTPUT_DIR", str(tmp_path))
    file_id = "fd010412-46b8-4667-889b-162ee1a3c2b0"
    rel = "images/example2.jpg"
    _touch(tmp_path / "mineru" / file_id / rel)

    evidences = [
        {
            "chunk_id": "chunk-2",
            "content": f"![图](%s)" % rel,
            "provenance": {
                "metadata": {
                    "chunk_metadata": {
                        "filename": "RAG-ARC/demo2.pdf",
                        "source_file_id": file_id,
                        "image_urls": [rel],
                    }
                }
            },
        }
    ]

    paths = collect_image_paths_from_deepsearch_evidences(evidences, max_images=10)
    assert [p.resolve() for p in paths] == [(tmp_path / "mineru" / file_id / rel).resolve()]


def test_collect_images_from_metadata_when_content_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARSER_OUTPUT_DIR", str(tmp_path))
    file_id = "03576cff-a4b6-4ae2-8586-59700dd72747"
    rel = "images/example3.jpg"
    _touch(tmp_path / "mineru" / file_id / rel)

    evidences = [
        {
            "chunk_id": "chunk-3",
            # no markdown image in content; rely on metadata.image_urls
            "content": "",
            "provenance": {"metadata": {"filename": "RAG-ARC/demo3.pdf", "source_file_id": file_id, "image_urls": [rel]}},
        }
    ]

    paths = collect_image_paths_from_deepsearch_evidences(evidences, max_images=10)
    assert [p.resolve() for p in paths] == [(tmp_path / "mineru" / file_id / rel).resolve()]

