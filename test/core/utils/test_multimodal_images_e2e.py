"""End-to-end test: virtual image paths -> real MinerU image files.

These tests use the actual parsed_files on disk rather than mocked data.
They are skipped in CI environments where no parsed files exist.
"""
import os

import pytest
from pathlib import Path

# Resolve the real data directory (not the test-hermetic temp dir).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_STORE_BASE = _REPO_ROOT / "data" / "localdb"
PARSED_BASE = _REAL_STORE_BASE / "parsed_files" / "mineru"
HAS_PARSED = PARSED_BASE.is_dir() and any(PARSED_BASE.iterdir())


@pytest.fixture(autouse=True)
def _use_real_store(monkeypatch):
    """Point IO_STORE_BASE_PATH to the actual data directory for these E2E tests."""
    monkeypatch.setenv("IO_STORE_BASE_PATH", str(_REAL_STORE_BASE))
    monkeypatch.setenv("PARSER_OUTPUT_DIR", "io://parsed_files")


@pytest.mark.skipif(not HAS_PARSED, reason="No local parsed files")
class TestRealImageResolution:

    def test_germanwings_images_resolve(self):
        """795054b3 (germanwings PPT) has 50 images - verify resolution works."""
        from core.utils.multimodal_images import collect_image_paths_from_deepsearch_evidences

        fid = "795054b3-8b41-4212-8a0c-2ff36b2b6113"
        img_dir = PARSED_BASE / fid / "images"
        if not img_dir.is_dir():
            pytest.skip("germanwings not parsed locally")
        sample_img = next(img_dir.iterdir()).name
        evidences = [
            {
                "content": f"![img](images/{sample_img})",
                "provenance": {
                    "metadata": {
                        "source_file_id": fid,
                        "filename": "germanwings.pdf",
                        "image_urls": [f"images/{sample_img}"],
                    }
                },
            }
        ]
        paths = collect_image_paths_from_deepsearch_evidences(evidences, max_images=12)
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)

    def test_max_images_cap_at_12(self):
        """Find a file with 15+ images, verify cap at 12."""
        from core.utils.multimodal_images import collect_image_paths_from_deepsearch_evidences

        for fid_dir in PARSED_BASE.iterdir():
            img_dir = fid_dir / "images"
            if img_dir.is_dir() and len(list(img_dir.iterdir())) >= 15:
                imgs = sorted(img_dir.iterdir())[:15]
                evidences = [
                    {
                        "content": f"![img]({img.name})",
                        "provenance": {
                            "metadata": {
                                "source_file_id": fid_dir.name,
                                "filename": "test.pdf",
                                "image_urls": [f"images/{img.name}"],
                            }
                        },
                    }
                    for img in imgs
                ]
                paths = collect_image_paths_from_deepsearch_evidences(evidences, max_images=12)
                assert len(paths) == 12
                return
        pytest.skip("No file with 15+ images found")

    def test_think_images_from_evidence_chunks(self):
        """Verify ThinkTool._collect_think_images converts EvidenceChunk -> paths."""
        from encapsulation.data_model.deepsearch import EvidenceChunk
        from core.deepsearch.tools.think.think import ThinkTool

        fid = "795054b3-8b41-4212-8a0c-2ff36b2b6113"
        img_dir = PARSED_BASE / fid / "images"
        if not img_dir.is_dir():
            pytest.skip("germanwings not parsed locally")
        sample_img = next(img_dir.iterdir()).name
        ec = EvidenceChunk(
            chunk_id="test-1",
            source="read.pages",
            content=f"![img](images/{sample_img})",
            provenance={
                "metadata": {
                    "source_file_id": fid,
                    "filename": "germanwings.pdf",
                    "image_urls": [f"images/{sample_img}"],
                }
            },
        )
        paths = ThinkTool._collect_think_images([ec])
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
