from pathlib import Path
import importlib.util
import sys
import types

import pytest


class _StubCaptioner:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def caption_image(self, image_path: Path, language: str, *, attempt: int, local_context: str = "") -> str:
        self.calls.append(image_path)
        # Must be "meaningful" for captioning.is_caption_meaningful(language="en").
        return "useful diagram"


def _import_asset_post_processor(monkeypatch: pytest.MonkeyPatch):
    """
    Import `mineru_server.postprocess.AssetPostProcessor` without requiring the full
    MinerU server runtime dependencies (fastapi/loguru/...) to be installed in the
    unit test environment.
    """
    mineru_server_dir = Path(__file__).resolve().parents[2] / "mineru" / "mineru_server"

    # Some MinerU modules depend on loguru; keep tests hermetic with a tiny stub.
    loguru = types.ModuleType("loguru")

    class _Logger:
        def debug(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

        def error(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

    loguru.logger = _Logger()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "loguru", loguru)

    # Create a synthetic `mineru_server` package so we can load selected submodules
    # without executing `mineru_server/__init__.py` (which imports fastapi).
    pkg = types.ModuleType("mineru_server")
    pkg.__path__ = [str(mineru_server_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru_server", pkg)

    def _load(name: str) -> None:
        path = mineru_server_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"mineru_server.{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"mineru_server.{name}", module)
        spec.loader.exec_module(module)

    for mod in ("prompts", "captioning", "config", "postprocess"):
        _load(mod)

    from mineru_server.postprocess import AssetPostProcessor  # type: ignore  # noqa: E402

    return AssetPostProcessor


def _make_task_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    task_root = tmp_path / "task_root"
    method_dir = task_root / "doc" / "method"
    images_dir = method_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Minimal markdown referencing 3 images in order.
    md_path = method_dir / "doc.md"
    md_path.write_text(
        "\n".join(
            [
                "Some text.",
                "![image](images/a.png)",
                "![image](images/b.png)",
                "![image](images/c.png)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for name in ("a.png", "b.png", "c.png"):
        (images_dir / name).write_bytes(b"not-a-real-image")

    return task_root, method_dir, md_path


@pytest.mark.parametrize("caption_max_images", [0, -1])
def test_mineru_caption_max_images_leq_zero_means_no_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caption_max_images: int) -> None:
    AssetPostProcessor = _import_asset_post_processor(monkeypatch)

    task_root, method_dir, md_path = _make_task_layout(tmp_path)
    captioner = _StubCaptioner()
    post = AssetPostProcessor(caption_mode="llm", captioner=captioner, caption_max_images=caption_max_images)

    _manifest_path, images_meta = post.process(
        task_id="t1",
        task_root=task_root,
        doc_name="doc",
        method_dir=method_dir,
        markdown_path=md_path,
        content_list_path=None,
    )

    assert len(images_meta) == 3
    assert len(captioner.calls) == 3
    assert all(entry.get("caption_source") == "llm" for entry in images_meta)


def test_mineru_caption_max_images_positive_limits_llm_captioning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    AssetPostProcessor = _import_asset_post_processor(monkeypatch)

    task_root, method_dir, md_path = _make_task_layout(tmp_path)
    captioner = _StubCaptioner()
    post = AssetPostProcessor(caption_mode="llm", captioner=captioner, caption_max_images=2)

    _manifest_path, images_meta = post.process(
        task_id="t1",
        task_root=task_root,
        doc_name="doc",
        method_dir=method_dir,
        markdown_path=md_path,
        content_list_path=None,
    )

    assert len(images_meta) == 3
    assert len(captioner.calls) == 2
    assert [e.get("caption_source") for e in images_meta] == ["llm", "llm", "fallback"]
