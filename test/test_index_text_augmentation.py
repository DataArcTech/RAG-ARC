import importlib


def _reload_index_text_modules():
    # Reload in dependency order so env changes are reflected.
    import config.index_text_augmentation as cfg

    importlib.reload(cfg)
    import core.utils.index_text_augmentation as util

    importlib.reload(util)
    return cfg, util


def test_prepend_title_prefix_enabled_injects_title(monkeypatch):
    monkeypatch.setenv("INDEX_TEXT_TITLE_PREFIX_ENABLED", "true")
    monkeypatch.setenv("INDEX_TEXT_TITLE_MAX_CHARS", "160")
    _cfg, util = _reload_index_text_modules()

    out = util.prepend_title_prefix(text="hello world", filename="RAG-ARC/local/user_files/x/Some Product.pdf")
    assert out.startswith("title=Some Product\n")


def test_prepend_title_prefix_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("INDEX_TEXT_TITLE_PREFIX_ENABLED", "false")
    _cfg, util = _reload_index_text_modules()

    out = util.prepend_title_prefix(text="hello world", filename="RAG-ARC/local/user_files/x/Some Product.pdf")
    assert out == "hello world"


def test_prepend_title_prefix_does_not_duplicate(monkeypatch):
    monkeypatch.setenv("INDEX_TEXT_TITLE_PREFIX_ENABLED", "true")
    _cfg, util = _reload_index_text_modules()

    base = "title=Some Product\nhello"
    out = util.prepend_title_prefix(text=base, filename="Some Product.pdf")
    assert out == base
