"""Stopword utilities.

Primary source: the lightweight `stopwordsiso` package (if installed).
Fallback source: bundled stopword text files used by the BM25 tokenizer manager.
"""
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_stopwords_file(path: str | Path) -> frozenset[str]:
    file_path = Path(path)
    if not file_path.exists():
        return frozenset()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="gbk", errors="ignore")
    tokens: list[str] = []
    for line in raw.splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            tokens.append(token)
    return frozenset(tokens)


def _fallback_stopwords(lang: str) -> frozenset[str]:
    root = _project_root()
    stopwords_dir = root / "encapsulation" / "database" / "utils" / "stopwords"
    if lang.startswith("zh"):
        return load_stopwords_file(stopwords_dir / "chinese_default.txt")
    if lang.startswith("en"):
        return load_stopwords_file(stopwords_dir / "english_default.txt")
    return frozenset()


@lru_cache(maxsize=16)
def get_stopwords(lang: str) -> frozenset[str]:
    """Return stopwords for a language code (e.g. 'en', 'zh', 'zh-cn').

    The returned set is best-effort and may be empty if no source is available.
    """

    value = (lang or "").strip().lower().replace("_", "-")
    if not value:
        return frozenset()

    try:
        from stopwordsiso import stopwords  # type: ignore

        words = stopwords(value)
        if words:
            return frozenset(words)
    except Exception:
        pass

    return _fallback_stopwords(value)
