import re


def normalize_entity_key(name: str) -> str:
    return " ".join(str(name or "").strip().casefold().split())


def iter_candidate_phrases(text: str, *, max_words: int) -> list[str]:
    if max_words <= 0:
        return []
    pattern = rf"\b[A-Z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){{0,{max_words - 1}}}\b"
    return [m.group(0).strip() for m in re.finditer(pattern, text or "")]

