import base64
import io
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from loguru import logger

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

from mineru_server.prompts import build_caption_system_prompt


IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_rel_path(raw_url: str) -> Optional[str]:
    token = (raw_url or "").strip()
    if not token:
        return None
    if "://" in token:
        parsed = urlparse(token)
        token = parsed.path
    token = token.split("?")[0].split("#")[0].strip().replace("\\", "/")
    token = token.lstrip("./")
    return token or None


def escape_md_alt(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("]", r"\]")
    return text[:220]


def looks_like_hash_filename(name: str) -> bool:
    base = Path(name).stem
    if len(base) < 16:
        return False
    return bool(re.fullmatch(r"[0-9a-fA-F]{16,}", base))


def is_generic_caption(text: str, language: Optional[str]) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    low = re.sub(r"\s+", " ", low).strip()

    if low in {
        "test",
        "caption",
        "test caption",
        "image caption",
        "a caption",
        "n/a",
        "none",
        "null",
        "unknown",
        "untitled",
    }:
        return True

    # Generic noun phrases like "an image", "a figure", "a table".
    if re.fullmatch(r"(an?|the)\s+(image|figure|diagram|chart|photo|screenshot|illustration|table|formula|equation)", low):
        return True

    if language == "en" and low in {"image", "figure", "diagram", "chart", "photo", "screenshot", "illustration", "table", "equation", "formula"}:
        return True
    if language == "zh" and t in {"图片", "图", "示意图", "图示", "插图", "照片", "表格", "公式"}:
        return True

    return False


def is_caption_meaningful(caption: str, *, base_name: str, language: Optional[str]) -> bool:
    text = (caption or "").strip()
    if not text:
        return False
    if is_generic_caption(text, language):
        return False

    lower = text.lower()
    base_lower = (base_name or "").lower()

    if base_lower and base_lower in lower:
        return False
    if re.search(r"\.(png|jpe?g|webp|gif|bmp|tiff?)\b", lower):
        return False
    if looks_like_hash_filename(text):
        return False
    if re.search(r"\b[0-9a-f]{24,}\b", lower):
        return False

    if language == "zh":
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return cjk >= 2
    if language == "en":
        letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
        return letters >= 3
    return True


def detect_doc_language(markdown_text: str, fallback_text: str = "") -> str:
    sample = (markdown_text or "").strip()
    if not sample:
        sample = (fallback_text or "").strip()
    if not sample:
        return "en"
    sample = sample[:4000]
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    if cjk >= 30 and (cjk / max(1, len(sample))) >= 0.02:
        return "zh"
    return "en"


def fallback_caption(idx: int, language: str) -> str:
    return f"图{idx}" if language == "zh" else f"Figure {idx}"


def build_caption_map_from_content_list(content_list: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(content_list, list):
        return out
    for item in content_list:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        img_path = str(item.get("img_path") or "").strip()
        caption_data = item.get("image_caption") or []

        caption_value = ""
        if isinstance(caption_data, list):
            for entry in caption_data:
                if isinstance(entry, str) and entry.strip():
                    caption_value = entry.strip()
                    break
        elif isinstance(caption_data, str) and caption_data.strip():
            caption_value = caption_data.strip()

        if not caption_value:
            continue

        candidates = set()
        if img_path:
            candidates.add(img_path)
            base = Path(img_path).name
            if base:
                candidates.add(base)
                candidates.add(f"images/{base}")
        for key in candidates:
            if key:
                out[key] = caption_value
    return out


def iter_blocks_v2(content_list_v2: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(content_list_v2, dict):
        yield content_list_v2
        return
    if isinstance(content_list_v2, list):
        for item in content_list_v2:
            yield from iter_blocks_v2(item)
        return


def build_caption_map_from_content_list_v2(content_list_v2: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for block in iter_blocks_v2(content_list_v2):
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        content = block.get("content")
        if not isinstance(content, dict):
            continue

        image_source = content.get("image_source")
        if not isinstance(image_source, dict):
            continue
        img_path = str(image_source.get("path") or "").strip()
        if not img_path:
            continue
        base_name = Path(img_path).name
        if not base_name:
            continue

        caption_list: Any = None
        if block_type in ("image", "figure"):
            caption_list = content.get("image_caption")
        elif block_type == "table":
            caption_list = content.get("table_caption")

        caption_value = ""
        if isinstance(caption_list, list):
            for entry in caption_list:
                if isinstance(entry, str) and entry.strip():
                    caption_value = entry.strip()
                    break
        elif isinstance(caption_list, str) and caption_list.strip():
            caption_value = caption_list.strip()

        if not caption_value:
            continue

        for key in {base_name, f"images/{base_name}", img_path}:
            if key:
                out[key] = caption_value
    return out


class ChatCaptioner:
    """OpenAI-compatible multimodal chat client used for generating short captions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: int = 60,
        *,
        fixed_context: str = "",
    ):
        if requests is None:
            raise RuntimeError("requests is required for captioning but is not installed.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = int(timeout_s)
        self.fixed_context = str(fixed_context or "")
        self._session = requests.Session()
        self._unavailable_until = 0.0
        self._cooldown_s = 300.0

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            raw = image_path.read_bytes()
            encoded = base64.b64encode(raw).decode("utf-8")
            suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
            return f"data:image/{suffix};base64,{encoded}"

    def _system_prompt(self, language: str, attempt: int) -> str:
        return build_caption_system_prompt(language=language, attempt=attempt, fixed_context=self.fixed_context)

    def caption_image(
        self,
        image_path: Path,
        language: str,
        *,
        attempt: int = 1,
        local_context: str = "",
    ) -> str:
        now = time.time()
        if now < self._unavailable_until:
            return ""

        data_url = self._image_to_data_url(image_path)
        system_prompt = self._system_prompt(language, attempt)
        local_context = (local_context or "").strip()
        user_items = []
        if local_context:
            user_items.append({"type": "text", "text": f"Document context around this image:\n{local_context}"})
        user_items.append({"type": "text", "text": "Caption this image."})
        user_items.append({"type": "image_url", "image_url": {"url": data_url}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_items,
                },
            ],
            "temperature": 0.2,
            "max_tokens": 64,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            connect_timeout = min(8.0, float(self.timeout_s))
            resp = self._session.post(
                self._endpoint(),
                json=payload,
                headers=headers,
                timeout=(connect_timeout, float(self.timeout_s)),
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = str(text or "").strip().replace("\n", " ").strip()
            return text[:200]
        except Exception as exc:
            try:
                if isinstance(
                    exc,
                    (
                        requests.exceptions.ConnectTimeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout,
                    ),
                ):
                    self._unavailable_until = time.time() + self._cooldown_s
            except Exception:
                pass
            logger.warning(f"Captioning failed for {image_path.name}: {exc}")
            return ""


def build_captioner(
    *,
    caption_mode: str,
    chat_api_base_url: Optional[str],
    chat_api_key: Optional[str],
    chat_api_key_file: Optional[str],
    chat_model: str,
    chat_timeout_s: int,
    caption_context: str = "",
    caption_context_file: Optional[str] = None,
) -> Optional[ChatCaptioner]:
    if caption_mode not in ("llm", "content_list_then_llm"):
        return None

    api_key = chat_api_key
    if not api_key and chat_api_key_file:
        try:
            api_key = Path(chat_api_key_file).read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"Failed to read chat api key file {chat_api_key_file}: {exc}")
            api_key = None

    if not (chat_api_base_url and api_key and chat_model):
        return None

    fixed_context = (caption_context or "").strip()
    if (not fixed_context) and caption_context_file:
        try:
            fixed_context = Path(caption_context_file).read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as exc:
            logger.warning(f"Failed to read caption context file {caption_context_file}: {exc}")
            fixed_context = ""

    return ChatCaptioner(
        base_url=chat_api_base_url,
        api_key=api_key,
        model=chat_model,
        timeout_s=chat_timeout_s,
        fixed_context=fixed_context,
    )


def build_token_spans(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    starts: list[int] = []
    spans: list[tuple[int, int]] = []
    for m in TOKEN_PATTERN.finditer(text or ""):
        starts.append(m.start())
        spans.append((m.start(), m.end()))
    return starts, spans


def extract_context_window_tokens(text: str, *, char_pos: int, up: int, down: int) -> str:
    """
    Extract a substring covering [up, down] tokens around `char_pos`.
    Tokenization is a lightweight regex approximation.
    """
    if not text:
        return ""
    up = max(0, int(up))
    down = max(0, int(down))
    starts, spans = build_token_spans(text)
    if not spans:
        return ""

    import bisect

    i = bisect.bisect_right(starts, max(0, int(char_pos))) - 1
    if i < 0:
        i = 0
    else:
        # If char_pos is beyond this token span, move to next token.
        if char_pos >= spans[i][1] and i + 1 < len(spans):
            i += 1

    left_i = max(0, i - up)
    right_i = min(len(spans), i + down + 1)
    if right_i <= left_i:
        left_i = max(0, i - up)
        right_i = min(len(spans), i + 1)

    start_char = spans[left_i][0]
    end_char = spans[right_i - 1][1]
    return (text[start_char:end_char] or "").strip()
