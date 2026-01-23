from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mineru_server import captioning
from mineru_server.config import safe_load_json


@dataclass
class AssetPostProcessor:
    caption_mode: str
    captioner: Optional[captioning.ChatCaptioner]
    caption_max_images: int
    caption_up_tokens: int = 500
    caption_down_tokens: int = 500

    def process(
        self,
        *,
        task_id: str,
        task_root: Path,
        doc_name: str,
        method_dir: Path,
        markdown_path: Optional[Path],
        content_list_path: Optional[Path],
    ) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
        if markdown_path is None or not markdown_path.exists():
            return None, []

        images_dir = method_dir / "images"
        if not images_dir.exists():
            return None, []

        original_md = markdown_path.read_text(encoding="utf-8", errors="ignore")

        content_list = safe_load_json(content_list_path) if content_list_path else None
        content_list_v2_path = method_dir / f"{doc_name}_content_list_v2.json"
        content_list_v2 = safe_load_json(content_list_v2_path) if content_list_v2_path.exists() else None

        caption_map = {
            **captioning.build_caption_map_from_content_list(content_list),
            **captioning.build_caption_map_from_content_list_v2(content_list_v2),
        }

        # Build a token-context stream from content_list_v2 so we can still provide context even when
        # VLM markdown doesn't reference images (we append a gallery at the end).
        v2_stream, v2_marker_pos = _build_v2_context_stream(content_list_v2)
        v2_anchors = _build_v2_anchors(content_list_v2)

        # language detection uses md + some fallback textual content from content_list_v2
        fallback_text = ""
        if isinstance(content_list, list):
            parts: List[str] = []
            for item in content_list:
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
                        if sum(len(p) for p in parts) >= 2000:
                            break
            fallback_text = "\n".join(parts)
        language = captioning.detect_doc_language(original_md, fallback_text)

        images: List[Dict[str, Any]] = []
        by_filename: Dict[str, Dict[str, Any]] = {}
        referenced: set[str] = set()

        def record_image(filename: str) -> Dict[str, Any]:
            if filename in by_filename:
                return by_filename[filename]
            idx = len(images) + 1
            rel_for_md = f"images/{filename}"
            task_rel = (method_dir / "images" / filename).relative_to(task_root).as_posix()
            entry = {
                "id": idx,
                "filename": filename,
                "relative_path": rel_for_md,  # for markdown
                "task_rel_path": task_rel,  # for downloading via /task/<id>/file/<path>
                "absolute_path": str(images_dir / filename),
                "caption": "",
                "caption_source": "",
                "exists": (images_dir / filename).exists(),
                "uri": f"{task_id}/{rel_for_md}",
                # Track positions separately so context windows stay meaningful:
                # - original: position in the original markdown before we insert/move images
                # - updated: position in the updated markdown after insertion/moves
                "md_char_pos_original": None,
                "md_char_pos_updated": None,
                # Back-compat: best known position (prefer updated).
                "md_char_pos": None,
            }
            images.append(entry)
            by_filename[filename] = entry
            return entry

        def pick_initial_caption(*, filename: str, alt: str, url: str) -> Tuple[str, str]:
            normalized = captioning.normalize_rel_path(url) or ""
            base_name = Path(normalized).name or filename

            for key in (base_name, f"images/{base_name}", normalized):
                candidate = caption_map.get(key)
                if candidate and captioning.is_caption_meaningful(candidate, base_name=base_name, language=language):
                    return candidate.strip(), "content_list"

            if alt and captioning.is_caption_meaningful(alt, base_name=base_name, language=language):
                return alt.strip(), "markdown_alt"

            return "", ""

        def rewrite(match):
            alt = (match.group("alt") or "").strip()
            url = (match.group("url") or "").strip()
            normalized = captioning.normalize_rel_path(url)
            if not normalized:
                return match.group(0)

            filename = Path(normalized).name
            if not filename:
                return match.group(0)

            referenced.add(filename)
            entry = record_image(filename)
            if entry.get("md_char_pos_original") is None:
                entry["md_char_pos_original"] = int(match.start())
            if entry.get("md_char_pos") is None:
                entry["md_char_pos"] = int(match.start())
            if not entry["caption"]:
                cap, src = pick_initial_caption(filename=filename, alt=alt, url=url)
                entry["caption"] = cap
                entry["caption_source"] = src

            caption_out = entry["caption"] or captioning.fallback_caption(int(entry["id"]), language)
            return f"![{captioning.escape_md_alt(caption_out)}](images/{filename})"

        updated_md = captioning.IMAGE_PATTERN.sub(rewrite, original_md)

        # If a previous run appended a trailing `## Images` gallery, try to relocate those images
        # back near their contextual anchor (so retrieval keeps image+text together).
        gallery_pos = _find_gallery_header_pos(updated_md)
        if gallery_pos is not None:
            prefix = updated_md[:gallery_pos]
            suffix = updated_md[gallery_pos:]
            moved_any = False

            # Consider images whose first occurrence is inside the gallery suffix.
            candidates = sorted(
                [e for e in images if isinstance(e.get("md_char_pos"), int) and int(e["md_char_pos"]) >= gallery_pos],
                key=lambda e: int(e.get("md_char_pos") or 0),
            )
            for entry in candidates:
                filename = str(entry.get("filename") or "")
                if not filename:
                    continue
                anchors = v2_anchors.get(filename) or []
                if not anchors:
                    continue
                insert_at = None
                for anchor in anchors:
                    pos = _find_anchor_pos(prefix, anchor)
                    if pos is not None:
                        insert_at = pos
                        break
                if insert_at is None:
                    continue

                # Remove from suffix (the gallery), then insert into prefix.
                new_suffix, removed = _remove_image_ref_from_suffix(suffix, filename)
                if not removed:
                    continue

                cap = entry.get("caption") or captioning.fallback_caption(int(entry["id"]), language)
                image_line = f"\n\n![{captioning.escape_md_alt(cap)}](images/{filename})\n\n"
                prefix, ins_pos = _insert_after_paragraph(prefix, at=insert_at, insert_text=image_line)
                entry["md_char_pos_updated"] = ins_pos
                entry["md_char_pos"] = ins_pos
                moved_any = True
                suffix = new_suffix

            if moved_any:
                suffix = _cleanup_empty_gallery_suffix(suffix)
                updated_md = (prefix + suffix).rstrip() + "\n"

        # Ensure all extracted images are represented (even if markdown references only some of them).
        image_files = sorted(
            [
                p.name
                for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
            ]
        )
        extra_files: List[str] = []
        for filename in image_files:
            if filename in by_filename:
                continue
            extra_files.append(filename)
            entry = record_image(filename)
            if not entry["caption"]:
                for key in (filename, f"images/{filename}"):
                    candidate = caption_map.get(key)
                    if candidate and captioning.is_caption_meaningful(candidate, base_name=filename, language=language):
                        entry["caption"] = candidate.strip()
                        entry["caption_source"] = "content_list"
                        break

        # Place unreferenced images near their context, instead of appending everything at the end.
        # We use content_list_v2 anchors (table/paragraph/title) to locate an insertion point in markdown.
        unplaced: List[str] = []
        for filename in extra_files:
            entry = by_filename.get(filename)
            if not entry:
                continue
            cap = entry.get("caption") or captioning.fallback_caption(int(entry["id"]), language)
            image_line = f"\n\n![{captioning.escape_md_alt(cap)}](images/{entry['filename']})\n\n"
            inserted = False

            for anchor in (v2_anchors.get(filename) or []):
                pos = _find_anchor_pos(updated_md, anchor)
                if pos is None:
                    continue
                updated_md, ins_pos = _insert_after_paragraph(updated_md, at=pos, insert_text=image_line)
                entry["md_char_pos_updated"] = ins_pos
                entry["md_char_pos"] = ins_pos
                referenced.add(filename)
                inserted = True
                break

            if not inserted:
                unplaced.append(filename)

        # Fallback: append a small gallery section only for images we couldn't place.
        if unplaced:
            header = "## 图像" if language == "zh" else "## Images"
            lines: List[str] = []
            for filename in unplaced:
                entry = by_filename.get(filename)
                if not entry:
                    continue
                cap = entry.get("caption") or captioning.fallback_caption(int(entry["id"]), language)
                lines.append(f"![{captioning.escape_md_alt(cap)}](images/{entry['filename']})")
                referenced.add(filename)
            updated_md = (updated_md.rstrip() + "\n\n" + header + "\n\n" + "\n".join(lines) + "\n")

        # Special case: if markdown had no image references at all and there were no extra_files
        # (rare), still ensure we have a gallery so images are not lost.
        if (not referenced) and images:
            header = "## 图像" if language == "zh" else "## Images"
            lines = [
                f"![{captioning.escape_md_alt((entry.get('caption') or captioning.fallback_caption(int(entry['id']), language)))}](images/{entry['filename']})"
                for entry in images
            ]
            updated_md = (updated_md.rstrip() + "\n\n" + header + "\n\n" + "\n".join(lines) + "\n")
            referenced.update([e["filename"] for e in images if e.get("filename")])

        wants_llm = self.caption_mode in ("llm", "content_list_then_llm") and self.captioner is not None
        pending: List[Dict[str, Any]] = []
        if wants_llm:
            for entry in images:
                filename = entry.get("filename") or ""
                if not filename:
                    continue
                if captioning.is_caption_meaningful(entry.get("caption") or "", base_name=filename, language=language):
                    continue
                if not (images_dir / filename).exists():
                    continue
                pending.append(entry)

        # `caption_max_images` is a safety knob for LLM captioning.
        # If <= 0, treat it as "no limit" (caption all pending images).
        limit = int(self.caption_max_images)
        if limit > 0:
            pending = pending[:limit]
        if wants_llm and pending:
            for entry in pending:
                filename = str(entry["filename"])
                image_path = images_dir / filename
                local_ctx = ""
                if isinstance(entry.get("md_char_pos_original"), int):
                    local_ctx = captioning.extract_context_window_tokens(
                        original_md,
                        char_pos=int(entry["md_char_pos_original"]),
                        up=self.caption_up_tokens,
                        down=self.caption_down_tokens,
                    )
                elif isinstance(entry.get("md_char_pos_updated"), int):
                    local_ctx = captioning.extract_context_window_tokens(
                        updated_md,
                        char_pos=int(entry["md_char_pos_updated"]),
                        up=self.caption_up_tokens,
                        down=self.caption_down_tokens,
                    )
                if not local_ctx and v2_stream and filename in v2_marker_pos:
                    local_ctx = captioning.extract_context_window_tokens(
                        v2_stream,
                        char_pos=int(v2_marker_pos[filename]),
                        up=self.caption_up_tokens,
                        down=self.caption_down_tokens,
                    )

                cap = self.captioner.caption_image(image_path, language, attempt=1, local_context=local_ctx)
                if not (cap and captioning.is_caption_meaningful(cap, base_name=filename, language=language)):
                    cap = self.captioner.caption_image(image_path, language, attempt=2, local_context=local_ctx)

                if cap and captioning.is_caption_meaningful(cap, base_name=filename, language=language):
                    entry["caption"] = cap
                    entry["caption_source"] = "llm"
                else:
                    entry["caption"] = captioning.fallback_caption(int(entry["id"]), language)
                    entry["caption_source"] = "fallback"

            def rewrite_final(match):
                url = (match.group("url") or "").strip()
                normalized = captioning.normalize_rel_path(url)
                if not normalized:
                    return match.group(0)
                filename = Path(normalized).name
                if not filename or filename not in by_filename:
                    return match.group(0)
                entry = by_filename[filename]
                cap = entry.get("caption") or captioning.fallback_caption(int(entry["id"]), language)
                return f"![{captioning.escape_md_alt(cap)}](images/{filename})"

            updated_md = captioning.IMAGE_PATTERN.sub(rewrite_final, updated_md)

        # Ensure caption_source
        for entry in images:
            if entry.get("caption_source"):
                continue
            filename = entry.get("filename") or ""
            cap = entry.get("caption") or ""
            if captioning.is_caption_meaningful(cap, base_name=filename, language=language):
                entry["caption_source"] = "unknown"
            else:
                entry["caption"] = captioning.fallback_caption(int(entry["id"]), language)
                entry["caption_source"] = "fallback"

        if updated_md != original_md:
            markdown_path.write_text(updated_md, encoding="utf-8")

        manifest_path = markdown_path.with_name(f"{markdown_path.stem}_assets.json")
        manifest = {
            "task_id": task_id,
            "document": doc_name,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
            "language": language,
            "images": images,
        }
        manifest_path.write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path, images


def _extract_text_from_v2_block(block: Dict[str, Any]) -> str:
    t = str(block.get("type") or "")
    content = block.get("content")
    if not isinstance(content, dict):
        return ""

    if t == "title":
        title = content.get("title_content") or []
        parts = []
        if isinstance(title, list):
            for item in title:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("content") or "").strip())
        return " ".join([p for p in parts if p])

    if t == "paragraph":
        para = content.get("paragraph_content") or []
        parts = []
        if isinstance(para, list):
            for item in para:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("content") or "").strip())
                elif item.get("type") in ("equation_inline", "equation_interline"):
                    parts.append(str(item.get("content") or "").strip())
        return " ".join([p for p in parts if p])

    if t == "table":
        html = str(content.get("html") or "").strip()
        return html

    if t == "equation_interline":
        mc = content.get("math_content") or ""
        return str(mc).strip()

    return ""


def _build_v2_context_stream(content_list_v2: Any) -> tuple[str, Dict[str, int]]:
    """
    Build a pseudo-markdown stream where each image block inserts a marker:
      ![img](images/<filename>)
    so we can extract token windows around it.
    """
    if not content_list_v2:
        return "", {}

    blocks = list(captioning.iter_blocks_v2(content_list_v2))
    parts: List[str] = []
    marker_pos: Dict[str, int] = {}
    cursor = 0

    def push(s: str) -> None:
        nonlocal cursor
        if not s:
            return
        parts.append(s)
        cursor += len(s)

    for block in blocks:
        if not isinstance(block, dict):
            continue
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        image_source = content.get("image_source")
        filename = ""
        if isinstance(image_source, dict):
            p = str(image_source.get("path") or "").strip()
            filename = Path(p).name if p else ""

        if filename and filename not in marker_pos:
            marker = f"![img](images/{filename})\n"
            marker_pos[filename] = cursor
            push(marker)

        text = _extract_text_from_v2_block(block)
        if text:
            push(text + "\n\n")

    return "".join(parts).strip(), marker_pos


def _strip_md_noise(text: str) -> str:
    s = (text or "").replace("\\", "")
    s = __import__("re").sub(r"\s+", " ", s).strip()
    return s


def _build_v2_anchors(content_list_v2: Any) -> Dict[str, List[str]]:
    """
    For each image filename, build a list of nearby textual anchors from content_list_v2,
    preferring the closest *following* paragraph/title/table text, then preceding.
    """
    if not content_list_v2:
        return {}

    blocks = list(captioning.iter_blocks_v2(content_list_v2))
    texts: List[str] = []
    image_at: List[tuple[int, str]] = []

    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        content = block.get("content")
        if isinstance(content, dict) and isinstance(content.get("image_source"), dict):
            p = str(content["image_source"].get("path") or "").strip()
            filename = Path(p).name if p else ""
            if filename:
                image_at.append((len(texts), filename))
            # still allow text extraction for tables/equations with image_source; handled below too

        t = _extract_text_from_v2_block(block)
        if t:
            texts.append(_strip_md_noise(t))

    anchors: Dict[str, List[str]] = {}
    if not texts:
        return anchors

    for pos, filename in image_at:
        candidates: List[str] = []
        # Prefer next text chunk
        if 0 <= pos < len(texts):
            # next
            if pos < len(texts):
                candidates.append(texts[pos])
            # prev
            if pos - 1 >= 0:
                candidates.append(texts[pos - 1])
        # Deduplicate, keep order, and keep only reasonably short anchors
        out: List[str] = []
        seen = set()
        for c in candidates:
            c = (c or "").strip()
            if not c:
                continue
            c = c[:300].strip()
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        if out:
            anchors[filename] = out

    return anchors


def _find_anchor_pos(markdown_text: str, anchor: str) -> Optional[int]:
    """
    Case-insensitive substring search for an anchor (and a shorter prefix if needed).
    Returns char index or None.
    """
    hay = markdown_text or ""
    needle = _strip_md_noise(anchor)
    if not hay or not needle:
        return None
    hay_low = hay.lower()
    needle_low = needle.lower()
    idx = hay_low.find(needle_low)
    if idx >= 0:
        return idx
    # fall back to searching a prefix
    prefix = needle_low[:80].strip()
    if prefix and len(prefix) >= 12:
        idx = hay_low.find(prefix)
        if idx >= 0:
            return idx
    return None


def _insert_after_paragraph(markdown_text: str, *, at: int, insert_text: str) -> tuple[str, int]:
    """
    Insert after the paragraph that contains char index `at`.
    Returns (new_text, insertion_char_pos).
    """
    text = markdown_text or ""
    at = max(0, min(len(text), int(at)))
    # find end of paragraph (double newline), otherwise end
    para_end = text.find("\n\n", at)
    if para_end == -1:
        para_end = len(text)
        suffix = "\n\n"
    else:
        para_end = para_end + 2
        suffix = ""
    insertion_pos = para_end
    new_text = text[:para_end] + insert_text + suffix + text[para_end:]
    return new_text, insertion_pos


def _find_gallery_header_pos(markdown_text: str) -> Optional[int]:
    text = markdown_text or ""
    # Prefer the last gallery header (closest to the end).
    candidates = []
    for marker in ("\n## Images\n", "\n## 图像\n"):
        pos = text.rfind(marker)
        if pos != -1:
            candidates.append(pos)
    return max(candidates) if candidates else None


def _remove_image_ref_from_suffix(suffix: str, filename: str) -> tuple[str, bool]:
    pat = __import__("re").compile(rf"(?m)^!\[[^\]]*\]\(images/{__import__('re').escape(filename)}\)\s*$\n?")
    new, n = pat.subn("", suffix, count=1)
    return new, (n > 0)


def _cleanup_empty_gallery_suffix(suffix: str) -> str:
    # If the gallery header exists but has no remaining image refs, drop it.
    for header in ("## Images", "## 图像"):
        pat = __import__("re").compile(rf"(?s)\n{__import__('re').escape(header)}\n(?:\s*\n)*\Z")
        suffix = pat.sub("\n", suffix)
    return suffix
