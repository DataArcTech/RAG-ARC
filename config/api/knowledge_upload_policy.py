"""Knowledge upload validation policy for API routers.

This module centralizes upload constraints so they aren't hard-coded inside
router handlers.
"""

from __future__ import annotations

from typing import Final


# Supported file extensions for the `/knowledge` upload endpoint.
ALLOWED_UPLOAD_EXTENSIONS: Final[set[str]] = {
    ".docx",
    ".xlsx",
    ".pptx",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".txt",
    ".html",
    ".md",
}

# Maximum upload size in bytes for the `/knowledge` upload endpoint.
MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024  # 10MB

