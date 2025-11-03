import os
import json
import logging
import io
from typing import List, Dict, Any, TYPE_CHECKING
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from PIL import Image
import fitz
from dotenv import load_dotenv

load_dotenv()

from .base import AbstractParser
from framework.singleton_decorator import singleton

# Import only necessary utilities
from .dots_ocr_utils.consts import image_extensions

if TYPE_CHECKING:
    from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig

logger = logging.getLogger(__name__)


@singleton
class VLMOcrParser(AbstractParser):
    """
    VLM-based OCR document parser implementation for simple text extraction.

    This class provides a lightweight document parsing solution using VLM Vision LLM service,
    focusing on pure OCR text extraction without layout analysis. Optimized for minimal
    token usage and fast processing.

    Key features:
    - Multi-format support: PDF, JPG, JPEG, PNG
    - Pure text extraction (no layout analysis)
    - Multi-threaded PDF processing for performance
    - Simple Markdown output
    - Uses encapsulation LLM service for model inference

    Configuration:
        llm_service: VLM OCR LLM service instance for inference
        dpi: DPI for PDF page conversion (default: 200)
        num_threads: Thread count for PDF processing
    """

    def __init__(self, config: "VLMOcrParserConfig"):
        """Initialize VLM OCR parser with LLM service"""
        super().__init__(config)

        # Get LLM service for inference
        vlm_ocr_config = getattr(self.config, 'vlm_ocr', None)
        if vlm_ocr_config is None:
            raise ValueError("VLM OCR parser requires vlm_ocr configuration")
        self.llm_service = vlm_ocr_config.build()

    async def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Parse a file (PDF or image) from binary data - simple OCR mode"""

        # Check if file type is supported
        base_filename, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lower()
        supported_extensions = self.get_supported_extensions()

        if file_ext not in supported_extensions:
            error_msg = f"File extension '{file_ext}' not supported. Supported extensions: {supported_extensions}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get output directory from environment variable
        output_dir = os.getenv('VLMOCR_OUTPUT_DIR', './vlmocr/output')
        output_dir = os.path.abspath(output_dir)
        save_dir = os.path.join(output_dir, base_filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self._parse_pdf(file_data, base_filename, save_dir, **kwargs)
        elif file_ext in image_extensions:
            results = self._parse_image(file_data, base_filename, save_dir, **kwargs)

        logger.info(f"Parsing finished, results saved to {save_dir}")

        with open(os.path.join(output_dir, f"{base_filename}.jsonl"), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return ['.pdf', '.jpg', '.jpeg', '.png']

    def _parse_image(
        self,
        file_data: bytes,
        filename: str,
        save_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Parse a single image file from binary data - simple OCR"""
        origin_image = Image.open(io.BytesIO(file_data))
        result = self._parse_single_image(
            origin_image, save_dir, filename, source="image", page_idx=0
        )
        result['filename'] = filename
        return [result]

    def _parse_pdf(
        self,
        file_data: bytes,
        filename: str,
        save_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Parse a PDF file from binary data - simple OCR"""

        logger.info(f"Loading PDF: {filename}")

        # Create fitz document from binary data
        pdf_doc = fitz.open("pdf", file_data)
        images_origin = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            # Convert page to image at specified DPI
            dpi = getattr(self.config, 'dpi', 200)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            images_origin.append(image)

        pdf_doc.close()
        total_pages = len(images_origin)

        tasks = [
            {
                "origin_image": image,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        # Use single thread for stability
        num_threads = min(total_pages, getattr(self.config, 'num_threads', 1))
        logger.info(f"Parsing PDF with {total_pages} pages using {num_threads} threads...")

        results = []
        with ThreadPool(num_threads) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for result in results:
            result['filename'] = filename
        return results

    def _parse_single_image(
        self,
        origin_image: Image.Image,
        save_dir: str,
        save_name: str,
        source: str = "image",
        page_idx: int = 0,
    ) -> Dict[str, Any]:
        """Parse a single image and return OCR text result - simplified version"""

        prompt = "Extract all text content from this image. Output the text in markdown format, preserving the structure and formatting as much as possible."

        # Use LLM service for inference
        response = self.llm_service.infer(origin_image, prompt)

        # Clean up markdown code blocks from response
        cleaned_response = self._clean_markdown_blocks(response)

        result = {
            'page_no': page_idx,
        }

        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"

        # Save markdown content
        md_file_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(cleaned_response)

        result.update({
            'md_content_path': md_file_path,
            'text_content': cleaned_response
        })

        return result

    def _clean_markdown_blocks(self, text: str) -> str:
        """Remove markdown code block markers (```markdown and ```) from text"""
        cleaned = text.strip()

        # Remove ```markdown at the beginning
        if cleaned.startswith('```markdown'):
            cleaned = cleaned[len('```markdown'):].lstrip('\n')
        elif cleaned.startswith('```'):
            # Handle generic ``` blocks
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]

        # Remove trailing ```
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].rstrip('\n')

        return cleaned



