import os
import json
import logging
import io
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from PIL import Image
import fitz
from dotenv import load_dotenv

load_dotenv()

from .base import AbstractParser
from framework.singleton_decorator import singleton

# Import DotsOCR utilities
from .dots_ocr_utils.image_utils import fetch_image, smart_resize, get_image_by_fitz_doc
from .dots_ocr_utils.prompts import dict_promptmode_to_prompt
from .dots_ocr_utils.layout_utils import pre_process_bboxes, post_process_output
# from .dots_ocr_utils.layout_utils import draw_layout_on_image  # Not needed - image output disabled
from .dots_ocr_utils.consts import MIN_PIXELS, MAX_PIXELS, image_extensions
from .dots_ocr_utils.format_transformer import layoutjson2md, clean_base64_images

if TYPE_CHECKING:
    from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig

logger = logging.getLogger(__name__)


@singleton
class DotsOCRParser(AbstractParser):
    """
    DotsOCR-based document parser implementation for advanced OCR and layout analysis.

    This class provides a complete document parsing solution using DotsOCR LLM service,
    containing all the parsing business logic including file handling, image processing,
    multi-threading, and output formatting. Uses the thin DotsOCR LLM service for inference.

    Key features:
    - Multi-format support: PDF, JPG, JPEG, PNG
    - Advanced layout analysis with bounding box detection
    - Multiple prompt modes for different parsing tasks
    - Multi-threaded PDF processing for performance
    - Structured output formats: JSON, Markdown, image annotations
    - Uses encapsulation LLM service for model inference

    Configuration:
        llm_service: DotsOCR LLM service instance for inference
        dpi: DPI for PDF page conversion (default: 200)
        min_pixels/max_pixels: Image size constraints
        num_threads: Thread count for PDF processing
    """

    def __init__(self, config: "DotsOCRParserConfig"):
        """Initialize DotsOCR parser with LLM service"""
        super().__init__(config)

        # Get LLM service for inference
        dots_ocr_config = getattr(self.config, 'dots_ocr', None)
        if dots_ocr_config is None:
            raise ValueError("DotsOCR parser requires dots_ocr configuration")
        self.llm_service = dots_ocr_config.build()

    async def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Parse a file (PDF or image) from binary data"""

        # Get parsing parameters from config
        prompt_mode = getattr(self.config, 'default_prompt_mode', "prompt_layout_all_en")
        bbox = getattr(self.config, 'default_bbox', None)
        fitz_preprocess = getattr(self.config, 'default_fitz_preprocess', False)

        # Check if file type is supported
        base_filename, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lower()
        supported_extensions = self.get_supported_extensions()

        if file_ext not in supported_extensions:
            error_msg = f"File extension '{file_ext}' not supported. Supported extensions: {supported_extensions}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get output directory from environment variable
        output_dir = os.getenv('DOTSOCR_OUTPUT_DIR', './dotsorc/output')
        output_dir = os.path.abspath(output_dir)
        save_dir = os.path.join(output_dir, base_filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self._parse_pdf(file_data, base_filename, save_dir, prompt_mode, **kwargs)
        elif file_ext in image_extensions:
            results = self._parse_image(
                file_data, base_filename, save_dir, prompt_mode,
                bbox=bbox, fitz_preprocess=fitz_preprocess, **kwargs
            )

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
        prompt_mode: str,
        bbox: Optional[Any] = None,
        fitz_preprocess: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Parse a single image file from binary data"""
        origin_image = Image.open(io.BytesIO(file_data))
        result = self._parse_single_image(
            origin_image, prompt_mode, save_dir, filename,
            source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        result['filename'] = filename
        return [result]

    def _parse_pdf(
        self,
        file_data: bytes,
        filename: str,
        save_dir: str,
        prompt_mode: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Parse a PDF file from binary data"""

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
                "prompt_mode": prompt_mode,
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
        prompt_mode: str,
        save_dir: str,
        save_name: str,
        source: str = "image",
        page_idx: int = 0,
        bbox: Optional[Any] = None,
        fitz_preprocess: bool = False,
    ) -> Dict[str, Any]:
        """Parse a single image and return result"""

        min_pixels = getattr(self.config, 'min_pixels', None)
        max_pixels = getattr(self.config, 'max_pixels', None)

        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None:
            assert min_pixels >= MIN_PIXELS
        if max_pixels is not None:
            assert max_pixels <= MAX_PIXELS

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=getattr(self.config, 'dpi', 200))
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)

        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self._get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # Use LLM service for inference
        response = self.llm_service.infer(image, prompt)

        result = {
            'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }

        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"

        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            if filtered and prompt_mode != 'prompt_layout_only_en':
                # json_file_path = os.path.join(save_dir, f"{save_name}.json")
                # with open(json_file_path, 'w', encoding="utf-8") as w:
                #     json.dump(response, w, ensure_ascii=False, indent=4)

                # Don't save image layout - not needed
                # image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                # origin_image.save(image_layout_path)
                # result.update({
                #     'layout_info_path': json_file_path,
                #     # 'layout_image_path': image_layout_path,
                # })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    # Clean base64 images from markdown content
                    cleaned_cells = clean_base64_images(cells)
                    md_file.write(cleaned_cells)
                result.update({
                    'md_content_path': md_file_path,
                    'filtered': True
                })
            else:
                # Don't draw layout on image - not needed
                # try:
                #     image_with_layout = draw_layout_on_image(origin_image, cells)
                # except Exception as e:
                #     logger.info(f"Error drawing layout on image: {e}")
                #     image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False, indent=4)

                # Don't save image layout - not needed
                # image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                # image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    # 'layout_image_path': image_layout_path,
                })

                if prompt_mode != "prompt_layout_only_en":
                    # Only generate one markdown file (with page headers/footers)
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    # Don't generate _nohf.md - not needed
                    # md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)

                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)

                    # Don't save _nohf.md - not needed
                    # md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    # with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                    #     md_file.write(md_content_no_hf)

                    result.update({
                        'md_content_path': md_file_path,
                        # 'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            # Don't save image layout - not needed
            # image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            # origin_image.save(image_layout_path)
            # result.update({
            #     'layout_image_path': image_layout_path,
            # })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                # Clean base64 images from markdown content
                cleaned_md_content = clean_base64_images(md_content)
                md_file.write(cleaned_md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result

    def _get_prompt(self, prompt_mode: str, bbox: Optional[Any] = None, origin_image: Optional[Image.Image] = None,
                   image: Optional[Image.Image] = None, min_pixels: Optional[int] = None,
                   max_pixels: Optional[int] = None) -> str:
        """Get prompt for specific mode"""

        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(
                origin_image, bboxes,
                input_width=image.width,
                input_height=image.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )[0]
            prompt = prompt + str(bbox)
        return prompt