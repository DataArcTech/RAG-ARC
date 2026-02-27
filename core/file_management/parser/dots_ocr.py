import os
import json
import logging
import io
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from PIL import Image
import fitz

from .base import AbstractParser
from framework.thread_pool import get_thread_pool

# Import DotsOCR utilities
from .dots_ocr_utils.image_utils import fetch_image, smart_resize, get_image_by_fitz_doc
from .dots_ocr_utils.prompts import dict_promptmode_to_prompt
from .dots_ocr_utils.layout_utils import pre_process_bboxes, post_process_output
# from .dots_ocr_utils.layout_utils import draw_layout_on_image  # Not needed - image output disabled
from .dots_ocr_utils.consts import MIN_PIXELS, MAX_PIXELS, image_extensions
from .dots_ocr_utils.format_transformer import layoutjson2md, clean_base64_images
from core.utils.llm_json import call_prompt_json_with_retry_sync

if TYPE_CHECKING:
    from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig

logger = logging.getLogger(__name__)


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
        import app_registration

        io_manager = app_registration.registrator.get_object("io_manager")
        if io_manager is None:
            raise RuntimeError("io_manager is required for DotsOCRParser")

        prompt_mode = getattr(self.config, "default_prompt_mode")
        bbox = getattr(self.config, "default_bbox")
        fitz_preprocess = bool(getattr(self.config, "default_fitz_preprocess"))

        # Check if file type is supported
        base_filename, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lower()
        supported_extensions = self.get_supported_extensions()

        if file_ext not in supported_extensions:
            error_msg = f"File extension '{file_ext}' not supported. Supported extensions: {supported_extensions}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        output_dir = getattr(self.config, "output_dir", None)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("DotsOCRParser requires config.output_dir (no implicit env defaults).")
        output_dir_virtual = str(output_dir).strip()
        if not output_dir_virtual.startswith("io://"):
            raise ValueError("DotsOCRParser config.output_dir must be an io:// virtual path")
        save_dir_virtual = f"{output_dir_virtual.rstrip('/')}/{base_filename}"

        # Run parsing in thread pool to avoid blocking the event loop
        # PDF parsing involves heavy I/O (LLM API calls) that can take a long time
        if file_ext == '.pdf':
            results = await get_thread_pool().run_blocking(
                self._parse_pdf,
                file_data,
                base_filename,
                prompt_mode,
                **kwargs
            )
        elif file_ext in image_extensions:
            results = await get_thread_pool().run_blocking(
                self._parse_image,
                file_data,
                base_filename,
                prompt_mode,
                bbox=bbox,
                fitz_preprocess=fitz_preprocess,
                **kwargs
            )

        logger.info("Parsing finished, results saved to %s", save_dir_virtual)

        persisted_results: List[Dict[str, Any]] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            artifact_name = str(item.get("artifact_name") or base_filename).strip() or base_filename
            md_content = item.get("md_content")
            layout_info = item.get("layout_info")

            if isinstance(md_content, str) and md_content.strip():
                md_path_virtual = f"{save_dir_virtual.rstrip('/')}/{artifact_name}.md"
                io_manager.put_text_path(md_path_virtual, text=md_content, content_type="text/markdown; charset=utf-8")
                item["md_content_path"] = md_path_virtual
                item["text"] = md_content
            if layout_info is not None:
                json_path_virtual = f"{save_dir_virtual.rstrip('/')}/{artifact_name}.json"
                io_manager.put_json_path(json_path_virtual, payload=layout_info)
                item["layout_info_path"] = json_path_virtual

            item.pop("md_content", None)
            item.pop("layout_info", None)
            item.pop("artifact_name", None)
            persisted_results.append(item)

        io_manager.put_text_path(
            f"{output_dir_virtual.rstrip('/')}/{base_filename}.jsonl",
            text="".join([json.dumps(row, ensure_ascii=False) + "\n" for row in persisted_results]),
            content_type="application/jsonl; charset=utf-8",
        )

        return persisted_results

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return ['.pdf', '.jpg', '.jpeg', '.png']

    def _parse_image(
        self,
        file_data: bytes,
        filename: str,
        prompt_mode: str,
        bbox: Optional[Any] = None,
        fitz_preprocess: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Parse a single image file from binary data"""
        origin_image = Image.open(io.BytesIO(file_data))
        result = self._parse_single_image(
            origin_image, prompt_mode, filename,
            source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        result['filename'] = filename
        return [result]

    def _parse_pdf(
        self,
        file_data: bytes,
        filename: str,
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
            dpi = int(getattr(self.config, "dpi"))
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
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        # Use single thread for stability
        num_threads = min(total_pages, int(getattr(self.config, "num_threads")))
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
        save_name: str,
        source: str = "image",
        page_idx: int = 0,
        bbox: Optional[Any] = None,
        fitz_preprocess: bool = False,
    ) -> Dict[str, Any]:
        """Parse a single image and return result"""

        min_pixels = getattr(self.config, 'min_pixels', None)
        max_pixels = getattr(self.config, 'max_pixels', None)

        if max_pixels is None:
            raise ValueError("DotsOCRParser config.max_pixels must be set (no implicit fallbacks).")

        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
        
        if min_pixels is not None:
            assert min_pixels >= MIN_PIXELS
        if max_pixels is not None:
            assert max_pixels <= MAX_PIXELS

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=int(getattr(self.config, "dpi")))
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)

        current_pixels = image.width * image.height
        if max_pixels and current_pixels > max_pixels:
            logger.warning(
                f"Image size ({image.width}x{image.height}, {current_pixels} pixels) exceeds max_pixels ({max_pixels}), "
                f"resizing to fit limit"
            )
            # Calculate new dimensions
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            from .dots_ocr_utils.consts import IMAGE_FACTOR
            new_width = (new_width // IMAGE_FACTOR) * IMAGE_FACTOR
            new_height = (new_height // IMAGE_FACTOR) * IMAGE_FACTOR
            if new_width < IMAGE_FACTOR:
                new_width = IMAGE_FACTOR
            if new_height < IMAGE_FACTOR:
                new_height = IMAGE_FACTOR
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_width}x{new_height} ({new_width * new_height} pixels)")

        input_height, input_width = smart_resize(
            image.height, 
            image.width,
            min_pixels=min_pixels or MIN_PIXELS,
            max_pixels=max_pixels
        )
        
        # Log image dimension information
        logger.debug(
            f"Processing image: original={origin_image.width}x{origin_image.height}, "
            f"resized={image.width}x{image.height}, "
            f"input_dimensions={input_width}x{input_height}, "
            f"max_pixels={max_pixels}"
        )
        
        prompt = self._get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # Use LLM service for inference.
        json_prompt_modes = {"prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr"}
        if prompt_mode in json_prompt_modes:
            payload = call_prompt_json_with_retry_sync(
                infer_once=lambda next_prompt: str(self.llm_service.infer(image, next_prompt) or ""),
                prompt=prompt,
                expected="list",
                return_raw=True,
            )
            if isinstance(payload, tuple):
                parsed_payload, response = payload
            else:
                parsed_payload, response = payload, ""
            if isinstance(parsed_payload, list):
                response = json.dumps(parsed_payload, ensure_ascii=False)
        else:
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

                result.update({
                    'artifact_name': save_name,
                    'md_content': clean_base64_images(cells),
                    'filtered': True
                })
            else:
                # Don't draw layout on image - not needed
                # try:
                #     image_with_layout = draw_layout_on_image(origin_image, cells)
                # except Exception as e:
                #     logger.info(f"Error drawing layout on image: {e}")
                #     image_with_layout = origin_image

                # Don't save image layout - not needed
                # image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                # image_with_layout.save(image_layout_path)
                result.update({
                    'artifact_name': save_name,
                    'layout_info': cells,
                    # 'layout_image_path': image_layout_path,
                })

                if prompt_mode != "prompt_layout_only_en":
                    # Only generate one markdown file (with page headers/footers)
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    # Don't generate _nohf.md - not needed
                    # md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)

                    # Don't save _nohf.md - not needed
                    # md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    # with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                    #     md_file.write(md_content_no_hf)

                    result.update({
                        'md_content': md_content,
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
            result.update({
                'artifact_name': save_name,
                'md_content': clean_base64_images(md_content),
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
