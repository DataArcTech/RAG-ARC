import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from .base import ParserBase

from .dots_ocr_src.utils.image_utils import PILimage_to_base64, get_image_by_fitz_doc, fetch_image, smart_resize
from .dots_ocr_src.utils.prompts import dict_promptmode_to_prompt
from .dots_ocr_src.utils.layout_utils import pre_process_bboxes, post_process_output, draw_layout_on_image
from .dots_ocr_src.utils.consts import MIN_PIXELS, MAX_PIXELS, image_extensions
from .dots_ocr_src.utils.format_transformer import layoutjson2md
from .dots_ocr_src.utils.doc_utils import load_images_from_pdf

logger = logging.getLogger(__name__)


@dataclass
class DotsOCRParser(ParserBase):
    """
    DotsOCR-based document parser implementation for advanced OCR and layout analysis.

    This class provides a complete document parsing solution using the DotsOCR model,
    supporting both local HuggingFace inference and remote VLLM server deployment.
    Specializes in multilingual document layout parsing with high accuracy OCR capabilities.

    Key features:
    - Dual inference modes: HuggingFace (local) and VLLM (server-based)
    - Multi-format support: PDF, JPG, JPEG, PNG
    - Advanced layout analysis with bounding box detection
    - Multiple prompt modes for different parsing tasks
    - Multi-threaded PDF processing for performance
    - Structured output formats: JSON, Markdown, image annotations

    Main parameters:
        config (AbstractConfig): Configuration object containing model settings, server info, etc.
        model: Loaded HuggingFace model (when use_hf=True)
        processor: HuggingFace processor for tokenization
        inference_with_vllm: VLLM inference function (when use_hf=False)

    Core methods:
        - load_hf_model/load_vllm_model: Initialize inference engines
        - parse_file: Main entry point for any supported file
        - parse_image/parse_pdf: Specialized parsing for different formats
        - get_prompt: Generate prompts for different parsing modes

    Parsing modes:
        - prompt_layout_all_en: Full layout + text extraction
        - prompt_layout_only_en: Layout detection only
        - prompt_grounding_ocr: Targeted OCR with bounding boxes

    Performance considerations:
        - HuggingFace mode: Single-threaded, local GPU required
        - VLLM mode: Multi-threaded, external server required
        - PDF processing: Configurable thread count for parallel page processing
        - Image preprocessing: DPI and pixel constraints for optimal results

    Typical usage:
        >>> config = DotsOCRConfig(use_hf=True, dpi=200)
        >>> parser = DotsOCRParser(config)
        >>> results = parser.parse_file("document.pdf")

    Attributes:
        config: Parser configuration
        model: HuggingFace model instance (if loaded)
        processor: HuggingFace processor instance (if loaded)
        inference_with_vllm: VLLM inference function (if loaded)
    """

    def __init__(self, config):
        """Initialize DotsOCR parser and load model/client based on config"""
        super().__init__(config)

        # Initialize based on inference mode
        if getattr(self.config, 'use_hf', True):
            logger.info("Initializing HuggingFace model for local inference")
            self._load_hf_model()
        else:
            logger.info("Initializing VLLM client for server-based inference")
            self._load_vllm_client()
    

    def _load_hf_model(self):
        """Load HuggingFace model for local inference"""
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from qwen_vl_utils import process_vision_info

            model_path = "/home/yangcehao/doc_analysis/dots.ocr/weights/DotsOCR"
            device = getattr(self.config, 'device', 'auto')  # Default to 'auto'

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            self.process_vision_info = process_vision_info
            
            logger.info(f"HuggingFace model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise
    
    def _load_vllm_client(self):
        """Initialize VLLM client connection"""
        
        try:
            from openai import OpenAI
            import os
            
            # Get base_url and api_key from config
            base_url = getattr(self.config, 'base_url', "http://localhost:8000/v1")
            api_key = getattr(self.config, 'api_key', "sk-xxx")
            
            self.vllm_client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"VLLM client initialized - connecting to {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize VLLM client: {str(e)} Make sure you launch vllm server using vllm_launch.py")
            raise
    
    def _inference_with_hf(self, image, prompt):
        """Internal HF inference method"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(getattr(self.config, 'device', 'cuda'))

        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def _inference_with_vllm(self, image, prompt):
        """Internal VLLM inference method"""

        import requests

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
            ],
        }]

        try:
            response = self.vllm_client.chat.completions.create(
                messages=messages,
                model=getattr(self.config, 'model_name', 'model'),
                max_completion_tokens=getattr(self.config, 'max_completion_tokens', 16384),
                temperature=getattr(self.config, 'temperature', 0.1),
                top_p=getattr(self.config, 'top_p', 1.0)
            )
            return response.choices[0].message.content
        except requests.exceptions.RequestException as e:
            print(f"VLLM request error: {e}")
            return None

    def _get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
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

    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
    ):
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
        
        if getattr(self.config, 'use_hf', True):
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        
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
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False, indent=4)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path,
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False, indent=4)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                
                if prompt_mode != "prompt_layout_only_en":
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
                    
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result

    def parse_image(
        self, 
        input_path: str, 
        filename: str, 
        save_dir: str, 
        prompt_mode="prompt_layout_all_en",
        bbox=None, 
        fitz_preprocess=False,
        **kwargs
    ) -> List[dict]:
        """Parse a single image file"""
        
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(
            origin_image, prompt_mode, save_dir, filename, 
            source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        result['file_path'] = input_path
        return [result]

    def parse_pdf(
        self, 
        input_path: str, 
        filename: str, 
        save_dir: str, 
        prompt_mode="prompt_layout_all_en",
        **kwargs
    ) -> List[dict]:
        """Parse a PDF file"""
        
        print(f"Loading PDF: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=getattr(self.config, 'dpi', 200))
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

        use_hf = getattr(self.config, 'use_hf', True)
        num_threads = 1 if use_hf else min(total_pages, getattr(self.config, 'num_threads', 4))
        print(f"Parsing PDF with {total_pages} pages using {num_threads} threads...")

        results = []
        with ThreadPool(num_threads) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for result in results:
            result['file_path'] = input_path
        return results

    def parse_file(
        self, 
        input_path: str,
        output_dir: Optional[str] = None,
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
        **kwargs
    ) -> List[dict]:
        """Parse a file (PDF or image)"""
        
        # Check if file type is supported
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        file_ext = file_ext.lower()
        supported_extensions = self.get_supported_extensions()

        if file_ext not in supported_extensions:
            error_msg = f"File extension '{file_ext}' not supported. Supported extensions: {supported_extensions}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        output_dir = output_dir or getattr(self.config, 'output_dir', './test_output/dots_ocr_results')
        output_dir = os.path.abspath(output_dir)
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, save_dir, prompt_mode, **kwargs)
        elif file_ext in image_extensions:
            results = self.parse_image(
                input_path, filename, save_dir, prompt_mode,
                bbox=bbox, fitz_preprocess=fitz_preprocess, **kwargs
            )

        print(f"Parsing finished, results saved to {save_dir}")

        with open(os.path.join(output_dir, f"{filename}.jsonl"), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return ['.pdf', '.jpg', '.jpeg', '.png']