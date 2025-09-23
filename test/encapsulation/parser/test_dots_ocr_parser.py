"""
Test for DotsOCR Parser - testing with real PDF documents and images
"""

import os
from typing import Literal

from framework.config import AbstractConfig
from encapsulation.parser.dots_ocr import DotsOCRParser


class DotsOCRConfig(AbstractConfig):
    """Configuration for DotsOCR Parser testing"""
    type: Literal["dots_ocr"] = "dots_ocr"
    use_hf: bool = False  # Use VLLM by default for testing
    device: str = "cuda:4"
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "sk-xxx"
    model_name: str = "model"

    def build(self) -> DotsOCRParser:
        return DotsOCRParser(self)


def main():
    print("Testing DotsOCR Parser - Real Document Processing")

    config = DotsOCRConfig()

    try:
        print("=== Testing DotsOCR Parser Interface ===")

        # 1. Test build
        print("\n--- Test 1: build ---")
        parser = config.build()
        print(f"  DotsOCR Parser built from config")
        print(f"  Use HuggingFace: {getattr(parser.config, 'use_hf', True)}")
        print(f"  Device: {getattr(parser.config, 'device', 'auto')}")
        print(f"  Base URL: {getattr(parser.config, 'base_url', 'http://localhost:8000/v1')}")
        print(f"  Model name: {getattr(parser.config, 'model_name', 'model')}")
        print(f"  Output directory: {getattr(parser.config, 'output_dir', './test_output/dots_ocr_results')}")
        print(f"  DPI setting: {getattr(parser.config, 'dpi', 200)}")

        # 2. Test get_supported_extensions
        print("\n--- Test 2: get_supported_extensions ---")
        extensions = parser.get_supported_extensions()
        print(f"  Supported extensions: {extensions}")
        print(f"  Extension count: {len(extensions)}")

        # 3. Test with sample image file (if exists)
        print("\n--- Test 3: parse_image ---")
        sample_image_paths = [
            "./test_data/sample.png",
        ]

        image_found = False
        for image_path in sample_image_paths:
            if os.path.exists(image_path):
                print(f"  Found sample image: {image_path}")
                try:
                    results = parser.parse_file(
                        image_path,
                        prompt_mode="prompt_layout_all_en"
                    )
                    print(f"  Parsed image successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      File path: {result.get('file_path')}")
                        print(f"      Page: {result.get('page_no', 0)}")
                        print(f"      Input size: {result.get('input_width')}x{result.get('input_height')}")
                        if 'md_content_path' in result:
                            print(f"      Markdown saved: {result['md_content_path']}")
                        if 'layout_info_path' in result:
                            print(f"      Layout JSON saved: {result['layout_info_path']}")
                        if 'layout_image_path' in result:
                            print(f"      Layout image saved: {result['layout_image_path']}")

                    image_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse image {image_path}: {e}")

        if not image_found:
            print(f"  No sample image found in test paths: {sample_image_paths}")
            print(f"  Skipping image parsing test")

        # 4. Test with sample PDF file (if exists)
        print("\n--- Test 4: parse_pdf ---")
        sample_pdf_paths = [
            "./test_data/parser_test_sample.pdf",
        ]

        pdf_found = False
        for pdf_path in sample_pdf_paths:
            if os.path.exists(pdf_path):
                print(f"  Found sample PDF: {pdf_path}")
                try:
                    results = parser.parse_file(
                        pdf_path,
                        prompt_mode="prompt_layout_all_en"
                    )
                    print(f"  Parsed PDF successfully")
                    print(f"  Results count (pages): {len(results)}")

                    # Show first few pages
                    for i, result in enumerate(results[:3]):  # Show first 3 pages
                        print(f"    Page {result.get('page_no', i)}:")
                        print(f"      File path: {result.get('file_path')}")
                        print(f"      Input size: {result.get('input_width')}x{result.get('input_height')}")
                        if 'md_content_path' in result:
                            print(f"      Markdown saved: {result['md_content_path']}")
                        if 'layout_info_path' in result:
                            print(f"      Layout JSON saved: {result['layout_info_path']}")

                    if len(results) > 3:
                        print(f"    ... and {len(results) - 3} more pages")

                    pdf_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse PDF {pdf_path}: {e}")

        if not pdf_found:
            print(f"  No sample PDF found in test paths: {sample_pdf_paths}")
            print(f"  Skipping PDF parsing test")

        # 5. Test different prompt modes
        print("\n--- Test 5: different_prompt_modes ---")
        prompt_modes = [
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_grounding_ocr"
        ]

        # Try with the first available image/PDF
        test_file = None
        for path in sample_image_paths + sample_pdf_paths:
            if os.path.exists(path):
                test_file = path
                break

        if test_file:
            print(f"  Testing prompt modes with: {test_file}")
            for mode in prompt_modes:
                try:
                    if mode == "prompt_grounding_ocr":
                        # Skip grounding OCR as it requires bbox parameter
                        print(f"    {mode}: Skipped (requires bbox parameter)")
                        continue

                    print(f"    Testing {mode}...")
                    results = parser.parse_file(
                        test_file,
                        prompt_mode=mode,
                        output_dir=f"./test_output/dots_ocr_{mode}"
                    )
                    print(f"      Success: {len(results)} results")
                except Exception as e:
                    print(f"      Failed with {mode}: {e}")
        else:
            print(f"  No test file available for prompt mode testing")

        # 6. Test error handling
        print("\n--- Test 6: error_handling ---")

        # Test unsupported file extension
        try:
            parser.parse_file("nonexistent.txt")
            print(f"  ERROR: Should have failed with unsupported extension")
        except ValueError as e:
            print(f"  Correctly caught unsupported extension: {e}")
        except Exception as e:
            print(f"  Unexpected error type: {e}")

        # Test non-existent file
        try:
            parser.parse_file("nonexistent.pdf")
            print(f"  ERROR: Should have failed with non-existent file")
        except Exception as e:
            print(f"  Correctly caught file error: {type(e).__name__}: {e}")

        # 7. Test output directory creation
        print("\n--- Test 7: output_directory_creation ---")
        custom_output = "./test_output/custom_dots_ocr_output"
        print(f"  Testing custom output directory: {custom_output}")

        if test_file:
            try:
                results = parser.parse_file(
                    test_file,
                    output_dir=custom_output
                )
                print(f"  Output directory created and used successfully")
                print(f"  Directory exists: {os.path.exists(custom_output)}")
            except Exception as e:
                print(f"  Failed to create custom output: {e}")
        else:
            print(f"  No test file available")

        print("\n All DotsOCR Parser tests completed!")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()