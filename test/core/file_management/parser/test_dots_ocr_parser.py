"""
Test for DotsOCR Parser (Core Layer) - testing with real PDF documents and images
Separated by loading method: HuggingFace vs VLLM
"""

import asyncio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from config.core.file_management.parser.dots_ocr import DotsOCRParserConfig
from config.encapsulation.llm.parse.dots_ocr import DotsOCRConfig


def get_test_files():
    """Get available test files for parsing tests"""
    sample_image_paths = [
        "./test_data/test_png.png",
    ]

    sample_pdf_paths = [
        "./test/test_pdf.pdf",
    ]

    return sample_image_paths, sample_pdf_paths


def test_basic_functionality(parser, loading_method_name):
    """Test basic parser functionality (common for both loading methods)"""
    print(f"\n=== Testing {loading_method_name} Basic Functionality ===")

    # 1. Test build
    print("\n--- Test 1: build ---")
    print(f"  DotsOCR Parser built successfully")
    print(f"  LLM Service loading method: {parser.llm_service.loading_method}")
    print(f"  LLM Service device: {getattr(parser.llm_service.config, 'device', 'auto')}")
    print(f"  Parser DPI setting: {getattr(parser.config, 'dpi', 200)}")
    print(f"  Parser num_threads: {getattr(parser.config, 'num_threads', 1)}")
    print(f"  Output directory from env: {os.getenv('DOTSOCR_OUTPUT_DIR', './dotsorc/output')}")

    # 2. Test get_supported_extensions
    print("\n--- Test 2: get_supported_extensions ---")
    extensions = parser.get_supported_extensions()
    print(f"  Supported extensions: {extensions}")
    print(f"  Extension count: {len(extensions)}")

    # 3. Test LLM service info
    print("\n--- Test 3: llm_service_info ---")
    try:
        llm_info = parser.llm_service.get_model_info()
        print(f"  LLM Service Info:")
        for key, value in llm_info.items():
            print(f"    {key}: {value}")
    except Exception as e:
        print(f"  Failed to get LLM service info: {e}")


def test_file_parsing(parser, loading_method_name):
    """Test file parsing functionality (common for both loading methods)"""
    print(f"\n=== Testing {loading_method_name} File Parsing ===")

    sample_image_paths, sample_pdf_paths = get_test_files()

    # Test image parsing
    print("\n--- Test 1: parse_image ---")
    image_found = False
    for image_path in sample_image_paths:
        if os.path.exists(image_path):
            print(f"  Found sample image: {image_path}")
            try:
                # Read file as binary data
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                filename = os.path.basename(image_path)

                results = asyncio.run(parser.parse_file(image_data, filename))
                print(f"  Parsed image successfully")
                print(f"  Results count: {len(results)}")

                for i, result in enumerate(results):
                    print(f"    Result {i+1}:")
                    print(f"      Filename: {result.get('filename')}")
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

    # Test PDF parsing
    print("\n--- Test 2: parse_pdf ---")
    pdf_found = False
    for pdf_path in sample_pdf_paths:
        if os.path.exists(pdf_path):
            print(f"  Found sample PDF: {pdf_path}")
            try:
                # Read file as binary data
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                filename = os.path.basename(pdf_path)

                results = asyncio.run(parser.parse_file(pdf_data, filename))
                print(f"  Parsed PDF successfully")
                print(f"  Results count (pages): {len(results)}")

                # Show first few pages
                for i, result in enumerate(results[:3]):  # Show first 3 pages
                    print(f"    Page {result.get('page_no', i)}:")
                    print(f"      Filename: {result.get('filename')}")
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


def test_error_handling(parser, loading_method_name):
    """Test error handling (common for both loading methods)"""
    print(f"\n=== Testing {loading_method_name} Error Handling ===")

    # Test unsupported file extension
    try:
        # Create fake binary data for unsupported file
        fake_data = b"fake content"
        asyncio.run(parser.parse_file(fake_data, "nonexistent.txt"))
        print(f"  ERROR: Should have failed with unsupported extension")
    except ValueError as e:
        print(f"  Correctly caught unsupported extension: {e}")
    except Exception as e:
        print(f"  Unexpected error type: {e}")

    # Test with invalid data
    try:
        # Create fake binary data for PDF
        fake_pdf_data = b"fake pdf content"
        asyncio.run(parser.parse_file(fake_pdf_data, "fake.pdf"))
        print(f"  ERROR: Should have failed with invalid PDF data")
    except Exception as e:
        print(f"  Correctly caught data error: {type(e).__name__}: {e}")


def test_huggingface_loading_method():
    """Test DotsOCR Parser with HuggingFace loading method"""
    print("\n" + "="*80)
    print("TESTING HUGGINGFACE LOADING METHOD")
    print("="*80)

    try:
        # Create HuggingFace config
        hf_service_config = DotsOCRConfig(
            loading_method="huggingface",
            use_china_mirror=True,
            cache_folder="./models/dots_ocr",
            use_snapshot_download=True,
            device="cuda:1"
        )

        hf_parser_config = DotsOCRParserConfig(
            dots_ocr=hf_service_config
        )

        hf_parser = hf_parser_config.build()

        # Run all tests
        test_basic_functionality(hf_parser, "HuggingFace")
        test_file_parsing(hf_parser, "HuggingFace")
        test_error_handling(hf_parser, "HuggingFace")

        print(f"\n✅ HuggingFace loading method tests completed successfully!")

    except Exception as e:
        print(f"\n❌ HuggingFace loading method test failed: {e}")
        import traceback
        traceback.print_exc()


def test_vllm_loading_method():
    """Test DotsOCR Parser with VLLM loading method"""
    print("\n" + "="*80)
    print("TESTING VLLM LOADING METHOD")
    print("="*80)

    try:
        # Create VLLM config
        vllm_service_config = DotsOCRConfig(
            loading_method="vllm"
        )

        vllm_parser_config = DotsOCRParserConfig(
            dots_ocr=vllm_service_config,
            num_threads=2  # Can use more threads with VLLM
        )

        vllm_parser = vllm_parser_config.build()

        # Run all tests
        test_basic_functionality(vllm_parser, "VLLM")
        test_file_parsing(vllm_parser, "VLLM")
        test_error_handling(vllm_parser, "VLLM")

        print(f"\n✅ VLLM loading method tests completed successfully!")

    except Exception as e:
        print(f"\n❌ VLLM loading method test failed (server not available or error): {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function - runs both loading method tests"""
    print("Testing DotsOCR Parser (Core Layer) - Separated by Loading Method")
    print("="*80)

    # Test HuggingFace loading method
    test_huggingface_loading_method()

    # Test VLLM loading method
    test_vllm_loading_method()

    print("\n" + "="*80)
    print("ALL DOTSOCR PARSER TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()