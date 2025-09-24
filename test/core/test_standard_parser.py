"""
Test for StandardParser - testing the core parser interface methods
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.file_management.parser.standard import StandardParser
from config.core.file_management.parser.standard_parser_config import StandardParserConfig
from config.encapsulation.parser.dots_ocr import DotsOCRConfig
from config.encapsulation.parser.native import NativeParserConfig


def main():
    print("Testing StandardParser - Core Parser Interface Methods")

    try:
        """Test the StandardParser interface methods"""
        print("=== Testing StandardParser Interface ===")

        # Test data paths - real files from test_data directory
        test_data_paths = {
            # 'pdf': "./test_data/test_pdf.pdf",
            # 'png': "./test_data/test_png.png",
            'docx': "./test_data/test_docx.docx",
            'xlsx': "./test_data/test_xlsx.xlsx",
            'html': "./test_data/test_html.html"
        }

        # Load real test files
        test_files = {}
        for file_type, file_path in test_data_paths.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    test_files[file_type] = {
                        'filename': os.path.basename(file_path),
                        'content': file_content,
                        'path': file_path
                    }
                    print(f"  Loaded test file: {file_path} ({len(file_content)} bytes)")
                except Exception as e:
                    print(f"  Failed to load {file_path}: {e}")
            else:
                print(f"  Test file not found: {file_path}")

        if not test_files:
            print("  No test files found in test_data directory")
            print("  Available files in current directory:")
            if os.path.exists("./"):
                for f in os.listdir("./"):
                    if os.path.isfile(f):
                        print(f"    {f}")
            return

        # Build all parser configurations once at the top
        print("\n=== Building Parser Configurations ===")

        # 1. Build parser without specific configuration (auto-select mode)
        print("\n--- Building auto-select parser ---")
        config_no_parser = StandardParserConfig()
        parser_no_config = config_no_parser.build()
        print(f"  StandardParser built for auto-selection")
        print(f"  Parser instance created: {parser_no_config is not None}")
        print(f"  Internal parser is None: {parser_no_config.parser is None}")

        # 2. Build parser with Native configuration
        print("\n--- Building Native parser ---")
        native_config = NativeParserConfig()
        config_with_native = StandardParserConfig(parser=native_config)
        parser_with_native = config_with_native.build()
        print(f"  StandardParser built with Native parser config")
        print(f"  Parser instance created: {parser_with_native is not None}")
        print(f"  Internal parser configured: {parser_with_native.parser is not None}")
        print(f"  Internal parser type: {type(parser_with_native.parser).__name__}")

        # 3. Build parser with DotsOCR configuration
        print("\n--- Building DotsOCR parser ---")
        parser_with_dots = None
        try:
            dots_config = DotsOCRConfig() 
            config_with_dots = StandardParserConfig(parser=dots_config)
            parser_with_dots = config_with_dots.build()
            print(f"  StandardParser built with DotsOCR parser config")
            print(f"  Parser instance created: {parser_with_dots is not None}")
            print(f"  Internal parser configured: {parser_with_dots.parser is not None}")
            print(f"  Internal parser type: {type(parser_with_dots.parser).__name__}")
            print(f"  Using VLLM mode: {not getattr(parser_with_dots.parser.config, 'use_hf', True)}")
        except Exception as e:
            print(f"  Failed to build DotsOCR parser: {e}")
            print("  This may be expected if VLLM server dependencies are missing")

        # Now run tests using the pre-built parsers
        print("\n=== Running Tests with Pre-built Parsers ===")

        # Test 1: Native parser with real files
        print("\n--- Test 1: Native parser with real files ---")
        try:
            # Test with real DOCX file if available
            if 'docx' in test_files:
                print(f"  Testing with real DOCX file: {test_files['docx']['filename']}")
                try:
                    results = parser_with_native.parse_file(
                        file_data=test_files['docx']['content'],
                        filename=test_files['docx']['filename']
                    )
                    print(f"    Parsed DOCX successfully: {len(results)} results")
                    for i, result in enumerate(results):
                        print(f"      Result {i+1}: {result.get('content_type', 'unknown')} - {result.get('filename')}")
                        if 'output_paths' in result:
                            print(f"        Output files: {list(result['output_paths'].keys())}")
                        if 'metadata' in result:
                            print(f"        Metadata: {result['metadata']}")
                except Exception as e:
                    print(f"    Failed to parse DOCX: {e}")

            # Test with real HTML file if available
            elif 'html' in test_files:
                print(f"  Testing with real HTML file: {test_files['html']['filename']}")
                try:
                    results = parser_with_native.parse_file(
                        file_data=test_files['html']['content'],
                        filename=test_files['html']['filename']
                    )
                    print(f"    Parsed HTML successfully: {len(results)} results")
                    for i, result in enumerate(results):
                        print(f"      Result {i+1}: {result.get('content_type', 'unknown')} - {result.get('filename')}")
                except Exception as e:
                    print(f"    Failed to parse HTML: {e}")
            else:
                print("  No native parser compatible files found (docx, xlsx, html)")

        except Exception as e:
            print(f"  Failed with Native parser: {e}")

        # Test 2: DotsOCR parser with real files (if available)
        print("\n--- Test 2: DotsOCR parser with real files ---")
        if parser_with_dots is not None:
            try:
                # Test with real PDF file if available
                if 'pdf' in test_files:
                    print(f"  Testing with real PDF file: {test_files['pdf']['filename']}")
                    try:
                        results = parser_with_dots.parse_file(
                            file_data=test_files['pdf']['content'],
                            filename=test_files['pdf']['filename'],
                            prompt_mode="prompt_layout_all_en"
                        )
                        print(f"    Parsed PDF successfully: {len(results)} pages")
                        for i, result in enumerate(results[:2]):  # Show first 2 pages
                            print(f"      Page {result.get('page_no', i)}: {result.get('filename')}")
                            print(f"        Size: {result.get('input_width')}x{result.get('input_height')}")
                            if 'md_content_path' in result:
                                print(f"        Markdown: {result['md_content_path']}")
                    except Exception as e:
                        print(f"    Failed to parse PDF (may need VLLM server): {e}")

                # Test with real PNG file if available
                elif 'png' in test_files:
                    print(f"  Testing with real PNG file: {test_files['png']['filename']}")
                    try:
                        results = parser_with_dots.parse_file(
                            file_data=test_files['png']['content'],
                            filename=test_files['png']['filename'],
                            prompt_mode="prompt_layout_all_en"
                        )
                        print(f"    Parsed PNG successfully: {len(results)} results")
                        for i, result in enumerate(results):
                            print(f"      Result {i+1}: {result.get('filename')}")
                            print(f"        Size: {result.get('input_width')}x{result.get('input_height')}")
                    except Exception as e:
                        print(f"    Failed to parse PNG (may need VLLM server): {e}")
                else:
                    print("  No DotsOCR compatible files found (pdf, png, jpg)")

            except Exception as e:
                print(f"  Failed with DotsOCR parser: {e}")
        else:
            print("  DotsOCR parser not available (failed to build)")

        # Test 3: Auto-selection with real files
        print("\n--- Test 3: Auto-selection with real files ---")

        if test_files:
            # Use the pre-built auto-select parser
            for file_type, file_info in test_files.items():
                print(f"  Testing auto-selection for {file_type.upper()}: {file_info['filename']}")

                try:
                    results = parser_no_config.parse_file(
                        file_data=file_info['content'],
                        filename=file_info['filename']
                    )
                    print(f"    Auto-selection successful: {len(results)} results")
                    print(f"    First result filename: {results[0].get('filename', 'N/A')}")

                    # Show which parser was likely used based on file extension
                    ext = os.path.splitext(file_info['filename'])[1].lower()
                    if ext in ['.pdf', '.png', '.jpg', '.jpeg']:
                        print(f"    Expected parser: DotsOCR (for {ext})")
                    elif ext in ['.docx', '.xlsx', '.html']:
                        print(f"    Expected parser: Native (for {ext})")

                except Exception as e:
                    print(f"    Auto-selection failed: {e}")
                    print(f"    This may be expected if required parser dependencies are missing")
        else:
            print("  No test files available for auto-selection testing")

        # Test 4: Unsupported file type error handling
        print("\n--- Test 4: Unsupported file type handling ---")

        try:
            fake_data = b"fake content for unsupported file"
            results = parser_no_config.parse_file(
                file_data=fake_data,
                filename="test.unsupported"
            )
            print(f"  ERROR: Should have raised exception for unsupported file type")
        except ValueError as e:
            print(f"  Correctly raised ValueError: {str(e)}")
        except Exception as e:
            print(f"  Raised exception: {type(e).__name__}: {str(e)}")

        # 5. Test error propagation from underlying parsers
        print("\n--- Test 5: error propagation ---")

        # Test with corrupted file data
        if test_files:
            first_file = list(test_files.values())[0]
            print(f"  Testing with corrupted version of: {first_file['filename']}")

            try:
                parser_auto_select = StandardParser(config_no_parser)
                corrupted_data = b"corrupted file content that should fail parsing"

                results = parser_auto_select.parse_file(
                    file_data=corrupted_data,
                    filename=first_file['filename']  # Keep original filename for extension detection
                )
                print(f"    WARNING: Parsing succeeded with corrupted data (unexpected)")
                print(f"    Results: {len(results)} items")

            except Exception as e:
                print(f"    Correctly propagated parsing error: {type(e).__name__}: {str(e)[:100]}...")

        print(f"\n=== Test Summary ===")
        print(f"Total test files loaded: {len(test_files)}")
        if test_files:
            print(f"Test files used:")
            for file_type, file_info in test_files.items():
                print(f"  {file_type.upper()}: {file_info['filename']} ({len(file_info['content'])} bytes)")

        print(f"\nStandardParser testing completed!")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()