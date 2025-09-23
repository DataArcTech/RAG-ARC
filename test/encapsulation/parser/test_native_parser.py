"""
Test for Native Parser - testing all supported document formats
"""

import os
from typing import Literal

from framework.config import AbstractConfig
from encapsulation.parser.native import NativeParser


class NativeParserConfig(AbstractConfig):
    """Configuration for Native Parser testing"""
    type: Literal["native"] = "native"

    def build(self) -> NativeParser:
        return NativeParser(self)


def main():
    print("Testing Native Parser - Multi-format Document Processing")

    config = NativeParserConfig()

    try:
        print("=== Testing Native Parser Interface ===")

        # 1. Test build
        print("\n--- Test 1: build ---")
        parser = config.build()
        print(f"  Native Parser built from config")
        print(f"  Output directory default: {getattr(parser.config, 'output_dir', 'output')}")

        # 2. Test get_supported_extensions
        print("\n--- Test 2: get_supported_extensions ---")
        extensions = parser.get_supported_extensions()
        print(f"  Supported extensions: {extensions}")
        print(f"  Extension count: {len(extensions)}")

        # 3. Test with sample HTML file (if exists)
        print("\n--- Test 3: parse_html ---")
        sample_html_paths = [
            "./test_data/test_parser.html",
        ]

        html_found = False
        for html_path in sample_html_paths:
            if os.path.exists(html_path):
                print(f"  Found sample HTML: {html_path}")
                try:
                    results = parser.parse_file(html_path)
                    print(f"  Parsed HTML successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      File path: {result.get('file_path')}")
                        if 'output_paths' in result:
                            print(f"      Output files: {len(result['output_paths'])}")
                        if 'metadata' in result:
                            print(f"      Metadata keys: {list(result['metadata'].keys())}")

                    html_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse HTML {html_path}: {e}")

        if not html_found:
            print(f"  No sample HTML found in test paths: {sample_html_paths}")
            print(f"  Creating test HTML file...")
            test_html_path = "./test_output/test_sample.html"
            os.makedirs(os.path.dirname(test_html_path), exist_ok=True)

            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>Test Document</title></head>
            <body>
                <h1>Test HTML Document</h1>
                <p>This is a test paragraph.</p>
                <table>
                    <tr><th>Column 1</th><th>Column 2</th></tr>
                    <tr><td>Data 1</td><td>Data 2</td></tr>
                </table>
            </body>
            </html>
            """

            with open(test_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            try:
                results = parser.parse_file(test_html_path)
                print(f"  Created and parsed test HTML successfully")
                print(f"  Results count: {len(results)}")
            except Exception as e:
                print(f"  Failed to parse created HTML: {e}")

        # 4. Test with sample DOCX file (if exists)
        print("\n--- Test 4: parse_docx ---")
        sample_docx_paths = [
            "./test_data/test_parser.docx",
        ]

        docx_found = False
        for docx_path in sample_docx_paths:
            if os.path.exists(docx_path):
                print(f"  Found sample DOCX: {docx_path}")
                try:
                    results = parser.parse_file(docx_path)
                    print(f"  Parsed DOCX successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      File path: {result.get('file_path')}")
                        if 'output_paths' in result:
                            print(f"      Output files: {len(result['output_paths'])}")

                    docx_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse DOCX {docx_path}: {e}")

        if not docx_found:
            print(f"  No sample DOCX found in test paths: {sample_docx_paths}")
            print(f"  Skipping DOCX parsing test")

        # 5. Test with sample Excel file (if exists)
        print("\n--- Test 5: parse_excel ---")
        sample_excel_paths = [
            "./test_data/test_parser.xlsx",
            "./test_data/test_parser.xls",
            "./test_data/test_parser.csv",
        ]

        excel_found = False
        for excel_path in sample_excel_paths:
            if os.path.exists(excel_path):
                print(f"  Found sample Excel: {excel_path}")
                try:
                    results = parser.parse_file(excel_path)
                    print(f"  Parsed Excel successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      File path: {result.get('file_path')}")

                    excel_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse Excel {excel_path}: {e}")

        if not excel_found:
            print(f"  No sample Excel found in test paths: {sample_excel_paths}")
            print(f"  Creating test CSV file...")
            test_csv_path = "./test_output/test_sample.csv"
            os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)

            csv_content = "Name,Age,City\nJohn,30,New York\nJane,25,London\nBob,35,Paris"

            with open(test_csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            try:
                results = parser.parse_file(test_csv_path)
                print(f"  Created and parsed test CSV successfully")
                print(f"  Results count: {len(results)}")
            except Exception as e:
                print(f"  Failed to parse created CSV: {e}")

        # 6. Test with sample PowerPoint file (if exists)
        print("\n--- Test 6: parse_pptx ---")
        sample_pptx_paths = [
            "./test_data/test_parser.pptx",
        ]

        pptx_found = False
        for pptx_path in sample_pptx_paths:
            if os.path.exists(pptx_path):
                print(f"  Found sample PPTX: {pptx_path}")
                try:
                    results = parser.parse_file(pptx_path)
                    print(f"  Parsed PPTX successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      File path: {result.get('file_path')}")
                        if 'output_paths' in result:
                            print(f"      Output files: {len(result['output_paths'])}")

                    pptx_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse PPTX {pptx_path}: {e}")

        if not pptx_found:
            print(f"  No sample PPTX found in test paths: {sample_pptx_paths}")
            print(f"  Skipping PPTX parsing test")

        # 7. Test URL detection
        print("\n--- Test 7: url_detection ---")
        test_urls = [
            "https://example.com/page.html",
            "http://example.com",
            "https://example.com/file.docx"
        ]

        for url in test_urls:
            is_url = parser._is_url(url)
            if is_url:
                is_html = parser._is_html_url(url)
                print(f"  '{url}' -> URL: {is_url}, HTML: {is_html}")
            else:
                print(f"  '{url}' -> Not a URL")

        # 8. Test error handling
        print("\n--- Test 8: error_handling ---")

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
            parser.parse_file("nonexistent.docx")
            print(f"  ERROR: Should have failed with non-existent file")
        except FileNotFoundError as e:
            print(f"  Correctly caught file not found: {e}")
        except Exception as e:
            print(f"  Unexpected error type: {e}")

        # Test unsupported URL
        try:
            parser.parse_file("https://example.com/file.docx")
            print(f"  ERROR: Should have failed with unsupported URL")
        except ValueError as e:
            print(f"  Correctly caught unsupported URL: {e}")
        except Exception as e:
            print(f"  Unexpected error type: {e}")

        # 9. Test output directory creation
        print("\n--- Test 9: output_directory_creation ---")
        custom_output = "./test_output/custom_native_output"
        print(f"  Testing custom output directory: {custom_output}")

        if os.path.exists("./test_output/test_sample.html"):
            try:
                results = parser.parse_file(
                    "./test_output/test_sample.html",
                    output_dir=custom_output
                )
                print(f"  Output directory created and used successfully")
                print(f"  Directory exists: {os.path.exists(custom_output)}")
            except Exception as e:
                print(f"  Failed to create custom output: {e}")
        else:
            print(f"  No test file available")

        print("\n All Native Parser tests completed!")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()