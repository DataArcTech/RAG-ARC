"""
Test for Native Parser - testing all supported document formats
"""

import os
import asyncio
from core.file_management.parser.native import NativeParserConfig


def main():
    print("Testing Native Parser - Multi-format Document Processing")

    config = NativeParserConfig()

    try:
        print("=== Testing Native Parser Interface ===")

        # 1. Test build
        print("\n--- Test 1: build ---")
        parser = config.build()
        print(f"  Native Parser built from config")
        print(f"  Output directory from env: {os.getenv('RAG_OUTPUT_DIR', './output/parsed_documents')}")

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
                    # Read file as binary data
                    with open(html_path, 'rb') as f:
                        html_data = f.read()
                    filename = os.path.basename(html_path)

                    results = asyncio.run(parser.parse_file(html_data, filename))
                    print(f"  Parsed HTML successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      Filename: {result.get('filename')}")
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
                # Convert string to bytes for testing
                html_data = html_content.encode('utf-8')
                results = asyncio.run(parser.parse_file(html_data, "test_sample.html"))
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
                    # Read file as binary data
                    with open(docx_path, 'rb') as f:
                        docx_data = f.read()
                    filename = os.path.basename(docx_path)

                    results = asyncio.run(parser.parse_file(docx_data, filename))
                    print(f"  Parsed DOCX successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      Filename: {result.get('filename')}")
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
                    # Read file as binary data
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()
                    filename = os.path.basename(excel_path)

                    results = asyncio.run(parser.parse_file(excel_data, filename))
                    print(f"  Parsed Excel successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      Filename: {result.get('filename')}")

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
                # Convert CSV string to bytes for testing
                csv_data = csv_content.encode('utf-8')
                results = asyncio.run(parser.parse_file(csv_data, "test_sample.csv"))
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
                    # Read file as binary data
                    with open(pptx_path, 'rb') as f:
                        pptx_data = f.read()
                    filename = os.path.basename(pptx_path)

                    results = asyncio.run(parser.parse_file(pptx_data, filename))
                    print(f"  Parsed PPTX successfully")
                    print(f"  Results count: {len(results)}")

                    for i, result in enumerate(results):
                        print(f"    Result {i+1}:")
                        print(f"      Content type: {result.get('content_type')}")
                        print(f"      Filename: {result.get('filename')}")
                        if 'output_paths' in result:
                            print(f"      Output files: {len(result['output_paths'])}")

                    pptx_found = True
                    break
                except Exception as e:
                    print(f"  Failed to parse PPTX {pptx_path}: {e}")

        if not pptx_found:
            print(f"  No sample PPTX found in test paths: {sample_pptx_paths}")
            print(f"  Skipping PPTX parsing test")

        # 7. Test error handling
        print("\n--- Test 7: error_handling ---")

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
            # Create fake binary data for DOCX
            fake_docx_data = b"fake docx content"
            asyncio.run(parser.parse_file(fake_docx_data, "fake.docx"))
            print(f"  ERROR: Should have failed with invalid DOCX data")
        except Exception as e:
            print(f"  Correctly caught data error: {type(e).__name__}: {e}")

        # 8. Test environment-based output directory
        print("\n--- Test 8: environment_output_directory ---")
        output_dir = os.getenv('RAG_OUTPUT_DIR', './output/parsed_documents')
        print(f"  Using output directory from environment: {output_dir}")

        # Test with the created HTML content
        try:
            html_data = html_content.encode('utf-8')
            results = asyncio.run(parser.parse_file(html_data, "env_test.html"))
            print(f"  Environment output directory used successfully")
            print(f"  Directory exists: {os.path.exists(output_dir)}")
        except Exception as e:
            print(f"  Failed with environment output: {e}")

        print("\n All Native Parser tests completed!")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()