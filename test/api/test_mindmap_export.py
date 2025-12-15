"""Utility script to validate the mind map export endpoint.

This script authenticates with the API, optionally uploads a file,
and then polls the `/knowledge/mindmap/export` endpoint until a
successful response is returned or the retry limit is reached.

Example usage:

    python test/api/test_mindmap_export.py \
        --base-url http://localhost:8005 \
        --username test_user --password test_password \
        --file-id 44bc3b38-e209-4c34-acdb-17a105c06e80

    # Upload a file before exporting
    python test/api/test_mindmap_export.py \
        --base-url http://localhost:8005 \
        --username test_user --password test_password \
        --upload-file test/test.json \
        --retries 10 --wait-seconds 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def obtain_token(base_url: str, username: str, password: str) -> str:
    response = requests.post(
        f"{base_url}/auth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("Access token not found in auth response")
    return token


def upload_file(base_url: str, token: str, file_path: Path) -> str:
    with file_path.open("rb") as fp:
        files = {"file": (file_path.name, fp)}
        response = requests.post(
            f"{base_url}/knowledge",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def trigger_indexing(base_url: str, token: str, file_id: str) -> None:
    response = requests.post(
        f"{base_url}/knowledge/trigger_indexing",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"file_ids": [file_id]}),
        timeout=60,
    )
    response.raise_for_status()


def request_mindmap(
    base_url: str,
    token: str,
    file_id: str,
) -> requests.Response:
    return requests.post(
        f"{base_url}/knowledge/mindmap/export",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"file_id": file_id}),
        timeout=60,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mind map export endpoint tester")
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. http://localhost:8005")
    parser.add_argument("--username", required=True, help="Username for authentication")
    parser.add_argument("--password", required=True, help="Password for authentication")
    parser.add_argument("--file-id", help="Existing file ID to export mind map for")
    parser.add_argument("--upload-file", type=Path, help="Optional file to upload before exporting")
    parser.add_argument("--trigger-indexing", action="store_true", help="Trigger indexing after upload")
    parser.add_argument("--retries", type=int, default=5, help="Number of retries for mind map export (default: 5)")
    parser.add_argument("--wait-seconds", type=int, default=5, help="Wait time between retries in seconds (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose debug output")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    token = obtain_token(args.base_url, args.username, args.password)
    if args.verbose:
        print(f"Obtained token: {token[:10]}...")

    file_id: Optional[str] = args.file_id

    if args.upload_file:
        if not args.upload_file.exists():
            raise FileNotFoundError(f"Upload file not found: {args.upload_file}")
        file_id = upload_file(args.base_url, token, args.upload_file)
        if args.verbose:
            print(f"Uploaded file, received file_id={file_id}")

        if args.trigger_indexing:
            if args.verbose:
                print("Triggering indexing for uploaded file...")
            trigger_indexing(args.base_url, token, file_id)

    if not file_id:
        raise ValueError("A file_id must be provided via --file-id or --upload-file")

    last_response: Optional[requests.Response] = None
    for attempt in range(1, args.retries + 1):
        if args.verbose:
            print(f"Attempt {attempt}/{args.retries}: requesting mind map...")
        response = request_mindmap(args.base_url, token, file_id)
        last_response = response

        if response.status_code == 200:
            payload: Dict[str, Any] = response.json()
            print("✅ Mind map export succeeded")
            print("TSV preview:")
            tsv_lines = (payload.get("tsv") or "").splitlines()
            for preview_line in tsv_lines[:5]:
                print(preview_line)
            print("...") if len(tsv_lines) > 5 else None
            print(f"Nodes: {len(payload.get('nodes', []))}, Edges: {len(payload.get('edges', []))}")
            return

        if response.status_code == 404:
            if args.verbose:
                print(f"Mind map not ready yet (404). Waiting {args.wait_seconds} seconds...")
            time.sleep(args.wait_seconds)
            continue

        # Any other error should abort immediately
        print(f"❌ Mind map export failed with status {response.status_code}")
        print(response.text)
        response.raise_for_status()

    assert last_response is not None
    print(
        "❌ Mind map export did not succeed within the retry limit. "
        f"Last status: {last_response.status_code}, body: {last_response.text}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

