import json
import sys

import httpx


def read_stream(url: str, token: str | None) -> dict[str, object]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    parts: list[str] = []
    payload_calls = 0
    payload_has_evidence = False
    payload_has_subgraph = False

    with httpx.stream("GET", url, headers=headers, timeout=120.0) as response:
        if response.status_code != 200:
            body_bytes = response.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
            return {"http_status": response.status_code, "body": body_text}

        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line.split(":", 1)[1].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk, dict) and "data" in chunk and "code" in chunk:
                chunk = chunk.get("data") or {}
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            parts.append(delta.get("content") or "")
            for tool_call in delta.get("tool_calls") or []:
                fn = (tool_call or {}).get("function") or {}
                if fn.get("name") != "rag_arc_payload":
                    continue
                payload_calls += 1
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                if "evidence" in args:
                    payload_has_evidence = True
                if "subgraph" in args:
                    payload_has_subgraph = True

    return {
        "http_status": 200,
        "text": "".join(parts),
        "payload_calls": payload_calls,
        "payload_has_evidence": payload_has_evidence,
        "payload_has_subgraph": payload_has_subgraph,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python stream_chat_sse_comprehensive_test.py <url> [token]")
        return 1
    url = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) >= 3 else None
    result = read_stream(url, token)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
