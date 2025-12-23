import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class ConcurrencyConfig:
    api_base: str
    sse_parallel: int = 5
    chat_parallel: int = 5
    ws_parallel: int = 3


def _ws_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base.replace("https://", "wss://").replace("http://", "ws://")


async def _read_sse(url: str, headers: dict[str, str]) -> dict[str, object]:
    parts: list[str] = []
    progress_hits = 0
    done = False
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return {"status": resp.status_code, "body": body.decode("utf-8", errors="replace")}
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line.split(":", 1)[1].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                parts.append(delta.get("content") or "")
                for tool_call in delta.get("tool_calls") or []:
                    fn = (tool_call or {}).get("function") or {}
                    if fn.get("name") == "rag_arc_progress":
                        progress_hits += 1
    return {"status": 200, "done": done, "progress_hits": progress_hits, "text_len": len("".join(parts))}


async def _post_chat(url: str, headers: dict[str, str], payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        return {"status": resp.status_code, "body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text}


async def _ws_roundtrip(ws_url: str, cookie_token: str, payload: str) -> dict:
    import websockets

    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=5,
        additional_headers={"Cookie": f"auth_token={cookie_token}"},
    ) as websocket:
        await websocket.send(payload)
        raw = await websocket.recv()
        return json.loads(raw)


async def run(config: ConcurrencyConfig, *, out_dir: Path) -> None:
    api_base = config.api_base.rstrip("/")
    auth_endpoint = f"{api_base}/auth"
    session_endpoint = f"{api_base}/session"
    knowledge_endpoint = f"{api_base}/knowledge"

    username = f"conc_{uuid.uuid4().hex[:8]}"
    password = "conc_password"
    unique_phrase = f"CONCURRENCY_UNIQUE_{uuid.uuid4().hex}"
    uploaded_file_id: str | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{auth_endpoint}/register",
            json={"name": "Concurrency User", "user_name": username, "password": password},
        )
        if r.status_code not in (200, 201, 400):
            raise RuntimeError(f"register failed: {r.status_code} {r.text}")
        token_resp = await client.post(
            f"{auth_endpoint}/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_resp.raise_for_status()
        token = token_resp.json()["access_token"]

        sess = await client.post(session_endpoint, headers={"Authorization": f"Bearer {token}"})
        sess.raise_for_status()
        session_id = sess.json()

    headers = {"Authorization": f"Bearer {token}"}

    async def upload_and_trigger_indexing() -> dict[str, object]:
        nonlocal uploaded_file_id
        payload = f"RAG-ARC concurrency test file\nUniquePhrase: {unique_phrase}\n"
        async with httpx.AsyncClient(timeout=60.0) as client:
            up = await client.post(
                knowledge_endpoint,
                headers=headers,
                files={"file": ("concurrency.txt", payload.encode("utf-8"), "text/plain")},
                data={"relative_path": f"concurrency/{unique_phrase}.txt"},
            )
            up.raise_for_status()
            uploaded_file_id = up.json()

            trig = await client.post(
                f"{knowledge_endpoint}/trigger_indexing",
                headers={**headers, "Content-Type": "application/json"},
                json={"file_ids": [uploaded_file_id]},
            )
            trig.raise_for_status()
            return {"uploaded_file_id": uploaded_file_id, "trigger_message": trig.json().get("message")}

    async def wait_for_indexed(*, timeout_s: int = 90) -> dict[str, object]:
        if not uploaded_file_id:
            return {"indexed": False, "reason": "no uploaded_file_id"}
        started = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - started < timeout_s:
                resp = await client.get(f"{knowledge_endpoint}/list_files?limit=1000&offset=0", headers=headers)
                resp.raise_for_status()
                files = (resp.json() or {}).get("files") or []
                status = None
                for item in files:
                    if (item or {}).get("file_id") == uploaded_file_id:
                        status = (item or {}).get("status")
                        break
                if status == "INDEXED":
                    return {"indexed": True, "status": status, "wait_s": time.time() - started}
                if status in {"FAILED", "DELETED"}:
                    return {"indexed": False, "status": status, "wait_s": time.time() - started}
                await asyncio.sleep(1)
        return {"indexed": False, "status": "TIMEOUT", "wait_s": time.time() - started}

    async def validate_retrieval() -> dict[str, object]:
        if not uploaded_file_id:
            return {"ok": False, "reason": "no uploaded_file_id"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_base}/rag_inference/chat",
                headers={**headers, "Content-Type": "application/json"},
                json={"query": unique_phrase},
            )
            resp.raise_for_status()
            body = resp.json()
            chunks = body.get("chunks") or []
            returned = set()
            for ch in chunks:
                meta = (ch or {}).get("metadata") or {}
                src = meta.get("source_file_id") or meta.get("sourceFileId")
                if src:
                    returned.add(src)
            ok = uploaded_file_id in returned
            return {
                "ok": ok,
                "returned_source_file_ids": sorted(returned),
                "chunks_len": len(chunks),
            }

    async def cleanup_uploaded_file() -> dict[str, object]:
        if not uploaded_file_id:
            return {"deleted": False, "reason": "no uploaded_file_id"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.delete(f"{knowledge_endpoint}/{uploaded_file_id}", headers=headers)
            if r.status_code not in (202, 204, 404):
                return {"deleted": False, "status": r.status_code, "body": r.text}
            # Poll download until 404 to ensure it's truly gone
            for _ in range(30):
                d = await client.get(f"{knowledge_endpoint}/{uploaded_file_id}/download", headers=headers)
                if d.status_code == 404:
                    return {"deleted": True, "status": r.status_code}
                await asyncio.sleep(1)
            return {"deleted": False, "status": r.status_code, "reason": "download_not_404"}

    upload_info = await upload_and_trigger_indexing()

    sse_url = f"{api_base}/rag_inference/stream_chat/{session_id}?query=hello&include_evidence=false"
    chat_url = f"{api_base}/rag_inference/chat"
    ws_url = f"{_ws_base(api_base)}/rag_inference/stream_chat/{session_id}"

    tasks = []
    for i in range(config.sse_parallel):
        tasks.append(_read_sse(f"{sse_url}&_i={i}", headers))
    for i in range(config.chat_parallel):
        tasks.append(_post_chat(chat_url, headers, {"query": f"ping {i}"}))
    for i in range(config.ws_parallel):
        tasks.append(_ws_roundtrip(ws_url, token, json.dumps({"query": f"ws {i}"})))

    started = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - started

    index_wait = await wait_for_indexed()
    retrieval_check = await validate_retrieval() if index_wait.get("indexed") else {"ok": False, "reason": "not indexed"}
    cleanup = await cleanup_uploaded_file()

    out = {
        "duration_s": duration,
        "unique_phrase": unique_phrase,
        "upload": upload_info,
        "index_wait": index_wait,
        "retrieval_check": retrieval_check,
        "cleanup": cleanup,
        "results": [],
    }
    for item in results:
        if isinstance(item, Exception):
            out["results"].append({"error": str(item)})
        else:
            out["results"].append(item)

    out_path = out_dir / "concurrency_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [r for r in out["results"] if isinstance(r, dict) and r.get("status") and int(r["status"]) >= 400]
    if failures:
        raise RuntimeError(f"Concurrency failures: {failures[:3]}")
    if not out["index_wait"].get("indexed"):
        raise RuntimeError(f"Indexing did not complete: {out['index_wait']}")
    if not out["retrieval_check"].get("ok"):
        raise RuntimeError(f"Retrieval did not return uploaded chunk: {out['retrieval_check']}")
    if not out["cleanup"].get("deleted"):
        raise RuntimeError(f"Cleanup did not complete: {out['cleanup']}")


def main() -> int:
    api_base = os.getenv("API_BASE", "http://localhost:8000")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("local") / "e2e" / f"concurrency_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ConcurrencyConfig(api_base=api_base)
    asyncio.run(run(cfg, out_dir=out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
