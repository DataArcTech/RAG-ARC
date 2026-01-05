import asyncio
import json
import os
import sys
import uuid

import httpx


async def ws_roundtrip(ws_url: str, cookie_token: str, payload: str) -> dict:
    import websockets

    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=3,
        additional_headers={"Cookie": f"auth_token={cookie_token}"},
    ) as websocket:
        await websocket.send(payload)
        raw = await websocket.recv()
        return json.loads(raw)


def main() -> int:
    api_base = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
    auth_endpoint = f"{api_base}/auth"
    session_endpoint = f"{api_base}/session"

    ws_base = api_base.replace("https://", "wss://").replace("http://", "ws://")

    username = "ws_user"
    password = "ws_password"

    # Ensure user exists
    r = httpx.post(
        f"{auth_endpoint}/register",
        json={"name": "WS User", "user_name": username, "password": password},
        timeout=20.0,
    )
    if r.status_code not in (200, 201, 400):
        raise RuntimeError(f"register failed: {r.status_code} {r.text}")

    token_resp = httpx.post(
        f"{auth_endpoint}/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20.0,
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create session
    sess = httpx.post(session_endpoint, headers=headers, timeout=20.0)
    sess.raise_for_status()
    session_id = sess.json()

    ws_url = f"{ws_base}/rag_inference/stream_chat/{session_id}"

    # JSON payload (evidence + subgraph)
    resp = asyncio.run(
        ws_roundtrip(
            ws_url,
            token,
            json.dumps({"query": "Hello", "include_evidence": True, "return_subgraph": True}),
        )
    )
    assert resp["message"]["content"]["role"] == "assistant"
    assert isinstance(resp.get("chunks"), list)
    assert "evidence" in resp
    assert "subgraph" in resp

    # Plain text payload
    resp2 = asyncio.run(ws_roundtrip(ws_url, token, "ping"))
    assert resp2["message"]["content"]["role"] == "assistant"

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

