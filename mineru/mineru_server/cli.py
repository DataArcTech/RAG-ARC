import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from loguru import logger

from mineru_server.api import build_app, create_app
from mineru_server.client import MinerUServerClient
from mineru_server.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMP_DIR,
    DEFAULT_CONFIG_PATH,
    ServerConfig,
    apply_runtime_environment,
    ensure_mineru_import_path,
    save_config,
)
from mineru_server.env import load_dotenv

def _ensure_pythonpath_for_workers() -> None:
    """
    Uvicorn worker processes import the app via a module string.
    Ensure `MinerU/` (this repo subdir) is on PYTHONPATH so `import mineru_server...` works.
    """
    repo_mineru_dir = str(Path(__file__).resolve().parents[1])
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if repo_mineru_dir not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_mineru_dir, *parts]) if parts else repo_mineru_dir


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinerU server + client helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("server", help="Run FastAPI server.")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8899)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--max-jobs", type=int, default=1)
    s.add_argument(
        "--max-pending",
        type=int,
        default=int(os.getenv("MINERU_SERVER_MAX_PENDING_JOBS", "0")),
        help="Max pending parse jobs (queued + running). 0 means unlimited.",
    )

    s.add_argument("--backend", default="vlm-transformers")
    s.add_argument("--parse-method", default="auto")
    s.add_argument("--lang", default="ch")
    s.add_argument("--formula-enable", action="store_true", default=True)
    s.add_argument("--no-formula", action="store_false", dest="formula_enable")
    s.add_argument("--table-enable", action="store_true", default=True)
    s.add_argument("--no-table", action="store_false", dest="table_enable")

    s.add_argument("--model-source", default="modelscope")
    s.add_argument("--device", default="cuda", dest="device_mode")
    s.add_argument("--virtual-vram-gb", type=int, default=None)
    s.add_argument("--vllm-gpu-mem-util", type=float, default=0.5)
    s.add_argument("--vllm-enforce-eager", action="store_true", default=False)
    s.add_argument("--vllm-max-model-len", type=int, default=None)
    s.add_argument("--vllm-swap-space-gb", type=float, default=4.0)
    s.add_argument("--vllm-cpu-offload-gb", type=float, default=0.0)

    s.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    s.add_argument("--temp-dir", default=str(DEFAULT_TEMP_DIR))
    s.add_argument("--modelscope-cache-dir", default=None)
    s.add_argument("--hf-home", default=None)
    s.add_argument("--mineru-home", default=None)

    s.add_argument(
        "--caption-mode",
        default="content_list_then_llm",
        choices=["off", "content_list", "llm", "content_list_then_llm"],
    )
    s.add_argument("--chat-api-base-url", default=None)
    s.add_argument("--chat-api-key", default=None)
    s.add_argument("--chat-api-key-file", default=None)
    s.add_argument("--chat-model", default=None)
    s.add_argument("--chat-timeout-s", type=int, default=60)
    s.add_argument(
        "--caption-max-images",
        type=int,
        default=0,
        help="Max images to caption via LLM; <=0 means no limit.",
    )
    s.add_argument("--caption-context", default=None, help="Fixed context prepended to caption prompt.")
    s.add_argument("--caption-context-file", default=None, help="Read caption context from a file.")
    s.add_argument("--up", type=int, default=500, help="Context tokens above image reference.")
    s.add_argument("--down", type=int, default=500, help="Context tokens below image reference.")

    c = sub.add_parser("client", help="Parse via server and sync outputs.")
    c.add_argument("--base-url", default=os.environ.get("MINERU_SERVER_URL", "http://127.0.0.1:8899"))
    c.add_argument("--file", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, default=Path("./mineru_client_outputs"))
    c.add_argument("--backend", default="vlm-transformers")
    c.add_argument("--parse-method", default="auto")
    c.add_argument("--lang", default="ch")
    c.add_argument("--no-formula", action="store_false", dest="formula_enable")
    c.add_argument("--formula-enable", action="store_true", default=True)
    c.add_argument("--no-table", action="store_false", dest="table_enable")
    c.add_argument("--table-enable", action="store_true", default=True)
    c.add_argument("--start-page", type=int, default=0)
    c.add_argument("--end-page", type=int, default=None)
    c.add_argument("--output-format", default="mm_md", choices=["mm_md", "md_only", "content_list"])
    c.add_argument("--timeout", type=int, default=900)

    return parser.parse_args(argv)


def run_server(args: argparse.Namespace) -> None:
    _ensure_pythonpath_for_workers()
    # Load `.env` early so CLI flags can override it, but defaults can come from it.
    from mineru_server.config import SERVER_DIR

    load_dotenv(repo_dir=SERVER_DIR)
    env = os.environ
    chat_api_base_url = args.chat_api_base_url or env.get("CHAT_API_BASE_URL")
    chat_api_key = args.chat_api_key or env.get("CHAT_API_KEY")
    chat_api_key_file = args.chat_api_key_file or env.get("CHAT_API_KEY_FILE")
    chat_model = args.chat_model or env.get("OPENAI_CHAT_MODEL") or "gemini-2.5-flash"

    modelscope_cache_dir = args.modelscope_cache_dir or env.get("MODELSCOPE_CACHE")
    hf_home = args.hf_home or env.get("HF_HOME")

    mineru_home = args.mineru_home or str(Path(__file__).resolve().parents[1])

    caption_context = args.caption_context or env.get("CAPTION_CONTEXT") or ""
    caption_context_file = args.caption_context_file or env.get("CAPTION_CONTEXT_FILE")
    up_tokens = int(args.up or int(env.get("CAPTION_UP") or 500))
    down_tokens = int(args.down or int(env.get("CAPTION_DOWN") or 500))

    cfg = ServerConfig(
        host=str(args.host),
        port=int(args.port),
        workers=int(args.workers),
        max_jobs_per_worker=max(1, int(args.max_jobs)),
        max_pending_jobs=max(0, int(args.max_pending)),
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        temp_dir=str(Path(args.temp_dir).expanduser().resolve()),
        config_path=str(DEFAULT_CONFIG_PATH),
        mineru_home=str(Path(mineru_home).expanduser().resolve()),
        model_source=str(args.model_source),
        device_mode=str(args.device_mode),
        backend=str(args.backend),
        parse_method=str(args.parse_method),
        lang=str(args.lang),
        formula_enable=bool(args.formula_enable),
        table_enable=bool(args.table_enable),
        virtual_vram_gb=args.virtual_vram_gb,
        vllm_gpu_memory_utilization=float(args.vllm_gpu_mem_util),
        vllm_enforce_eager=bool(args.vllm_enforce_eager),
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_swap_space_gb=float(args.vllm_swap_space_gb),
        vllm_cpu_offload_gb=float(args.vllm_cpu_offload_gb),
        modelscope_cache_dir=str(Path(modelscope_cache_dir).expanduser().resolve()) if modelscope_cache_dir else ServerConfig().modelscope_cache_dir,
        hf_home=str(Path(hf_home).expanduser().resolve()) if hf_home else ServerConfig().hf_home,
        caption_mode=str(args.caption_mode),
        chat_api_base_url=chat_api_base_url,
        chat_api_key=chat_api_key,
        chat_api_key_file=chat_api_key_file,
        chat_model=str(chat_model),
        chat_timeout_s=int(args.chat_timeout_s),
        caption_max_images=int(args.caption_max_images),
        caption_context=str(caption_context or ""),
        caption_context_file=str(caption_context_file) if caption_context_file else None,
        caption_up_tokens=int(up_tokens),
        caption_down_tokens=int(down_tokens),
    )

    save_config(cfg, include_secrets=(cfg.workers > 1))

    logger.info(
        f"Starting MinerU server: host={cfg.host} port={cfg.port} workers={cfg.workers} "
        f"max_jobs={cfg.max_jobs_per_worker} backend={cfg.backend} device={cfg.device_mode} caption_mode={cfg.caption_mode}"
    )

    import uvicorn

    if cfg.workers <= 1:
        ensure_mineru_import_path(cfg)
        apply_runtime_environment(cfg)
        app = build_app(cfg)
        uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1, reload=False, log_level="info")
        return

    uvicorn.run("mineru_server.api:create_app", host=cfg.host, port=cfg.port, workers=cfg.workers, factory=True, reload=False, log_level="info")


def run_client(args: argparse.Namespace) -> None:
    client = MinerUServerClient(base_url=str(args.base_url), timeout_s=int(args.timeout))
    result = client.parse(
        file_path=args.file,
        backend=args.backend,
        parse_method=args.parse_method,
        lang=args.lang,
        formula_enable=bool(args.formula_enable),
        table_enable=bool(args.table_enable),
        start_page=int(args.start_page),
        end_page=int(args.end_page) if args.end_page is not None else None,
        output_format=str(args.output_format),
    )
    task_id = result.get("task_id")
    if not task_id:
        raise SystemExit(f"Parse failed: {result}")
    task_root = client.sync_task(str(task_id), args.output_dir)
    out = args.output_dir / "parse_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"parse_result": result, "synced_task_root": str(task_root)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"task_id": task_id, "synced_task_root": str(task_root), "parse_result_json": str(out)}, ensure_ascii=False))


def main(argv: Optional[List[str]] = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    # Backward compatible: `python MinerU/mineru_server.py --port 8899 ...` defaults to server mode.
    if not args_list or args_list[0].startswith("-") or args_list[0] not in {"server", "client"}:
        args_list = ["server", *args_list]
    args = parse_args(args_list)
    if args.cmd == "server":
        run_server(args)
        return
    if args.cmd == "client":
        run_client(args)
        return
    raise SystemExit("Unknown command")
