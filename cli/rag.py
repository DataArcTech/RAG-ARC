import asyncio
import json
import logging
import mimetypes
import os
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import List, Optional, Dict, Any
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import typer

from framework.runtime_warnings import configure_runtime_warnings

configure_runtime_warnings()

from application.knowledge.module import Knowledge
from application.rag_inference.cli_module import (
    PipelineArtifacts,
    RAGInferenceCLIModule,
)
from application.rag_inference.module import RAGInference
from config.application.deepsearch_tool_server_config import load_tool_server_config
from core.presentation.summary import PipelineSummary, DeepSearchReport
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from cli.bootstrap import CLIContext, initialize
from encapsulation.data_model.orm_models import FileStatus
from framework.register import Register
from dotenv import load_dotenv
from core.graph_adapter.scope_provider import configure_scope_provider
from config.output_limits import (
    CHAT_TOP_CHUNKS,
    DEEPSEARCH_TOP_CHUNKS,
)
from core.presentation.graph_chain import build_graph_chain
from core.utils.json_safe import json_safe

logger = logging.getLogger(__name__)
app = typer.Typer(help="Run RAG-ARC algorithms through CLI without HTTP layer.")
_CHAT_MCP_MODULE = None
_CHAT_MCP_MODULE = None

def _get_rag_runner() -> RAGInferenceCLIModule:
    registrator = Register()
    rag_module = registrator.get_object("rag_inference")
    if not isinstance(rag_module, RAGInference):
        raise RuntimeError("Registered rag_inference module is not of expected type")
    return RAGInferenceCLIModule(rag_module)


def _get_knowledge_module() -> Knowledge:
    registrator = Register()
    knowledge_module = registrator.get_object("knowledge")
    if not isinstance(knowledge_module, Knowledge):
        raise RuntimeError("Registered knowledge module is not of expected type")
    return knowledge_module


def _get_chat_mcp_server():
    """Lazy import chat MCP server after registry initialization."""

    global _CHAT_MCP_MODULE
    if _CHAT_MCP_MODULE is not None:
        return _CHAT_MCP_MODULE
    import app_registration

    app_registration.initialize()
    from api.mcp import server as chat_mcp_server

    _CHAT_MCP_MODULE = chat_mcp_server
    return chat_mcp_server


def _ensure_cli_output_dir(owner_id: UUID | str) -> Path:
    """Return the folder used for CLI JSON artifacts, creating it if needed."""

    folder = Path("local") / "cli" / str(owner_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_json_payload(payload: Dict[str, Any], owner_id: UUID | str, prefix: str) -> Path:
    """Persist a JSON payload to the CLI output folder with a timestamped filename."""

    destination = _ensure_cli_output_dir(owner_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = destination / f"{prefix}_{timestamp}.json"
    safe_payload = json_safe(payload)
    path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _emit_json_output(
    payload: Dict[str, Any],
    *,
    owner_id: UUID | str | None,
    prefix: str,
    print_to_console: bool = False,
    raw_payload: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist JSON payload to disk and optionally emit it to stdout."""

    payload_safe = json_safe(payload)
    serialized = json.dumps(payload_safe, ensure_ascii=False, indent=2, default=str)
    payload_size = len(serialized.encode("utf-8"))
    typer.echo(f"[debug] trimmed payload size: {payload_size} bytes")
    if print_to_console or owner_id is None:
        typer.echo(serialized)
        typer.echo("[debug] payload printed to console")
    if owner_id is None:
        return None
    target = _write_json_payload(payload_safe, owner_id, prefix)
    typer.echo(f"JSON payload saved to {target} ({target.stat().st_size} bytes)")
    if raw_payload is not None:
        raw_safe = json_safe(raw_payload)
        raw_serialized = json.dumps(raw_safe, ensure_ascii=False, indent=2, default=str)
        raw_target = _write_json_payload(raw_safe, owner_id, f"{prefix}_raw")
        typer.echo(
            f"Raw JSON saved to {raw_target} ({raw_target.stat().st_size} bytes; in-memory size {len(raw_serialized.encode('utf-8'))} bytes)"
        )
    return target


def _print_summary(
    summary: PipelineSummary,
    *,
    output_json: bool,
    owner_id: UUID | None,
    prefix: str,
) -> None:
    if output_json:
        payload = asdict(summary)
        raw_payload = deepcopy(payload)
        payload.pop("raw_chunks", None)
        payload.pop("subgraph", None)
        _emit_json_output(payload, owner_id=owner_id, prefix=prefix, raw_payload=raw_payload)
        return
    typer.echo(f"Owner ID: {summary.owner_id}")
    typer.echo(f"Original Query: {summary.original_query}")
    typer.echo(f"Rewritten Query: {summary.rewritten_query}")
    typer.echo("")
    if summary.llm_response:
        typer.echo("LLM Response:")
        typer.echo(summary.llm_response)
        typer.echo("")
    typer.echo("Top Chunks:")
    for preview in summary.chunk_previews:
        typer.echo(f"#{preview.index} [{preview.chunk_id}] {preview.preview}")
    if summary.subgraph:
        typer.echo("\nSubgraph:")
        typer.echo(json.dumps(summary.subgraph, ensure_ascii=False, indent=2))
    if summary.evidence and not output_json:
        seeds = summary.evidence.get("seed_entities") or []
        if seeds:
            typer.echo("\nSeed entities:")
            for name in seeds:
                typer.echo(f"- {name}")
        triples = summary.evidence.get("triples") or []
        if triples:
            typer.echo("\nGraph triples:")
            for triple in triples[:5]:
                typer.echo(f"- {triple['head']} -[{triple['relation']}]-> {triple['tail']}")


def _run_pipeline(
    ctx: CLIContext,
    query: str,
    return_subgraph: bool,
    skip_llm: bool,
    output_json: bool,
    include_evidence: bool,
    mode: str = "multipath",
    output_prefix: str = "pipeline",
) -> PipelineArtifacts:
    runner = _get_rag_runner()
    if mode == "graph":
        artifacts = runner.run_graph_pipeline(
            query=query,
            owner_id=ctx.owner_id,
            return_subgraph=return_subgraph,
            skip_llm=skip_llm,
        )
    else:
        artifacts = runner.run_pipeline(
            query=query,
            owner_id=ctx.owner_id,
            return_subgraph=return_subgraph,
            skip_llm=skip_llm,
        )
    summary = PipelineSummary.from_artifacts(
        owner_id=str(ctx.owner_id),
        artifacts=artifacts,
        max_chunks=CHAT_TOP_CHUNKS,
        max_chars=50,
        include_evidence=include_evidence,
    )
    _print_summary(
        summary,
        output_json=output_json,
        owner_id=ctx.owner_id,
        prefix=output_prefix,
    )
    return artifacts


def _gather_files(folder: Path, pattern: str, recursive: bool) -> List[Path]:
    candidates = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return [path for path in candidates if path.is_file()]


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"

def _safe_relative_filename(root: Path, path: Path) -> str:
    """Return a stable, safe relative path name suitable for storage metadata."""

    try:
        rel = path.resolve().relative_to(root.resolve())
        rel_str = rel.as_posix()
    except Exception:  # noqa: BLE001
        rel_str = path.name

    rel_str = rel_str.lstrip("/").replace("\\", "/").strip()
    if not rel_str:
        return path.name

    parts = [part for part in PurePosixPath(rel_str).parts if part not in {"", ".", ".."}]
    if not parts:
        return path.name
    return str(PurePosixPath(*parts))

def _project_relative_filename(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(
            f"path must be under project root: {PROJECT_ROOT.resolve()}"
        ) from exc
    rel_str = rel.as_posix().lstrip("/")
    return f"{PROJECT_ROOT.name}/{rel_str}" if rel_str else f"{PROJECT_ROOT.name}/"


REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOOL_SERVER_CONFIG = REPO_ROOT / "config/json_configs/deepsearch_tool_mcp_server.json"


def _ingest_single_file(path: Path, knowledge: Knowledge, owner_id, *, logical_filename: Optional[str] = None) -> bool:
    typer.echo(f"\n→ Ingesting {path}")
    try:
        file_bytes = path.read_bytes()
    except OSError as exc:
        typer.secho(f"  ! Failed to read file: {exc}", fg=typer.colors.RED)
        return False

    content_type = _guess_content_type(path)
    try:
        file_id = knowledge.file_storage.upload_file(
            filename=(str(logical_filename or "").strip() or path.name),
            file_data=file_bytes,
            owner_id=owner_id,
            content_type=content_type,
        )
        typer.echo(f"  • Stored as file_id={file_id} ({len(file_bytes)} bytes)")
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"  ! Upload failed: {exc}", fg=typer.colors.RED)
        return False

    try:
        index_result = asyncio.run(knowledge.file_index.index_file(file_id))
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"  ! Indexing failed: {exc}", fg=typer.colors.RED)
        return False

    if not index_result.get("success"):
        typer.secho(f"  ! Indexing failed: {index_result.get('error_message')}", fg=typer.colors.RED)
        return False

    chunk_count = len(index_result.get("chunk_ids") or [])
    typer.echo(f"  • Indexed successfully with {chunk_count} chunk(s)")
    return True


def _resolve_tool_server_config_path() -> Path:
    raw_path = os.getenv("DEEPSEARCH_TOOL_MCP_CONFIG_PATH")
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    return _DEFAULT_TOOL_SERVER_CONFIG


def _build_tool_server_from_config():
    config_path = _resolve_tool_server_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Tool MCP server config not found: {config_path}")
    config = load_tool_server_config(config_path)
    return config.build()


def _print_tool_catalog(server) -> None:
    """Print currently registered MCP tools to console."""

    try:
        descriptors = list(server.list_registered_tools())
    except AttributeError:
        logger.warning("Tool server does not expose list_registered_tools")
        return
    descriptors.sort(key=lambda item: item.name)
    typer.echo("Registered MCP tools:")
    if not descriptors:
        typer.echo("  (no tools enabled)")
        return
    for descriptor in descriptors:
        typer.echo(f"  - {descriptor.name} [{descriptor.channel}] :: {descriptor.description}")


def _print_deepsearch_result(
    payload: Dict[str, Any],
    *,
    output_json: bool,
    owner_id: UUID | None,
    prefix: str = "deepsearch",
    include_evidence: bool = False,
    raw_payload: Optional[Dict[str, Any]] = None,
    graph_store: Any | None = None,
) -> None:
    """Render DeepSearchService.run output."""

    trimmed_payload = trim_deepsearch_payload(
        payload,
        include_evidence=include_evidence,
        graph_store=graph_store,
    )

    if output_json:
        _emit_json_output(trimmed_payload, owner_id=owner_id, prefix=prefix, raw_payload=raw_payload)
        return

    structured = (trimmed_payload.get("report") or {}).get("structured_report") or {}
    text = structured.get("text")
    if isinstance(text, str) and text.strip():
        typer.echo(text.strip())
        return

    summary = DeepSearchReport.from_payload(
        trimmed_payload,
        top_chunk_limit=DEEPSEARCH_TOP_CHUNKS,
        graph_chain_builder=build_graph_chain,
    )
    typer.echo(f"Question: {summary.question}")
    if summary.final_answer:
        typer.echo("\nAnswer:")
        typer.echo(summary.final_answer)


def _print_chat_mcp_catalog(chat_mcp_server) -> None:
    """Print tool names exposed by the chat MCP server."""

    try:
        tools = asyncio.run(chat_mcp_server.list_tools())
    except RuntimeError:
        # Fallback for rare cases when an event loop is already running
        loop = asyncio.get_event_loop()
        tools = loop.run_until_complete(chat_mcp_server.list_tools())
    typer.echo("Chat MCP tools:")
    if not tools:
        typer.echo("  (no chat MCP tools registered)")
        return
    for name in sorted(tools):
        typer.echo(f"  - {name}")


@app.command("tool-mcp-server")
def tool_mcp_server(
    transport: str = typer.Option(
        "stdio",
        help="Transport to expose (stdio, sse, or streamable-http).",
    ),
    host: str = typer.Option("127.0.0.1", help="Host for HTTP/SSE transports."),
    port: int = typer.Option(8765, help="Port for HTTP/SSE transports."),
    path: str = typer.Option("/mcp/tools", help="Path for HTTP/SSE transports."),
) -> None:
    """Launch the DeepSearch tool MCP server for external agents."""

    load_dotenv()
    configure_scope_provider()
    server = _build_tool_server_from_config()
    _print_tool_catalog(server)
    normalized_path = (path or "").strip()
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    if transport == "stdio":
        asyncio.run(server.run_stdio_async())
        return
    if transport == "sse":
        asyncio.run(server.run_sse_async(host=host, port=port, path=normalized_path))
        return
    if transport in {"streamable", "streamable-http"}:
        asyncio.run(server.run_streamable_http_async(host=host, port=port, path=normalized_path))
        return
    raise typer.BadParameter("transport must be one of: stdio, sse, streamable-http")


@app.command("chat-mcp-server")
def chat_mcp_server_cmd(
    transport: str = typer.Option(
        "stdio",
        help="Transport to expose (stdio, sse, or streamable-http).",
    ),
    host: str = typer.Option("127.0.0.1", help="Host for HTTP/SSE transports."),
    port: int = typer.Option(8785, help="Port for HTTP/SSE transports."),
    path: str = typer.Option("/mcp/chat", help="Path for HTTP/SSE transports."),
) -> None:
    """Launch the account/chat MCP server for external agents."""

    load_dotenv()
    configure_scope_provider()
    chat_server = _get_chat_mcp_server()
    _print_chat_mcp_catalog(chat_server)
    normalized_path = (path or "").strip()
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path

    if transport == "stdio":
        asyncio.run(chat_server.run_stdio_async())
        return
    if transport == "sse":
        asyncio.run(chat_server.run_sse_async(host=host, port=port, path=normalized_path))
        return
    if transport in {"streamable", "streamable-http"}:
        asyncio.run(chat_server.run_streamable_http_async(host=host, port=port, path=normalized_path))
        return
    raise typer.BadParameter("transport must be one of: stdio, sse, streamable-http")


@app.command("deepsearch")
def deepsearch(
    question: str = typer.Argument(..., help="Question to execute through DeepSearch"),
    owner_id: Optional[str] = typer.Option(None, help="Optional owner UUID; defaults to CLI owner"),
    output_json: bool = typer.Option(False, "--json/--no-json", help="Choose between JSON or concise output"),
    include_evidence: bool = typer.Option(
        False,
        "--with-evidence/--no-evidence",
        help="Attach chunk/graph evidence bundle (chunks/seeds/graph_chain).",
    ),
    save_raw: bool = typer.Option(
        False,
        "--save-raw/--no-save-raw",
        help="Persist the full raw payload alongside the trimmed JSON output.",
    ),
) -> None:
    """Run DeepSearchService over the configured graph adapter."""

    ctx = initialize(owner_id=owner_id)
    registrator = Register()
    try:
        service = registrator.get_object("deepsearch_service")
    except KeyError:
        typer.secho("DeepSearch service is not registered; check DEEPSEARCH_SERVICE_CONFIG_PATH.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Running DeepSearch for owner {ctx.owner_id} ...")
    result = asyncio.run(
        service.run(
            question,
            owner_id=str(ctx.owner_id),
        )
    )
    graph_store = None
    try:
        rag = registrator.get_object("rag_inference")
        if isinstance(rag, RAGInference):
            graph_store = rag.get_graph_store()
    except Exception:  # noqa: BLE001
        graph_store = None
    _print_deepsearch_result(
        result,
        output_json=output_json,
        owner_id=ctx.owner_id,
        prefix="deepsearch",
        include_evidence=include_evidence,
        raw_payload=result if save_raw else None,
        graph_store=graph_store,
    )

@app.command()
def chat(
    query: str = typer.Argument(..., help="User question to run through RAG pipeline."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(False, "--subgraph", help="Export subgraph metadata."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
    include_evidence: bool = typer.Option(
        False,
        "--with-evidence/--no-evidence",
        help="Attach chunk/seeds/triple summaries (enables subgraph export internally).",
    ),
) -> None:
    """Run the full RAG pipeline and stream answer to terminal."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph or include_evidence,
        skip_llm=False,
        output_json=output_json,
        include_evidence=include_evidence,
        output_prefix="chat",
    )


@app.command()
def pipeline(
    query: str = typer.Argument(..., help="User question to run through RAG pipeline."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(False, "--subgraph", help="Export subgraph metadata."),
    skip_llm: bool = typer.Option(True, "--skip-llm/--with-llm", help="Skip LLM call for faster debugging."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
    include_evidence: bool = typer.Option(
        False,
        "--with-evidence/--no-evidence",
        help="Attach chunk/seeds/triple summaries (enables subgraph export internally).",
    ),
) -> None:
    """Run pipeline and inspect intermediate artifacts (defaults to skipping LLM)."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph or include_evidence,
        skip_llm=skip_llm,
        output_json=output_json,
        include_evidence=include_evidence,
        output_prefix="pipeline",
    )


@app.command("graph-qa")
def graph_qa(
    query: str = typer.Argument(..., help="Question answered purely via graph retriever."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(True, "--subgraph/--no-subgraph", help="Export subgraph metadata."),
    skip_llm: bool = typer.Option(False, "--skip-llm/--with-llm", help="Skip LLM call for debugging."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
    include_evidence: bool = typer.Option(
        False,
        "--with-evidence/--no-evidence",
        help="Attach chunk/seeds/triple summaries (enables subgraph export internally).",
    ),
) -> None:
    """Query only the graph retriever and optionally export the subgraph."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph or include_evidence,
        skip_llm=skip_llm,
        output_json=output_json,
        include_evidence=include_evidence,
        mode="graph",
        output_prefix="graph_qa",
    )


@app.command("ingest-file")
def ingest_file(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Single document to ingest.",
    ),
    owner_id: str = typer.Option(None, help="Optional owner UUID; defaults to a random UUID."),
) -> None:
    """Upload and index a single file."""
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    success = _ingest_single_file(path, knowledge, ctx.owner_id, logical_filename=_project_relative_filename(path))
    if not success:
        raise typer.Exit(code=1)


@app.command("ingest-folder")
def ingest_folder(
    folder: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Folder containing documents to ingest.",
    ),
    owner_id: str = typer.Option(None, help="Optional owner UUID; defaults to a random UUID."),
    pattern: str = typer.Option("*", help="Glob pattern for files inside the folder."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recursively search subdirectories."),
    limit: Optional[int] = typer.Option(None, help="Limit the number of files to ingest."),
) -> None:
    """Upload, index, and build graph data for all files inside a folder."""
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    _project_relative_filename(folder)
    files = _gather_files(folder, pattern, recursive)
    if limit is not None and limit > 0:
        files = files[:limit]

    if not files:
        typer.secho("No files matched the given pattern.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(files)} file(s) in {folder}")
    succeeded = 0
    for path in files:
        if _ingest_single_file(path, knowledge, ctx.owner_id, logical_filename=_project_relative_filename(path)):
            succeeded += 1

    typer.echo(f"\nCompleted ingestion: {succeeded}/{len(files)} file(s) indexed successfully.")


@app.command("list-files")
def list_files(
    owner_id: str = typer.Option(None, help="Optional owner UUID; defaults to a random UUID."),
    limit: int = typer.Option(20, min=1, max=500, help="Maximum number of files to display."),
    offset: int = typer.Option(0, min=0, help="Number of files to skip."),
    status: Optional[str] = typer.Option(None, help="Filter by file status (e.g., STORED, INDEXED)."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """List files accessible to the owner."""
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    status_enum = None
    if status:
        try:
            status_enum = FileStatus[status.upper()]
        except KeyError:
            typer.secho(f"Unknown status '{status}'. Use values like STORED/INDEXED/FAILED.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    files = knowledge.list_user_files(user_id=ctx.owner_id, status=status_enum, limit=limit, offset=offset)
    total = knowledge.count_user_files(ctx.owner_id, status=status_enum)

    records = [
        {
            "file_id": file.file_id,
            "filename": file.filename,
            "status": file.status.value,
            "updated_at": file.updated_at.isoformat(),
            "created_at": file.created_at.isoformat(),
            "size": file.file_size,
            "content_type": file.content_type,
        }
        for file in files
    ]

    if output_json:
        payload = {
            "owner_id": str(ctx.owner_id),
            "limit": limit,
            "offset": offset,
            "total": total,
            "files": records,
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    typer.echo(f"Total files: {total} | Showing {len(records)} entries (offset {offset})")
    for entry in records:
        typer.echo(
            f"- {entry['file_id']} | {entry['status']} | {entry['filename']} | {entry['updated_at']} | {entry['size']} bytes"
        )


@app.command("delete-file")
def delete_file(
    file_id: str = typer.Argument(..., help="File ID to delete."),
    owner_id: str = typer.Option(None, help="Optional owner UUID overriding default."),
) -> None:
    """Soft delete a file for CLI testing (metadata-only)."""
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    try:
        result = asyncio.run(knowledge.mark_file_deleted_cli(file_id, ctx.owner_id))
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to delete file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    status_msg = result.get("status") if isinstance(result, dict) else "marked"
    typer.echo(f"File {file_id} marked as deleted (status: {status_msg}).")


@app.command("trigger-index")
def trigger_index(
    file_ids: List[str] = typer.Argument(..., help="One or more file IDs to re-index.", metavar="FILE_ID"),
    owner_id: str = typer.Option(None, help="Optional owner UUID overriding default."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-index files even when their status is already INDEXED (skips only when in progress).",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for indexing to finish (recommended for CLI runs without Celery).",
    ),
) -> None:
    """Trigger indexing for existing files."""
    if not file_ids:
        typer.secho("Provide at least one FILE_ID.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    try:
        message = asyncio.run(knowledge.trigger_indexing(file_ids, ctx.owner_id, force=force, wait=wait))
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to trigger indexing: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(message)


@app.command("export-graph")
def export_graph(
    owner_id: str = typer.Option(None, help="Optional owner UUID; retained for consistency."),
    max_nodes: int = typer.Option(500, help="Maximum number of nodes to include."),
    max_edges: int = typer.Option(2000, help="Maximum number of edges to include."),
    include_node_types: Optional[List[str]] = typer.Option(
        None,
        "--include-node-type",
        help="Filter node types (repeatable, e.g., --include-node-type entity).",
    ),
    output: Optional[Path] = typer.Option(None, help="Optional path to write the graph JSON."),
    output_json: bool = typer.Option(False, "--json", help="Print graph JSON to stdout."),
) -> None:
    """Export the entire knowledge graph to stdout or a file."""
    ctx = initialize(owner_id=owner_id)
    runner = _get_rag_runner()
    graph_store = runner.get_graph_store()
    if not graph_store:
        typer.secho("Current retriever does not expose a graph store.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    graph_store_class_name = graph_store.__class__.__name__
    if graph_store_class_name == "PrunedHippoRAGNeo4jStore":
        from encapsulation.database.utils.graph_export_utils_neo4j import (
            GraphExporterNeo4j as GraphExporter,
        )
    else:
        from encapsulation.database.utils.graph_export_utils import GraphExporter

    scope = str(ctx.owner_id) if ctx.owner_id else None
    graph_data = GraphExporter.export_full_graph(
        graph_store=graph_store,
        max_nodes=max_nodes,
        max_edges=max_edges,
        include_node_types=include_node_types,
        owner_id=scope,
        owner_scope_label=scope or "GLOBAL_ADMIN",
    )

    json_payload = json.dumps(graph_data, ensure_ascii=False, indent=2)
    if output:
        output.write_text(json_payload)
        typer.echo(f"Graph JSON written to {output}")
    if output_json or not output:
        typer.echo(json_payload)
    else:
        typer.echo(
            f"Graph summary: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges"
        )


@app.command("kg-maintenance-l2")
def kg_maintenance_l2(
    owner_id: str = typer.Option(None, help="Optional owner UUID overriding default."),
    file_ids: Optional[List[str]] = typer.Option(
        None,
        "--file-id",
        help="Optional file_id scope (repeatable). When omitted, runs owner-global repair.",
    ),
    rebuild_same_as: bool = typer.Option(
        False,
        "--rebuild-same-as",
        help="Supersede and rebuild SAME_AS edges for the owner (expensive).",
    ),
    backfill_missing_mentions: bool = typer.Option(
        False,
        "--backfill-missing-mentions",
        help="Backfill missing EntityMention nodes before running repair (can be expensive).",
    ),
    output_json: bool = typer.Option(True, "--json/--no-json", help="Print JSON result to stdout."),
) -> None:
    """Run L2 KG maintenance (repair) for an owner scope."""
    ctx = initialize(owner_id=owner_id)
    runner = _get_rag_runner()
    graph_store = runner.get_graph_store()
    if not graph_store or not callable(getattr(graph_store, "run_kg_maintenance_l2_for_owner", None)):
        typer.secho("Current retriever does not expose a graph store with L2 maintenance.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    stats = graph_store.run_kg_maintenance_l2_for_owner(
        owner_id=str(ctx.owner_id),
        file_ids=[str(x).strip() for x in (file_ids or []) if str(x or "").strip()],
        rebuild_same_as=bool(rebuild_same_as),
        backfill_missing_mentions=bool(backfill_missing_mentions),
    )
    if output_json:
        typer.echo(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
        return
    typer.echo(f"L2 KG maintenance done: success={stats.get('success')} run_id={stats.get('run_id')}")


@app.command("semantic-unit-eval")
def semantic_unit_eval(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Markdown/text file to chunk and retrieve against (no DB required).",
    ),
    level: str = typer.Option(
        "standard",
        help="Semantic unit chunking level: disabled/basic/standard/advanced.",
    ),
    query: List[str] = typer.Option(
        None,
        "--query",
        help="Query to run against the temporary BM25 index (repeatable).",
    ),
    k: int = typer.Option(5, min=1, max=50, help="Top-k retrieval results."),
    force_slices: bool = typer.Option(
        False,
        "--force-slices",
        help="Force table/code/list to take anchor+slice path (sets *_small_max_tokens=1).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory (default: test_output/semantic_unit_eval/<stem>_<timestamp>).",
    ),
    owner_id: Optional[str] = typer.Option(
        None,
        help="Owner UUID for retrieval filtering (default: random UUID).",
    ),
) -> None:
    """
    Manual evaluation runner for semantic-unit chunking + anchor backfill + retrieval merge.

    This command avoids external services and databases by:
    - chunking input markdown/text with SemanticUnitChunker
    - assigning synthetic chunk IDs + backfilling anchor_chunk_id
    - building a temporary Tantivy BM25 index on disk
    - running MultiPath retrieval (BM25 only) to exercise anchor backfill + evidence sync
    """

    from config.core.file_management.chunker.chunker_config import (
        SemanticUnitChunkerConfig,
        TokenChunkerConfig,
    )
    from config.encapsulation.database.bm25_config import BM25BuilderConfig
    from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
    from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
    from core.file_management.index_manager import IndexManager
    from core.presentation.evidence import serialize_chunks
    from encapsulation.data_model.schema import Chunk

    if not query:
        query = []

    resolved_owner = str(uuid.UUID(owner_id)) if owner_id else str(uuid.uuid4())

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = Path("test_output") / "semantic_unit_eval" / f"{path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        typer.secho("Input file is empty.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    chunker_config = SemanticUnitChunkerConfig(
        level=level,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=800, chunk_overlap=80),
    )
    if force_slices:
        chunker_config.table_small_max_tokens = 1
        chunker_config.code_small_max_tokens = 1
        chunker_config.list_small_max_tokens = 1

    chunker = chunker_config.build()
    raw_chunks = chunker.chunk_text(
        text=text,
        metadata={
            "source_file_id": _project_relative_filename(path),
            "filename": _project_relative_filename(path),
            "owner_id": resolved_owner,
        },
        level=level,
    )

    if not raw_chunks:
        typer.secho("Chunker produced no chunks.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    chunk_ids = [str(uuid.uuid4()) for _ in raw_chunks]
    IndexManager._backfill_anchor_chunk_ids(raw_chunks, chunk_ids)

    role_counts: Dict[str, int] = {}
    unit_counts: Dict[str, int] = {}
    for chunk in raw_chunks:
        meta = chunk.get("metadata") or {}
        role = str(meta.get("chunk_role") or "unknown")
        unit_type = str(meta.get("semantic_unit_type") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        unit_counts[unit_type] = unit_counts.get(unit_type, 0) + 1

    output_chunks_path = output_dir / "chunks.jsonl"
    with output_chunks_path.open("w", encoding="utf-8") as handle:
        for chunk_id, chunk in zip(chunk_ids, raw_chunks):
            meta = (chunk.get("metadata") or {}).copy()
            src = chunk.get("source_metadata") or {}
            if src:
                meta.update(src)
            record = {
                "id": chunk_id,
                "content": chunk.get("content", ""),
                "metadata": meta,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    bm25_dir = output_dir / "bm25_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_builder = BM25BuilderConfig(index_path=str(bm25_dir)).build()

    chunk_objects: List[Chunk] = []
    for chunk_id, chunk in zip(chunk_ids, raw_chunks):
        meta = (chunk.get("metadata") or {}).copy()
        src = chunk.get("source_metadata") or {}
        if src:
            meta.update(src)
        owner = str(meta.get("owner_id") or resolved_owner)
        chunk_objects.append(
            Chunk(
                id=chunk_id,
                owner_id=owner,
                content=str(chunk.get("content") or ""),
                metadata=meta,
            )
        )

    ok = bm25_builder.update_index(chunk_objects)
    if not ok:
        typer.secho("BM25 indexing failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    multipath = MultiPathRetrieverConfig(
        retrievers=[
            TantivyBM25RetrieverConfig(
                index_config=BM25BuilderConfig(index_path=str(bm25_dir)),
                search_kwargs={"k": k, "with_score": True, "use_phrase_query": False},
            )
        ],
        fusion_method="rrf",
        rrf_k=60,
        search_kwargs={"k": k, "with_score": True},
    ).build()

    query_results: List[Dict[str, Any]] = []
    for q in query:
        hits = multipath.invoke(q, k=k, owner_id=resolved_owner)
        payload = {
            "query": q,
            "owner_id": resolved_owner,
            "k": k,
            "hits": len(hits),
            "chunks": serialize_chunks(hits),
        }
        query_results.append(payload)
        (output_dir / f"query_{len(query_results)}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "input": str(path),
        "output_dir": str(output_dir),
        "owner_id": resolved_owner,
        "level": level,
        "force_slices": force_slices,
        "chunker": chunker.get_chunker_info(),
        "chunks_total": len(raw_chunks),
        "role_counts": role_counts,
        "unit_type_counts": unit_counts,
        "queries": [entry["query"] for entry in query_results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"Output: {output_dir}")
    typer.echo(f"- chunks: {output_chunks_path}")
    typer.echo(f"- summary: {output_dir / 'summary.json'}")
    if query:
        typer.echo(f"- queries: {len(query)} (see {output_dir}/query_*.json)")


if __name__ == "__main__":
    app()
