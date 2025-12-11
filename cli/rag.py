import asyncio
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import typer

from application.knowledge.module import Knowledge
from application.rag_inference.cli_module import (
    PipelineArtifacts,
    RAGInferenceCLIModule,
)
from application.rag_inference.module import RAGInference
from config.application.deepsearch_tool_server_config import load_tool_server_config
from cli import types
from cli.bootstrap import CLIContext, initialize
from encapsulation.data_model.orm_models import FileStatus
from framework.register import Register
from dotenv import load_dotenv

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


def _print_summary(summary: types.PipelineSummary, output_json: bool) -> None:
    if output_json:
        typer.echo(json.dumps(summary.__dict__, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))
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


def _run_pipeline(
    ctx: CLIContext,
    query: str,
    return_subgraph: bool,
    skip_llm: bool,
    output_json: bool,
    mode: str = "multipath",
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
    summary = types.PipelineSummary.from_artifacts(
        owner_id=str(ctx.owner_id),
        artifacts=artifacts,
    )
    _print_summary(summary, output_json=output_json)
    return artifacts


def _gather_files(folder: Path, pattern: str, recursive: bool) -> List[Path]:
    candidates = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return [path for path in candidates if path.is_file()]


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOOL_SERVER_CONFIG = REPO_ROOT / "config/json_configs/deepsearch_tool_mcp_server.json"


def _ingest_single_file(path: Path, knowledge: Knowledge, owner_id) -> bool:
    typer.echo(f"\n→ Ingesting {path}")
    try:
        file_bytes = path.read_bytes()
    except OSError as exc:
        typer.secho(f"  ! Failed to read file: {exc}", fg=typer.colors.RED)
        return False

    content_type = _guess_content_type(path)
    try:
        file_id = knowledge.file_storage.upload_file(
            filename=path.name,
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
    path: str = typer.Option("mcp/tools", help="Path for HTTP/SSE transports."),
) -> None:
    """Launch the DeepSearch tool MCP server for external agents."""

    load_dotenv()
    server = _build_tool_server_from_config()
    _print_tool_catalog(server)
    if transport == "stdio":
        asyncio.run(server.run_stdio_async())
        return
    if transport == "sse":
        asyncio.run(server.run_sse_async(host=host, port=port, path=path))
        return
    if transport in {"streamable", "streamable-http"}:
        asyncio.run(server.run_streamable_http_async(host=host, port=port, path=path))
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
    path: str = typer.Option("mcp/chat", help="Path for HTTP/SSE transports."),
) -> None:
    """Launch the account/chat MCP server for external agents."""

    load_dotenv()
    chat_server = _get_chat_mcp_server()
    _print_chat_mcp_catalog(chat_server)
    if transport == "stdio":
        asyncio.run(chat_server.run_stdio_async())
        return
    if transport == "sse":
        asyncio.run(chat_server.run_sse_async(host=host, port=port, path=path))
        return
    if transport in {"streamable", "streamable-http"}:
        asyncio.run(chat_server.run_streamable_http_async(host=host, port=port, path=path))
        return
    raise typer.BadParameter("transport must be one of: stdio, sse, streamable-http")

@app.command()
def chat(
    query: str = typer.Argument(..., help="User question to run through RAG pipeline."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(False, "--subgraph", help="Export subgraph metadata."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Run the full RAG pipeline and stream answer to terminal."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph,
        skip_llm=False,
        output_json=output_json,
    )


@app.command()
def pipeline(
    query: str = typer.Argument(..., help="User question to run through RAG pipeline."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(False, "--subgraph", help="Export subgraph metadata."),
    skip_llm: bool = typer.Option(True, "--skip-llm/--with-llm", help="Skip LLM call for faster debugging."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Run pipeline and inspect intermediate artifacts (defaults to skipping LLM)."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph,
        skip_llm=skip_llm,
        output_json=output_json,
    )


@app.command("graph-qa")
def graph_qa(
    query: str = typer.Argument(..., help="Question answered purely via graph retriever."),
    owner_id: str = typer.Option(None, help="Optional owner UUID to filter retrieval results."),
    return_subgraph: bool = typer.Option(True, "--subgraph/--no-subgraph", help="Export subgraph metadata."),
    skip_llm: bool = typer.Option(False, "--skip-llm/--with-llm", help="Skip LLM call for debugging."),
    output_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Query only the graph retriever and optionally export the subgraph."""
    ctx = initialize(owner_id=owner_id)
    _run_pipeline(
        ctx=ctx,
        query=query,
        return_subgraph=return_subgraph,
        skip_llm=skip_llm,
        output_json=output_json,
        mode="graph",
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
    success = _ingest_single_file(path, knowledge, ctx.owner_id)
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
    files = _gather_files(folder, pattern, recursive)
    if limit is not None and limit > 0:
        files = files[:limit]

    if not files:
        typer.secho("No files matched the given pattern.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(files)} file(s) in {folder}")
    succeeded = 0
    for path in files:
        if _ingest_single_file(path, knowledge, ctx.owner_id):
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
) -> None:
    """Trigger indexing for existing files."""
    if not file_ids:
        typer.secho("Provide at least one FILE_ID.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    ctx = initialize(owner_id=owner_id)
    knowledge = _get_knowledge_module()
    try:
        message = asyncio.run(knowledge.trigger_indexing(file_ids, ctx.owner_id))
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


if __name__ == "__main__":
    app()
