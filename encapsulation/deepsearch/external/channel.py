"""External channel orchestrator for Tavily provider or MCP tools."""
import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext
from core.deepsearch.trace import emit_trace
from core.utils.json_safe import json_safe


class ExternalSearchChannel:
    """Trigger external web/search tools when the gap detector flags missing coverage."""

    def __init__(
        self,
        tool_manager,
        config: Optional[Dict[str, Any]] = None,
        *,
        telemetry_client: Any | None = None,
    ):
        self.tool_manager = tool_manager
        self.config = self._model_to_dict(config)
        self.telemetry_client = telemetry_client
        self.max_rounds = max(1, int(self.config.get("max_rounds", 2)))
        provider = self.config.get("default_provider") or os.getenv("DEEPSEARCH_WEB_PROVIDER")
        self.default_provider = self._normalize_provider(provider)
        self._context_limit = int(self.config.get("context_window_limit", 12))
        self._tavily_timeout = float(self.config.get("http_timeout", 20))
        self._tavily_max_results = int(self.config.get("max_results", 5))
        try:
            timeout = float(self.config.get("tool_timeout_seconds", 45.0))
        except (TypeError, ValueError):
            timeout = 45.0
        self._tool_timeout = max(0.0, timeout)
        self._cache_mode = (os.getenv("DEEPSEARCH_EXTERNAL_CACHE_MODE") or self.config.get("cache_mode") or "auto").strip().lower()
        self._cache_dir = os.getenv("DEEPSEARCH_EXTERNAL_CACHE_DIR") or self.config.get("cache_dir")

    async def run(
        self,
        tasks: List[Dict[str, Any]],
        *,
        reasoning_trace: Optional[Dict[str, Any]] = None,
        gap_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute pending external steps and return normalized evidence chunks."""

        if not self._is_enabled() or not tasks:
            return {"evidences": [], "logs": []}

        trace = reasoning_trace or {}
        question = (trace.get("question") or "").strip()
        graph_context = self._resolve_graph_context(trace.get("graph_context") or {})
        context_evidences = self._coerce_evidences(trace.get("evidences") or [])
        coverage_metrics = trace.get("coverage_metrics") or {}
        run_id = None
        artifact_dir: str | None = None
        if graph_context and isinstance(getattr(graph_context, "metadata", None), dict):
            run_id = (graph_context.metadata or {}).get("run_id")
            artifact_dir = (graph_context.metadata or {}).get("artifact_dir")

        cache_mode, cache_root = self._resolve_cache_context(artifact_dir)

        outputs: List[Dict[str, Any]] = []
        logs: List[Dict[str, Any]] = []
        for idx, task in enumerate(tasks):
            if idx >= self.max_rounds:
                break
            provider = self._task_provider(task)
            log_entry = self._build_log(task=task, provider=provider, gap_result=gap_result)
            call_id = uuid.uuid4().hex
            query = self._task_query(task, default=question or "")
            tool_name = str(task.get("tool") or "web.search")
            cache_key = self._cache_key(provider=provider, tool_name=tool_name, query=query, task=task)
            if cache_root is not None and cache_mode in {"replay", "auto"}:
                replay = self._load_cache(cache_root, cache_key)
                if replay is not None:
                    chunks, diagnostics, event = replay
                    log_entry["status"] = "replay"
                    log_entry["evidence_count"] = len(chunks)
                    log_entry["latency_ms"] = 0
                    log_entry["cache_key"] = cache_key
                    logs.append(log_entry)
                    try:
                        await emit_trace(
                            "tool_response",
                            json.dumps(
                                json_safe(
                                    {
                                        "call_id": call_id,
                                        "tool_name": tool_name,
                                        "provider": provider,
                                        "query": query,
                                        "status": "replay",
                                        "cache_key": cache_key,
                                        "evidence_count": len(chunks),
                                        "evidences": chunks[:8],
                                        "diagnostics": diagnostics,
                                    }
                                ),
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                            meta={"call_id": call_id, "tool_name": tool_name, "provider": provider, "ok": True, "cache_mode": cache_mode, "cache_key": cache_key},
                        )
                    except Exception:
                        pass
                    outputs.extend(chunks)
                    continue
            try:
                await emit_trace(
                    "tool_call",
                    json.dumps(
                        json_safe(
                            {
                                "call_id": call_id,
                                "tool_name": tool_name,
                                "provider": provider,
                                "plan_step": log_entry.get("plan_step") or task.get("step_id") or task.get("step") or "external",
                                "query": query,
                                "max_results": self._tavily_max_results,
                                "context_evidence_count": len(context_evidences),
                                "coverage_metrics": coverage_metrics,
                                "cache_mode": cache_mode,
                                "cache_key": cache_key,
                            }
                        ),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    meta={"call_id": call_id, "tool_name": tool_name, "provider": provider, "cache_mode": cache_mode, "cache_key": cache_key},
                )
            except Exception:
                pass
            task_start = time.perf_counter()
            try:
                chunks, diagnostics, event = await self._execute_task(
                    task,
                    provider=provider,
                    question=question,
                    graph_context=graph_context,
                    context_evidences=context_evidences,
                    coverage_metrics=coverage_metrics,
                    gap_result=gap_result,
                )
            except asyncio.TimeoutError:
                log_entry["status"] = "timeout"
                log_entry["latency_ms"] = int((time.perf_counter() - task_start) * 1000)
                self._log_event(
                    event="timeout",
                    task=task,
                    provider=provider,
                    run_id=run_id,
                    diagnostics={"error": "timeout"},
                )
                logs.append(log_entry)
                try:
                    await emit_trace(
                        "tool_response",
                        json.dumps(
                            json_safe(
                                {
                                    "call_id": call_id,
                                    "tool_name": tool_name,
                                    "provider": provider,
                                    "query": query,
                                    "error": "timeout",
                                }
                            ),
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        meta={"call_id": call_id, "tool_name": tool_name, "provider": provider, "ok": False},
                    )
                except Exception:
                    pass
                continue
            except Exception as exc:  # pragma: no cover - defensive telemetry path
                log_entry["status"] = "error"
                log_entry["error"] = str(exc)
                log_entry["latency_ms"] = int((time.perf_counter() - task_start) * 1000)
                self._log_event(
                    event="error",
                    task=task,
                    provider=provider,
                    run_id=run_id,
                    message=str(exc),
                )
                logs.append(log_entry)
                try:
                    await emit_trace(
                        "tool_response",
                        json.dumps(
                            json_safe(
                                {
                                    "call_id": call_id,
                                    "tool_name": tool_name,
                                    "provider": provider,
                                    "query": query,
                                    "error": str(exc),
                                }
                            ),
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        meta={"call_id": call_id, "tool_name": tool_name, "provider": provider, "ok": False},
                    )
                except Exception:
                    pass
                continue
            if cache_root is not None and cache_mode in {"record", "auto"}:
                self._save_cache(
                    cache_root,
                    cache_key,
                    record={
                        "schema_version": 1,
                        "run_id": run_id,
                        "provider": provider,
                        "tool_name": tool_name,
                        "query": query,
                        "task": json_safe(task),
                        "response": {
                            "chunks": json_safe(chunks),
                            "diagnostics": json_safe(diagnostics),
                            "event": event,
                        },
                        "created_at_ms": int(time.time() * 1000),
                    },
                )
            log_entry["status"] = "ok"
            log_entry["evidence_count"] = len(chunks)
            log_entry["latency_ms"] = int((time.perf_counter() - task_start) * 1000)
            logs.append(log_entry)
            self._log_event(
                event=event,
                task=task,
                provider=provider,
                run_id=run_id,
                diagnostics=diagnostics,
                latency_ms=log_entry["latency_ms"],
                evidence_count=log_entry["evidence_count"],
                status=log_entry["status"],
            )
            try:
                await emit_trace(
                    "tool_response",
                    json.dumps(
                        json_safe(
                            {
                                "call_id": call_id,
                                "tool_name": tool_name,
                                "provider": provider,
                                "query": query,
                                "latency_ms": log_entry.get("latency_ms"),
                                "evidence_count": len(chunks),
                                "evidences": chunks[:8],
                                "diagnostics": diagnostics,
                            }
                        ),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    meta={"call_id": call_id, "tool_name": tool_name, "provider": provider, "ok": True, "cache_mode": cache_mode, "cache_key": cache_key},
                )
            except Exception:
                pass
            outputs.extend(chunks)
        return {"evidences": outputs, "logs": logs}

    # ------------------------------------------------------------------
    def _resolve_cache_context(self, artifact_dir: str | None) -> tuple[str, Path | None]:
        mode = (self._cache_mode or "off").strip().lower()
        if mode not in {"off", "record", "replay", "auto"}:
            mode = "off"

        directory = self._cache_dir
        if not directory and artifact_dir:
            directory = str(Path(str(artifact_dir)) / "external_cache")
        if not directory:
            return mode, None
        try:
            root = Path(str(directory)).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return mode, root
        except Exception:
            return mode, None

    @staticmethod
    def _cache_key(*, provider: str, tool_name: str, query: str, task: Dict[str, Any]) -> str:
        stable_task = {
            "tool": task.get("tool"),
            "tool_args": task.get("tool_args") if isinstance(task.get("tool_args"), dict) else {},
            "metadata": task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            "channel": task.get("channel"),
            "requires_external": task.get("requires_external"),
        }
        payload = {
            "provider": str(provider or "").strip().lower(),
            "tool_name": str(tool_name or "").strip().lower(),
            "query": str(query or "").strip(),
            "task": json_safe(stable_task),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _load_cache(root: Path, key: str) -> tuple[List[Dict[str, Any]], Dict[str, Any], str] | None:
        path = root / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        response = record.get("response") if isinstance(record, dict) else None
        if not isinstance(response, dict):
            return None
        chunks = response.get("chunks") or []
        diagnostics = response.get("diagnostics") or {}
        event = response.get("event") or "replay"
        if not isinstance(chunks, list):
            chunks = []
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        return [item for item in chunks if isinstance(item, dict)], diagnostics, str(event)

    @staticmethod
    def _save_cache(root: Path, key: str, *, record: Dict[str, Any]) -> None:
        path = root / f"{key}.json"
        try:
            path.write_text(json.dumps(json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    # ------------------------------------------------------------------
    async def _execute_task(
        self,
        task: Dict[str, Any],
        *,
        provider: str,
        question: str,
        graph_context: Optional[GraphQueryContext],
        context_evidences: List[EvidenceChunk],
        coverage_metrics: Dict[str, Any],
        gap_result: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        tool = task.get("tool") or "web.search"
        provider = self._normalize_provider(provider or self.default_provider)
        if provider == "tavily":
            chunks, diagnostics = await self._execute_with_provider(task, provider=provider, question=question)
            return chunks, diagnostics, provider
        if provider in {"mcp", "tool"}:
            chunks, diagnostics = await self._execute_with_tool_manager(
                task,
                tool_name=tool,
                provider=provider,
                question=question,
                graph_context=graph_context,
                context_evidences=context_evidences,
                coverage_metrics=coverage_metrics,
                gap_result=gap_result,
            )
            return chunks, diagnostics, tool
        # Default: attempt tool manager first; fall back to Tavily HTTP provider.
        try:
            chunks, diagnostics = await self._execute_with_tool_manager(
                task,
                tool_name=tool,
                provider=provider,
                question=question,
                graph_context=graph_context,
                context_evidences=context_evidences,
                coverage_metrics=coverage_metrics,
                gap_result=gap_result,
            )
            return chunks, diagnostics, tool
        except Exception:
            chunks, diagnostics = await self._execute_with_provider(task, provider="tavily", question=question)
            return chunks, diagnostics, "tavily"

    async def _execute_with_provider(
        self,
        task: Dict[str, Any],
        *,
        provider: str,
        question: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if provider == "tavily":
            return await self._execute_with_tavily(task, question=question)
        raise RuntimeError(f"Unsupported provider: {provider}")

    def _decorate_chunk(
        self,
        chunk: Dict[str, Any],
        task: Dict[str, Any],
        provider: str,
        tool_name: str,
        *,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        provenance = dict(chunk.get("provenance") or {})
        provenance.setdefault("provider", provider)
        provenance.setdefault("tool", tool_name)
        provenance.setdefault("step_id", task.get("step_id"))
        provenance.setdefault("query", self._task_query(task, default=""))
        if diagnostics:
            provenance.setdefault("diagnostics", diagnostics)
        chunk["provenance"] = provenance
        return chunk

    def _build_tool_payload(
        self,
        task: Dict[str, Any],
        *,
        question: str,
        graph_context: Optional[GraphQueryContext],
        context_evidences: List[EvidenceChunk],
        coverage_metrics: Dict[str, Any],
        gap_result: Optional[Dict[str, Any]],
        provider: str,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "plan_step": task.get("step_id") or task.get("step") or "external",
            "context_evidences": self._context_window(context_evidences),
            "extra": {
                "provider": provider,
                "gap_result": gap_result or {},
                "coverage_metrics": coverage_metrics or {},
                "task": task,
            },
            "graph_context": graph_context.model_dump(exclude_none=True) if graph_context else None,
            "coverage_metrics": coverage_metrics or {},
        }

    async def _execute_with_tool_manager(
        self,
        task: Dict[str, Any],
        *,
        tool_name: str,
        provider: str,
        question: str,
        graph_context: Optional[GraphQueryContext],
        context_evidences: List[EvidenceChunk],
        coverage_metrics: Dict[str, Any],
        gap_result: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.tool_manager:
            raise KeyError("tool_manager is not configured")
        payload = self._build_tool_payload(
            task,
            question=question,
            graph_context=graph_context,
            context_evidences=context_evidences,
            coverage_metrics=coverage_metrics,
            gap_result=gap_result,
            provider=provider,
        )
        invocation = self.tool_manager.invoke(tool_name, payload=payload)
        if self._tool_timeout > 0:
            result = await asyncio.wait_for(invocation, timeout=self._tool_timeout)
        else:
            result = await invocation
        evidences = [self._decorate_chunk(chunk.model_dump(), task, provider, tool_name) for chunk in result.evidences]
        if not evidences and result.summary:
            evidences.append(
                self._decorate_chunk(
                    {
                        "chunk_id": f"external-{tool_name}-{uuid.uuid4().hex[:8]}",
                        "source": result.tool_name,
                        "content": result.summary,
                    },
                    task,
                    provider,
                    tool_name,
                    diagnostics=result.diagnostics,
                )
            )
        return evidences, result.diagnostics or {}

    async def _execute_with_tavily(
        self, task: Dict[str, Any], *, question: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        api_key = self._resolve_tavily_key()
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured; external search remains disabled.")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("httpx is required for Tavily search; install the 'dev' extras.") from exc

        query = self._task_query(task, default=question)
        if not query:
            return [], {"result_count": 0}

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": self._tavily_max_results,
        }
        async with httpx.AsyncClient(timeout=self._tavily_timeout) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()

        evidences = self._normalize_tavily_results(task, data, query)
        return evidences, {"result_count": len(evidences)}

    def _normalize_tavily_results(self, task: Dict[str, Any], data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        results = data.get("results") or []
        evidences: List[Dict[str, Any]] = []
        for idx, item in enumerate(results):
            snippet = (item.get("content") or "").strip()
            if not snippet:
                continue
            chunk = {
                "chunk_id": f"tavily-{task.get('step_id', 'ext')}-{idx}",
                "source": "web.tavily",
                "content": f"{item.get('title') or 'Result'}\n{snippet}",
                "score": item.get("score"),
                "provenance": {
                    "provider": "tavily",
                    "url": item.get("url"),
                    "step_id": task.get("step_id"),
                    "query": query,
                    "snippet_rank": idx + 1,
                },
            }
            evidences.append(chunk)
        if not evidences and data.get("answer"):
            evidences.append(
                {
                    "chunk_id": f"tavily-{task.get('step_id', 'ext')}-answer",
                    "source": "web.tavily",
                    "content": data["answer"],
                    "provenance": {
                        "provider": "tavily",
                        "url": None,
                        "step_id": task.get("step_id"),
                        "query": query,
                        "snippet_rank": 0,
                    },
                }
            )
        return evidences

    @staticmethod
    def _resolve_graph_context(payload: Dict[str, Any]) -> Optional[GraphQueryContext]:
        if not payload:
            return None
        if isinstance(payload, GraphQueryContext):
            return payload
        try:
            return GraphQueryContext.model_validate(payload)
        except Exception:
            adapter_name = payload.get("adapter_name") or "unknown"
            return GraphQueryContext(adapter_name=adapter_name, question=payload.get("question"))

    def _context_window(self, evidences: Sequence[EvidenceChunk]) -> List[Dict[str, Any]]:
        if not evidences:
            return []
        window = evidences
        limit = max(0, self._context_limit)
        if limit:
            window = evidences[-limit:]
        return [chunk.model_dump() for chunk in window]

    @staticmethod
    def _coerce_evidences(raw: Iterable[Any]) -> List[EvidenceChunk]:
        payloads: List[EvidenceChunk] = []
        for item in raw:
            if isinstance(item, EvidenceChunk):
                payloads.append(item)
                continue
            if isinstance(item, dict):
                try:
                    payloads.append(EvidenceChunk.model_validate(item))
                except Exception:
                    continue
        return payloads

    def _task_provider(self, task: Dict[str, Any]) -> str:
        metadata = task.get("metadata") or {}
        provider = metadata.get("provider") or metadata.get("external_provider")
        if isinstance(provider, str) and provider.strip():
            return self._normalize_provider(provider)
        return self._normalize_provider(self.default_provider)

    @staticmethod
    def _normalize_provider(provider: Any) -> str:
        token = str(provider or "").strip().lower()
        if token in {"mcp", "tool"}:
            return token
        return "tavily"

    def _task_query(self, task: Dict[str, Any], *, default: str) -> str:
        metadata = task.get("metadata") or {}
        for key in ("query", "search_query", "question"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        tool_args = task.get("tool_args") or {}
        for key in ("query", "question", "prompt"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        description = task.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return default

    def _resolve_tavily_key(self) -> Optional[str]:
        key = os.getenv("TAVILY_API_KEY")
        if key:
            return key.strip()
        return None

    def _is_enabled(self) -> bool:
        """Resolve enablement (config SoT; env overrides)."""

        env = self._read_env_bool("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED")
        config_enabled = bool(self.config.get("enabled"))
        if env is not None:
            return bool(env)
        return config_enabled

    @staticmethod
    def _read_env_bool(name: str) -> Optional[bool]:
        raw = os.getenv(name)
        if raw is None:
            return None
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return None

    def _log_event(
        self,
        event: str,
        *,
        task: Optional[Dict[str, Any]],
        provider: Optional[str],
        run_id: Optional[str] = None,
        **extras: Any,
    ) -> None:
        if not self.telemetry_client:
            return
        log_method = getattr(self.telemetry_client, "log_external_channel", None) or getattr(
            self.telemetry_client, "log_external_search", None
        )
        if not callable(log_method):
            return
        payload = {
            "event": event,
            "run_id": run_id,
            "provider": provider,
            "step_id": (task or {}).get("step_id"),
        }
        if extras:
            payload.update(extras)
        try:
            log_method(payload=payload)
        except Exception:  # pragma: no cover - telemetry guard
            return

    @staticmethod
    def _build_log(
        *,
        task: Dict[str, Any],
        provider: Optional[str],
        gap_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = task.get("metadata") or {}
        tool_args = task.get("tool_args") or {}
        return {
            "step_id": task.get("step_id"),
            "provider": provider,
            "tool": task.get("tool"),
            "query": metadata.get("query") or tool_args.get("query"),
            "gap_reason": gap_result.get("reason") if gap_result else None,
            "missing_topics": (gap_result or {}).get("missing_topics") or [],
        }

    @staticmethod
    def _model_to_dict(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "model_dump"):
            try:
                return config.model_dump()
            except TypeError:
                return config.model_dump(exclude_none=True)
        return dict(getattr(config, "__dict__", {}))
