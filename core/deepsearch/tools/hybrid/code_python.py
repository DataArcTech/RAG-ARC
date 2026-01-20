"""Deterministic Python execution tool for math/finance verification inside DeepSearch."""
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from config.core.deepsearch.tool_defaults import (
    CODE_PYTHON_BAD_RUNNER_STDERR_PREVIEW_CHARS,
    CODE_PYTHON_BAD_RUNNER_STDOUT_PREVIEW_CHARS,
    CODE_PYTHON_DEFAULT_ALLOWED_IMPORTS,
    CODE_PYTHON_DEFAULT_EMIT_RESULT_EVIDENCE,
    CODE_PYTHON_DEFAULT_MAX_CODE_CHARS,
    CODE_PYTHON_DEFAULT_MAX_MEMORY_MB,
    CODE_PYTHON_DEFAULT_MAX_RESULT_CHARS,
    CODE_PYTHON_DEFAULT_MAX_STDERR_CHARS,
    CODE_PYTHON_DEFAULT_MAX_STDOUT_CHARS,
    CODE_PYTHON_DEFAULT_TIMEOUT_SECONDS,
    CODE_PYTHON_MIN_TIMEOUT_SECONDS,
    CODE_PYTHON_RUNNER_STDERR_PREVIEW_CHARS,
    CODE_PYTHON_SUMMARY_PREVIEW_CHARS,
    CODE_PYTHON_TRACEBACK_LIMIT,
)
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, SCOPE_OWNER


@dataclass(frozen=True)
class _PythonExecLimits:
    timeout_seconds: float
    max_code_chars: int
    max_stdout_chars: int
    max_stderr_chars: int
    max_result_chars: int
    max_memory_mb: int | None


_RUNNER_SCRIPT = r"""
import ast
import builtins
import contextlib
import io
import json
import os
import platform
import sys
import time
import traceback

try:
    payload = json.load(sys.stdin)
except Exception as exc:
    sys.stdout.write(json.dumps({"exec_status": "failed", "error": {"type": "bad_payload", "message": str(exc)}}))
    raise SystemExit(0)

code = str(payload.get("code") or "")
inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
allowed_imports = payload.get("allowed_imports") if isinstance(payload.get("allowed_imports"), list) else []
allowed_imports = [str(item) for item in allowed_imports if str(item).strip()]
allowed_set = set(allowed_imports)
max_stdout = int(payload.get("max_stdout_chars") or 0)
max_stderr = int(payload.get("max_stderr_chars") or 0)
max_result = int(payload.get("max_result_chars") or 0)
max_memory_mb = payload.get("max_memory_mb")
disable_file_io = bool(payload.get("disable_file_io", True))
traceback_limit = payload.get("traceback_limit")
runner_stderr_preview = payload.get("runner_stderr_preview_chars")

try:
    traceback_limit = int(traceback_limit) if traceback_limit is not None else 8
except Exception:
    traceback_limit = 8
try:
    runner_stderr_preview = int(runner_stderr_preview) if runner_stderr_preview is not None else 1200
except Exception:
    runner_stderr_preview = 1200

def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return text, False
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 1)] + "…", True

class _BoundedWriter(io.TextIOBase):
    def __init__(self, limit: int):
        self._limit = max(0, int(limit))
        self._buf: list[str] = []
        self._count = 0
        self.truncated = False

    def write(self, s: str) -> int:
        text = str(s)
        if self._limit <= 0:
            self._buf.append(text)
            self._count += len(text)
            return len(text)
        remaining = self._limit - self._count
        if remaining <= 0:
            self.truncated = True
            return len(text)
        if len(text) <= remaining:
            self._buf.append(text)
            self._count += len(text)
            return len(text)
        self._buf.append(text[:remaining])
        self._count += remaining
        self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self._buf)


def _apply_resource_limits(max_memory_mb):
    if max_memory_mb is None:
        return
    try:
        mb = int(max_memory_mb)
    except Exception:
        return
    if mb <= 0:
        return
    try:
        import resource

        limit_bytes = mb * 1024 * 1024
        for key in ("RLIMIT_AS", "RLIMIT_DATA"):
            if hasattr(resource, key):
                res_key = getattr(resource, key)
                try:
                    resource.setrlimit(res_key, (limit_bytes, limit_bytes))
                except Exception:
                    continue
    except Exception:
        return


def _validate_imports(code: str, allowed: set[str]) -> list[str]:
    missing: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return missing
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = str(alias.name).split(".", 1)[0]
                if top and top not in allowed:
                    missing.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = str(node.module).split(".", 1)[0]
                if top and top not in allowed:
                    missing.append(top)
    return sorted(set(missing))


def _install_import_guard(allowed: set[str]):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            target = str(name)
        except Exception:
            target = name
        top = str(target).split(".", 1)[0]
        if top and top not in allowed:
            raise ImportError(f"Import blocked: '{top}' is not in allowed_imports")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def _disable_file_io():
    def blocked_open(*args, **kwargs):
        raise PermissionError("file I/O is disabled in code.python")

    builtins.open = blocked_open


def _json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


started = time.perf_counter()
stdout = _BoundedWriter(max_stdout)
stderr = _BoundedWriter(max_stderr)
status = "ok"
error = None
result_obj = None
result_truncated = False

blocked_imports = _validate_imports(code, allowed_set)
if blocked_imports:
    status = "failed"
    error = {"type": "import_blocked", "message": f"Blocked imports: {blocked_imports}", "blocked": blocked_imports}

if status == "ok":
    _apply_resource_limits(max_memory_mb)
    _install_import_guard(allowed_set)
    if disable_file_io:
        _disable_file_io()

    globals_env = {
        "__name__": "__main__",
        "INPUTS": dict(inputs),
    }
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            compiled = compile(code, "<code.python>", "exec")
            exec(compiled, globals_env, globals_env)
        result_obj = globals_env.get("result", None)
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        tb = traceback.format_exc(limit=traceback_limit)
        tb, _tb_truncated = _truncate(tb, runner_stderr_preview)
        error["traceback"] = tb

elapsed_ms = int((time.perf_counter() - started) * 1000)
stdout_text, stdout_trunc = _truncate(stdout.getvalue(), max_stdout)
stderr_text, stderr_trunc = _truncate(stderr.getvalue(), max_stderr)

result_safe = _json_safe(result_obj)
result_text = "" if result_safe is None else (result_safe if isinstance(result_safe, str) else json.dumps(result_safe, ensure_ascii=False, default=str))
result_text, result_trunc = _truncate(result_text, max_result)

payload_out = {
    "exec_status": status,
    "elapsed_ms": elapsed_ms,
    "stdout": stdout_text,
    "stderr": stderr_text,
    "stdout_truncated": stdout_trunc or getattr(stdout, "truncated", False),
    "stderr_truncated": stderr_trunc or getattr(stderr, "truncated", False),
    "result": result_safe,
    "result_text": result_text,
    "result_truncated": result_trunc,
    "error": error,
    "env": {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "allowed_imports": allowed_imports,
    },
}
sys.stdout.write(json.dumps(payload_out, ensure_ascii=False, default=str))
""".strip()


def _fence_python(code: str) -> str:
    return "```python\n" + (code.rstrip() + "\n" if code.strip() else "") + "```"


def _truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


class CodePythonTool(GraphTool):
    """Execute model-provided Python code for deterministic math/finance verification."""

    descriptor = ToolDescriptor(
        name="code.python",
        channel="text",
        description=(
            "Execute deterministic Python for math/finance verification (returns code + outputs). "
            "Evidence: derived computation results (NOT citeable). "
            "Good: provide full code + INPUTS and set result. Bad: placeholder code."
        ),
        speed="medium",
        cost="low",
        strategy_tags=("python", "deterministic", "finance", "verification", EVIDENCE_DERIVED, SCOPE_OWNER),
        profile="X",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.code.python",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "code": {"type": "string", "description": "Python source code to execute."},
                "inputs": {"type": "object", "description": "Optional JSON inputs injected as INPUTS."},
                "purpose": {"type": "string", "description": "Optional brief purpose shown in traces."},
                "expected": {"type": ["object", "string"], "description": "Optional expected result / assertion hint."},
            },
            required_extra_fields=("code",),
        ),
        example_args={
            "question": "计算年化收益率并验算",
            "plan_step": "plan_calc_01",
            "extra": {
                "purpose": "IRR verification",
                "inputs": {"cashflows": [-100, 30, 40, 50]},
                "code": "import numpy as np\ncf = INPUTS['cashflows']\n# numpy.irr is deprecated; implement a simple IRR solver\n\ndef irr(cashflows, guess=0.1):\n    r = guess\n    for _ in range(100):\n        f = sum(cf / (1+r)**i for i, cf in enumerate(cashflows))\n        df = sum(-i*cf / (1+r)**(i+1) for i, cf in enumerate(cashflows[1:], start=1))\n        r2 = r - f/df\n        if abs(r2-r) < 1e-10:\n            return r2\n        r = r2\n    return r\n\nresult = irr(cf)\nprint('irr', result)\n",
            },
        },
    )

    def __init__(
        self,
        *,
        allowed_imports: Iterable[str] = CODE_PYTHON_DEFAULT_ALLOWED_IMPORTS,
        timeout_seconds: float = CODE_PYTHON_DEFAULT_TIMEOUT_SECONDS,
        max_code_chars: int = CODE_PYTHON_DEFAULT_MAX_CODE_CHARS,
        max_stdout_chars: int = CODE_PYTHON_DEFAULT_MAX_STDOUT_CHARS,
        max_stderr_chars: int = CODE_PYTHON_DEFAULT_MAX_STDERR_CHARS,
        max_result_chars: int = CODE_PYTHON_DEFAULT_MAX_RESULT_CHARS,
        max_memory_mb: int | None = CODE_PYTHON_DEFAULT_MAX_MEMORY_MB,
        emit_result_evidence: bool = CODE_PYTHON_DEFAULT_EMIT_RESULT_EVIDENCE,
        disable_file_io: bool = True,
    ):
        self.allowed_imports = [str(item) for item in allowed_imports if str(item).strip()]
        self.limits = _PythonExecLimits(
            timeout_seconds=float(timeout_seconds),
            max_code_chars=int(max_code_chars),
            max_stdout_chars=int(max_stdout_chars),
            max_stderr_chars=int(max_stderr_chars),
            max_result_chars=int(max_result_chars),
            max_memory_mb=(int(max_memory_mb) if max_memory_mb is not None else None),
        )
        self.emit_result_evidence = bool(emit_result_evidence)
        self.disable_file_io = bool(disable_file_io)

    def runtime_hint(self) -> Dict[str, Any]:
        """Return a lightweight environment snapshot for tool_call traces."""

        packages: Dict[str, Any] = {}
        for name in self.allowed_imports:
            try:
                import importlib.util

                packages[name] = bool(importlib.util.find_spec(name))
            except Exception:
                packages[name] = None
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "allowed_imports": list(self.allowed_imports),
            "packages_available": packages,
        }

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = dict(request.extra or {})
        raw_code = str(extra.get("code") or "")
        code = raw_code.strip("\n")
        if not code.strip():
            diagnostics = {"exec_status": "failed", "error": "missing_code"}
            return ToolResult(summary="code.python failed: missing code.", diagnostics=diagnostics, evidences=[])

        if self.limits.max_code_chars > 0 and len(code) > self.limits.max_code_chars:
            truncated = _truncate(code, self.limits.max_code_chars)
            diagnostics = {
                "exec_status": "failed",
                "error": "code_too_large",
                "max_code_chars": self.limits.max_code_chars,
                "code_preview": truncated,
            }
            return ToolResult(
                summary=f"code.python failed: code exceeds max_code_chars={self.limits.max_code_chars}.",
                diagnostics=diagnostics,
                evidences=[],
            )

        inputs = extra.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}

        payload = {
            "code": code,
            "inputs": inputs,
            "allowed_imports": list(self.allowed_imports),
            "max_stdout_chars": self.limits.max_stdout_chars,
            "max_stderr_chars": self.limits.max_stderr_chars,
            "max_result_chars": self.limits.max_result_chars,
            "max_memory_mb": self.limits.max_memory_mb,
            "disable_file_io": self.disable_file_io,
            "traceback_limit": int(CODE_PYTHON_TRACEBACK_LIMIT),
            "runner_stderr_preview_chars": int(CODE_PYTHON_RUNNER_STDERR_PREVIEW_CHARS),
        }
        start = time.perf_counter()
        outcome = self._run_subprocess(payload)
        latency_ms = int((time.perf_counter() - start) * 1000)

        exec_status = str(outcome.get("exec_status") or "failed")
        stdout = str(outcome.get("stdout") or "")
        stderr = str(outcome.get("stderr") or "")
        result_text = str(outcome.get("result_text") or "")
        error = outcome.get("error")

        summary_parts: List[str] = [f"exec_status={exec_status}"]
        if exec_status == "ok":
            if result_text:
                summary_parts.append("result=" + _truncate(result_text, int(CODE_PYTHON_SUMMARY_PREVIEW_CHARS)))
            elif stdout.strip():
                summary_parts.append("stdout=" + _truncate(stdout.strip(), int(CODE_PYTHON_SUMMARY_PREVIEW_CHARS)))
        else:
            if isinstance(error, dict) and error.get("message"):
                summary_parts.append(
                    "error=" + _truncate(str(error.get("message")), int(CODE_PYTHON_SUMMARY_PREVIEW_CHARS))
                )
            elif stderr.strip():
                summary_parts.append("stderr=" + _truncate(stderr.strip(), int(CODE_PYTHON_SUMMARY_PREVIEW_CHARS)))
        summary_parts.append(f"latency_ms={latency_ms}")
        summary = "code.python: " + " ".join(summary_parts)

        diagnostics = dict(outcome)
        diagnostics["latency_ms"] = latency_ms
        diagnostics["limits"] = {
            "timeout_seconds": self.limits.timeout_seconds,
            "max_code_chars": self.limits.max_code_chars,
            "max_stdout_chars": self.limits.max_stdout_chars,
            "max_stderr_chars": self.limits.max_stderr_chars,
            "max_result_chars": self.limits.max_result_chars,
            "max_memory_mb": self.limits.max_memory_mb,
            "disable_file_io": self.disable_file_io,
        }
        diagnostics["code"] = code
        diagnostics["code_block"] = _fence_python(code)
        diagnostics["purpose"] = extra.get("purpose")
        diagnostics["expected"] = extra.get("expected")
        diagnostics.setdefault("env", {})
        diagnostics["env"].setdefault("python_version", platform.python_version())
        diagnostics["env"].setdefault("platform", platform.platform())
        diagnostics["env"]["allowed_imports"] = list(self.allowed_imports)

        evidences: List[EvidenceChunk] = []
        if self.emit_result_evidence:
            evidence_text = self._build_result_evidence_text(
                exec_status=exec_status,
                stdout=stdout,
                stderr=stderr,
                result_text=result_text,
                error=error,
            )
            if evidence_text.strip():
                evidences.append(
                    EvidenceChunk(
                        chunk_id=derived_chunk_id(
                            tool_name=self.descriptor.name,
                            plan_step=request.plan_step,
                            content=evidence_text,
                            label="result",
                        ),
                        source=self.descriptor.name,
                        content=evidence_text,
                        kind=EVIDENCE_KIND_DERIVED,
                        score=None,
                        provenance={
                            "tool_name": self.descriptor.name,
                            "plan_step": request.plan_step,
                            "exec_status": exec_status,
                        },
                    )
                )

        return ToolResult(summary=summary, diagnostics=diagnostics, evidences=evidences)

    def _run_subprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        timeout = max(float(CODE_PYTHON_MIN_TIMEOUT_SECONDS), float(self.limits.timeout_seconds))
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER_SCRIPT],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "exec_status": "failed",
                "error": {"type": "timeout", "message": f"Execution exceeded timeout_seconds={timeout}"},
                "stdout": "",
                "stderr": "",
                "result": None,
                "result_text": "",
                "env": {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "allowed_imports": list(self.allowed_imports),
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "exec_status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "stdout": "",
                "stderr": "",
                "result": None,
                "result_text": "",
                "env": {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "allowed_imports": list(self.allowed_imports),
                },
            }

        raw = (completed.stdout or "").strip()
        parsed: Dict[str, Any] = {}
        try:
            parsed_obj = json.loads(raw) if raw else {}
            parsed = parsed_obj if isinstance(parsed_obj, dict) else {"exec_status": "failed", "error": "bad_runner_output"}
        except Exception:
            parsed = {
                "exec_status": "failed",
                "error": {
                    "type": "bad_runner_output",
                    "message": "Runner did not return JSON",
                    "stdout_preview": _truncate(raw, int(CODE_PYTHON_BAD_RUNNER_STDOUT_PREVIEW_CHARS)),
                    "stderr_preview": _truncate(
                        (completed.stderr or ""), int(CODE_PYTHON_BAD_RUNNER_STDERR_PREVIEW_CHARS)
                    ),
                },
            }
        runner_stderr = (completed.stderr or "").strip()
        if runner_stderr:
            meta = parsed.get("error")
            if not isinstance(meta, dict):
                meta = {"type": "runner_error", "message": str(meta)}
            meta.setdefault(
                "runner_stderr", _truncate(runner_stderr, int(CODE_PYTHON_RUNNER_STDERR_PREVIEW_CHARS))
            )
            parsed["error"] = meta
            parsed["exec_status"] = "failed"
        return parsed

    @staticmethod
    def _build_result_evidence_text(
        *,
        exec_status: str,
        stdout: str,
        stderr: str,
        result_text: str,
        error: Any,
    ) -> str:
        lines: List[str] = [f"code.python exec_status={exec_status}"]
        if result_text.strip():
            lines.append("result:")
            lines.append(result_text.strip())
        if stdout.strip():
            lines.append("stdout:")
            lines.append(stdout.strip())
        if exec_status != "ok":
            if isinstance(error, dict):
                msg = error.get("message") or error.get("type")
                if msg:
                    lines.append("error:")
                    lines.append(str(msg))
            if stderr.strip():
                lines.append("stderr:")
                lines.append(stderr.strip())
        return "\n".join(lines).strip()
