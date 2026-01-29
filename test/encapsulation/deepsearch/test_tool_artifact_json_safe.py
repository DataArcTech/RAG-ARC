import json
import uuid
from pathlib import Path

from encapsulation.data_model.deepsearch import EvidenceChunk, ToolResultPayload
from encapsulation.deepsearch.telemetry import LoggingTelemetryClient
from encapsulation.deepsearch.tooling.manager import DeepSearchToolManager


def test_tool_artifact_persist_handles_uuid(tmp_path: Path) -> None:
    tool_configs = {
        "enable_builtin_tools": False,
        "max_remote_evidences": 1,
        "max_remote_context_chars": 1024,
        "artifact_dir": str(tmp_path),
        "enabled_tools": {},
        "remote_tools": {},
    }
    manager = DeepSearchToolManager(tool_configs=tool_configs, telemetry_client=LoggingTelemetryClient())

    payload = ToolResultPayload(
        tool_name="explore",
        namespace="rag-arc.deepsearch.tools.explore",
        channel="graph",
        profile="X",
        determinism="hybrid",
        summary="ok",
        evidences=[
            EvidenceChunk(
                chunk_id="c1",
                source="test",
                content="hello",
                provenance={"owner_id": uuid.uuid4()},
            )
        ],
        diagnostics={"file_id": uuid.uuid4()},
        think_notes=[],
    )

    artifact_path = manager._persist_artifact("explore", payload)
    assert artifact_path
    loaded = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    assert loaded["tool_name"] == "explore"
    # UUIDs should be serialized to strings.
    assert isinstance(loaded["diagnostics"]["file_id"], str)
    assert isinstance(loaded["evidences"][0]["provenance"]["owner_id"], str)

