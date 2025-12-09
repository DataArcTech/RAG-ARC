import uuid
from typer.testing import CliRunner

from cli.bootstrap import CLIContext
import cli.rag as rag


class _StubKnowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, uuid.UUID]] = []

    async def mark_file_deleted_cli(self, file_id: str, owner_id: uuid.UUID):
        self.calls.append(("mark", file_id, owner_id))
        return {"status": "marked"}


def _fake_ctx() -> CLIContext:
    return CLIContext(owner_id=uuid.uuid4())


def test_cli_delete_file_marks_only(monkeypatch):
    runner = CliRunner()
    stub = _StubKnowledge()

    monkeypatch.setattr(rag, "_get_knowledge_module", lambda: stub)
    monkeypatch.setattr(rag, "initialize", lambda owner_id=None: _fake_ctx())

    result = runner.invoke(rag.app, ["delete-file", "file-123"])

    assert result.exit_code == 0
    assert len(stub.calls) == 1
    assert stub.calls[0][0] == "mark"
    assert stub.calls[0][1] == "file-123"
    assert "marked as deleted" in result.stdout
