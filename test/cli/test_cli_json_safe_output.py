import json
import shutil
import uuid

from cli.rag import _emit_json_output


def test_emit_json_output_handles_uuid(tmp_path) -> None:  # noqa: ANN001
    # Use a unique owner id so test artifacts are isolated under local/cli/<owner>/.
    owner_id = str(uuid.uuid4())
    payload = {
        "run_id": uuid.uuid4(),
        "nested": {"owner_id": uuid.uuid4()},
    }

    path = _emit_json_output(payload, owner_id=owner_id, prefix="test_json_safe", print_to_console=False)
    try:
        assert path is not None
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data["run_id"], str)
        assert isinstance(data["nested"]["owner_id"], str)
    finally:
        if path is not None:
            shutil.rmtree(path.parent, ignore_errors=True)
