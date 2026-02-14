import logging
import time
from typing import Any, Dict, Optional

from application.deepsearch.artifacts import DeepSearchArtifactStore
from core.constants.io_namespaces import DEEPSEARCH_ARTIFACTS_NAMESPACE
from core.utils.json_safe import json_safe
from framework.virtual_paths import IO_PATH_PREFIX, io_key

logger = logging.getLogger(__name__)


class DeepSearchServiceArtifactsMixin:
    def _resolve_artifacts_config(self) -> Dict[str, Any]:
        from config.application.deepsearch_config import ArtifactsConfig

        raw = None
        if isinstance(self.config, dict):
            raw = self.config.get("artifacts")
        model = ArtifactsConfig.model_validate(raw or {})
        return model.model_dump()

    def _config_fingerprint(self) -> str:
        if not isinstance(self.config, dict) or not str(self.config.get("fingerprint") or "").strip():
            raise ValueError("DeepSearchService config fingerprint is required")
        return str(self.config["fingerprint"])

    @staticmethod
    def _coerce_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "model_dump"):
            try:
                return config.model_dump()
            except TypeError:
                return config.model_dump(exclude_none=True)
        if hasattr(config, "__dict__"):
            return {key: value for key, value in vars(config).items() if not key.startswith("_")}
        return {"value": config}

    def _resolve_experiment_dir(self) -> Optional[str]:
        candidate = None
        if isinstance(self.config, dict):
            candidate = self.config.get("experiment_output_dir")
        directory = str(candidate or "").strip()
        if not directory:
            return None
        if not directory.startswith(IO_PATH_PREFIX):
            raise ValueError("DeepSearchService config.experiment_output_dir must be an io:// virtual path")
        return directory

    def _resolve_artifact_store(self) -> DeepSearchArtifactStore | None:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config is required to resolve artifact store")
        artifacts_cfg = self._resolve_artifacts_config()
        if not bool(artifacts_cfg.get("enabled", True)):
            return None

        from app_registration import registrator

        io_manager = registrator.get_object("io_manager")
        if io_manager is None:
            raise RuntimeError("io_manager is required for DeepSearch artifacts")

        configured = str(self.config.get("artifact_dir") or f"{IO_PATH_PREFIX}{DEEPSEARCH_ARTIFACTS_NAMESPACE}").strip()
        if not configured.startswith(IO_PATH_PREFIX):
            raise ValueError("DeepSearchService config.artifact_dir must be an io:// virtual path")

        root_key = io_key(configured)
        namespace, prefix = (root_key.split("/", 1) + [""])[:2]
        namespace = namespace or DEEPSEARCH_ARTIFACTS_NAMESPACE
        return DeepSearchArtifactStore.from_io_manager(io_manager, namespace=namespace, prefix=prefix)

    def _persist_experiment_snapshot(
        self,
        *,
        question: str,
        plan: Dict[str, Any],
        reasoning: Dict[str, Any],
        report: Dict[str, Any],
        snapshot: Dict[str, Any],
        stage_timings: Dict[str, Any],
    ) -> None:
        if not self.experiment_output_dir:
            return
        from app_registration import registrator

        io_manager = registrator.get_object("io_manager")
        if io_manager is None:
            raise RuntimeError("io_manager is required for DeepSearch experiment snapshots")

        plan_payload = plan.get("plan") or {}
        plan_id = plan.get("plan_id") or plan_payload.get("plan_id") or snapshot.get("plan_metadata", {}).get("plan_id")
        reasoning_steps = reasoning.get("reasoning_steps") or []
        experiment_record = {
            "question": question,
            "plan_id": plan_id,
            "config_fingerprint": snapshot.get("config_fingerprint"),
            "stage_timings": stage_timings,
            "coverage_metrics": reasoning.get("coverage_metrics"),
            "plan_steps": plan_payload.get("steps") or [],
            "reasoning_steps": reasoning_steps,
            "think_notes": reasoning.get("think_notes") or [],
            "tool_results": reasoning.get("tool_results") or [],
            "answer": report.get("answer"),
            "highlights": report.get("highlights"),
            "evidence_ids": [chunk.get("chunk_id") for chunk in report.get("evidences") or []],
            "request_metadata": snapshot.get("request_metadata"),
        }
        filename = plan_id or snapshot.get("run_id") or f"run_{int(time.time() * 1000)}"
        root_key = io_key(str(self.experiment_output_dir))
        namespace, prefix = (root_key.split("/", 1) + [""])[:2]
        namespace = namespace or "deepsearch_experiments"
        key = "/".join([p for p in [prefix, f"{filename}.json"] if p])
        try:
            io_manager.put_json(namespace=namespace, key=key, payload=json_safe(experiment_record))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist DeepSearch experiment snapshot: %s", exc)
            return
