import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from application.deepsearch.artifacts import DeepSearchArtifactStore
from core.utils.json_safe import json_safe

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

    def _resolve_experiment_dir(self) -> Optional[Path]:
        candidate = None
        if isinstance(self.config, dict):
            candidate = self.config.get("experiment_output_dir")
        directory = candidate
        if not directory:
            return None
        return Path(str(directory)).expanduser()

    def _resolve_artifact_store(self) -> DeepSearchArtifactStore | None:
        if not isinstance(self.config, dict):
            raise ValueError("DeepSearchService config is required to resolve artifact store")
        artifacts_cfg = self._resolve_artifacts_config()
        if not bool(artifacts_cfg.get("enabled", True)):
            return None
        configured = self.config.get("artifact_dir")
        if not configured:
            raise ValueError("DeepSearchService config.artifact_dir is required (no implicit env fallback).")
        return DeepSearchArtifactStore.from_root_dir(str(configured))

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
        try:
            self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.warning("Failed to prepare experiment directory %s: %s", self.experiment_output_dir, exc)
            return

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
        path = self.experiment_output_dir / f"{filename}.json"
        try:
            path.write_text(
                json.dumps(json_safe(experiment_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.warning("Failed to persist DeepSearch experiment snapshot: %s", exc)
