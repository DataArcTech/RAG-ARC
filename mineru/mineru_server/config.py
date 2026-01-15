import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger


SERVER_DIR = Path(__file__).resolve().parents[1]  # MinerU/
DEFAULT_OUTPUT_DIR = SERVER_DIR / "mineru_outputs"
DEFAULT_TEMP_DIR = SERVER_DIR / ".tmp" / "mineru_temp"
DEFAULT_CONFIG_PATH = SERVER_DIR / ".tmp" / "mineru_server_config.json"


def _default_cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve()
    return (Path.home() / ".cache").expanduser().resolve()


def _default_modelscope_cache_dir() -> str:
    # ModelScope commonly uses ~/.cache/modelscope/hub.
    return str((_default_cache_root() / "modelscope" / "hub").resolve())


def _default_hf_home() -> str:
    return str((_default_cache_root() / "huggingface").resolve())


SAFE_FILENAME_PATTERN = __import__("re").compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class ServerConfig:
    # Service
    host: str = "0.0.0.0"
    port: int = 8899
    workers: int = 1
    max_jobs_per_worker: int = 1

    # Storage
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    temp_dir: str = str(DEFAULT_TEMP_DIR)
    config_path: str = str(DEFAULT_CONFIG_PATH)
    mineru_home: str = str(SERVER_DIR)

    # MinerU runtime defaults
    model_source: str = "modelscope"
    device_mode: str = "cuda"  # "cuda" | "cpu" | "npu" | "auto"
    backend: str = "vlm-transformers"
    parse_method: str = "auto"
    lang: str = "ch"
    formula_enable: bool = True
    table_enable: bool = True

    # vLLM knobs (only used for vLLM backends)
    virtual_vram_gb: Optional[int] = None
    vllm_gpu_memory_utilization: float = 0.5
    vllm_enforce_eager: bool = False
    vllm_max_model_len: Optional[int] = None
    vllm_swap_space_gb: float = 4.0
    vllm_cpu_offload_gb: float = 0.0

    # Dump options
    dump_md: bool = True
    dump_content_list: bool = True
    dump_middle_json: bool = False
    dump_model_output: bool = False

    # Cache locations (writable)
    modelscope_cache_dir: str = field(default_factory=_default_modelscope_cache_dir)
    hf_home: str = field(default_factory=_default_hf_home)

    # Captioning
    caption_mode: str = "content_list_then_llm"  # off | content_list | llm | content_list_then_llm
    chat_api_base_url: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_api_key_file: Optional[str] = None
    chat_model: str = "gemini-2.5-flash"
    chat_timeout_s: int = 60
    # Safety knob for LLM captioning. If <= 0, treat it as "no limit".
    caption_max_images: int = 32
    caption_context: str = ""
    caption_context_file: Optional[str] = None
    caption_up_tokens: int = 500
    caption_down_tokens: int = 500

    def output_dir_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def temp_dir_path(self) -> Path:
        return Path(self.temp_dir).expanduser().resolve()

    def config_path_path(self) -> Path:
        return Path(self.config_path).expanduser().resolve()

    def mineru_home_path(self) -> Path:
        return Path(self.mineru_home).expanduser().resolve()

    def modelscope_cache_path(self) -> Path:
        return Path(self.modelscope_cache_dir).expanduser().resolve()

    def hf_home_path(self) -> Path:
        return Path(self.hf_home).expanduser().resolve()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ServerConfig":
        allowed = set(ServerConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in (data or {}).items() if k in allowed}
        return ServerConfig(**filtered)


def secure_filename(filename: str) -> tuple[str, str]:
    name = Path(filename).name
    stem, suffix = os.path.splitext(name)

    safe_stem = SAFE_FILENAME_PATTERN.sub("_", stem).strip("._")
    if not safe_stem:
        safe_stem = f"file_{uuid.uuid4().hex}"
    safe_stem = safe_stem[:120]

    raw_suffix = SAFE_FILENAME_PATTERN.sub("", suffix)
    if raw_suffix and not raw_suffix.startswith("."):
        raw_suffix = f".{raw_suffix}"
    safe_suffix = raw_suffix[:16]

    return f"{safe_stem}{safe_suffix}", safe_stem


def safe_load_json(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load JSON from {path}: {exc}")
        return None


def apply_runtime_environment(cfg: ServerConfig) -> None:
    cfg.modelscope_cache_path().mkdir(parents=True, exist_ok=True)
    cfg.hf_home_path().mkdir(parents=True, exist_ok=True)

    os.environ["MODELSCOPE_CACHE"] = str(cfg.modelscope_cache_path())
    os.environ["HF_HOME"] = str(cfg.hf_home_path())

    os.environ["MINERU_MODEL_SOURCE"] = cfg.model_source
    os.environ["MINERU_DEVICE_MODE"] = cfg.device_mode
    os.environ["MINERU_DEFAULT_BACKEND"] = cfg.backend
    os.environ["MINERU_DEFAULT_PARSE_METHOD"] = cfg.parse_method
    os.environ["MINERU_DEFAULT_LANG"] = cfg.lang
    os.environ["MINERU_FORMULA_ENABLE"] = "true" if cfg.formula_enable else "false"
    os.environ["MINERU_TABLE_ENABLE"] = "true" if cfg.table_enable else "false"
    if cfg.virtual_vram_gb is not None:
        os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(cfg.virtual_vram_gb)

    os.environ["MINERU_OUTPUT_DIR"] = str(cfg.output_dir_path())
    os.environ["MINERU_TEMP_DIR"] = str(cfg.temp_dir_path())


def ensure_mineru_import_path(cfg: ServerConfig) -> None:
    mineru_home = cfg.mineru_home_path()
    if not mineru_home.exists():
        raise RuntimeError(f"MinerU home not found: {mineru_home}")
    if str(mineru_home) not in sys.path:
        sys.path.insert(0, str(mineru_home))


def save_config(cfg: ServerConfig, *, include_secrets: bool) -> None:
    path = cfg.config_path_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg) if include_secrets else {**asdict(cfg), "chat_api_key": None}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> Optional[ServerConfig]:
    if not path.exists():
        return None
    try:
        return ServerConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning(f"Failed to load config {path}: {exc}")
        return None


def clamp_vllm_gpu_memory_utilization(cfg: ServerConfig) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if not cfg.device_mode.startswith("cuda"):
        return extra
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return extra
        free_b, total_b = torch.cuda.mem_get_info()
        free_ratio = float(free_b) / float(total_b) if total_b else 0.0
        if free_ratio <= 0.0:
            return extra
        safe_ratio = min(cfg.vllm_gpu_memory_utilization, max(1e-4, free_ratio * 0.85))
        extra["gpu_memory_utilization"] = safe_ratio
    except Exception as exc:
        logger.warning(f"Failed to clamp vLLM gpu_memory_utilization: {exc}")
    return extra


def copy_upload_to_temp(upload_file, temp_path: Path) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(upload_file, buffer)
    finally:
        try:
            upload_file.close()
        except Exception:
            pass
