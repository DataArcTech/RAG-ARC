import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mineru_server.config import ServerConfig, clamp_vllm_gpu_memory_utilization


async def run_mineru_parse(
    *,
    cfg: ServerConfig,
    task_dir: Path,
    file_path: Path,
    doc_name: str,
    backend: str,
    parse_method: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
    start_page: int,
    end_page: Optional[int],
    output_format: str,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    from mineru.cli.common import aio_do_parse, do_parse, read_fn
    from mineru.utils.enum_class import MakeMode

    pdf_bytes = read_fn(file_path)

    if output_format == "mm_md":
        make_mode = MakeMode.MM_MD
    elif output_format == "md_only":
        make_mode = MakeMode.MD
    elif output_format == "content_list":
        make_mode = MakeMode.CONTENT_LIST
    else:
        make_mode = MakeMode.MM_MD

    extra_kwargs: Dict[str, Any] = {}
    if "vllm" in (backend or "") and cfg.device_mode.startswith("cuda"):
        extra_kwargs.update(clamp_vllm_gpu_memory_utilization(cfg))
        if cfg.vllm_enforce_eager:
            extra_kwargs["enforce_eager"] = True
        if cfg.vllm_max_model_len is not None:
            extra_kwargs["max_model_len"] = int(cfg.vllm_max_model_len)
        if cfg.vllm_swap_space_gb is not None:
            extra_kwargs["swap_space"] = float(cfg.vllm_swap_space_gb)
        if cfg.vllm_cpu_offload_gb is not None and float(cfg.vllm_cpu_offload_gb) > 0.0:
            extra_kwargs["cpu_offload_gb"] = float(cfg.vllm_cpu_offload_gb)

    if backend.startswith("vlm-vllm-async-engine"):
        await aio_do_parse(
            output_dir=str(task_dir),
            pdf_file_names=[doc_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            f_dump_md=cfg.dump_md,
            f_dump_content_list=cfg.dump_content_list,
            f_dump_middle_json=cfg.dump_middle_json,
            f_dump_model_output=cfg.dump_model_output,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_make_md_mode=make_mode,
            start_page_id=start_page,
            end_page_id=end_page,
            **extra_kwargs,
        )
    else:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: do_parse(
                output_dir=str(task_dir),
                pdf_file_names=[doc_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[lang],
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                f_dump_md=cfg.dump_md,
                f_dump_content_list=cfg.dump_content_list,
                f_dump_middle_json=cfg.dump_middle_json,
                f_dump_model_output=cfg.dump_model_output,
                f_dump_orig_pdf=False,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_make_md_mode=make_mode,
                start_page_id=start_page,
                end_page_id=end_page,
                **extra_kwargs,
            ),
        )

    doc_root = task_dir / doc_name
    if not doc_root.exists():
        raise RuntimeError(f"MinerU output not found: {doc_root}")
    candidates = [p for p in doc_root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No method directory under: {doc_root}")
    for cand in candidates:
        if (cand / f"{doc_name}.md").exists():
            method_dir = cand
            break
    else:
        method_dir = doc_root / "vlm" if (doc_root / "vlm").exists() else candidates[0]

    md_path = method_dir / f"{doc_name}.md" if cfg.dump_md else None
    content_list_path = method_dir / f"{doc_name}_content_list.json" if cfg.dump_content_list else None
    if md_path is not None and not md_path.exists():
        md_path = None
    if content_list_path is not None and not content_list_path.exists():
        content_list_path = None

    return method_dir, md_path, content_list_path