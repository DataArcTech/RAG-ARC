# Prompt Layering

This repo keeps prompt strings centralized under `core/prompts/` and uses `config/prompts/` for optional, editable layers.

## What gets composed

The user-facing RAG pipeline (`rag_inference`) composes a system prompt as:

1. **Base layer** (service-agnostic; defined in code)
2. **Citations addon** (rag_inference only; defined in code)
3. **Style layer** (optional; from YAML by `USER_TYPE`)
4. **Domain layer** (optional; from YAML by `USER_TYPE`, or via a referenced file path)

The CLI pipeline is for debugging and intentionally uses **base layer only** (no citations/style/domain constraints).

## YAML config

File: `config/prompts/rag_inference_prompts.yaml`

Fields per entry:
- `type`: integer (matched against env `USER_TYPE`)
- `style_prompt`: optional string
- `domain_prompt`: optional string
- `domain_prompt_path`: optional repo-relative file path to load domain prompt text

Backward compatibility:
- `system_prompt` (string) is still supported as a full override, but using layers is preferred.

