from __future__ import annotations

import hashlib
from typing import Any


def prepare_prompt_inputs(
    tokenizer: Any,
    renderer: Any,
    prompt: str,
    prompt_format: str,
    thinking: str,
    *,
    return_tensors: str | None,
) -> dict[str, Any]:
    if prompt_format == "raw":
        return tokenizer(prompt, return_tensors=return_tensors)
    if prompt_format != "chat":
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    # Text-only tokenizers (including DeepSeek-Coder) expect string content.
    content = [{"type": "text", "text": prompt}] if renderer is not tokenizer else prompt
    messages = [{"role": "user", "content": content}]
    template_kwargs: dict[str, Any] = {}
    if thinking == "enabled":
        template_kwargs["enable_thinking"] = True
    elif thinking == "disabled":
        template_kwargs["enable_thinking"] = False
    elif thinking != "auto":
        raise ValueError(f"Unsupported thinking mode: {thinking}")

    return renderer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors=return_tensors,
        **template_kwargs,
    )


def extract_prompt_token_ids(inputs: dict[str, Any]) -> list[int]:
    input_ids = inputs["input_ids"]
    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().tolist()
    elif hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"Expected one prompt, received batch size {len(input_ids)}")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def prompt_token_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
