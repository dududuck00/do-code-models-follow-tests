from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model_inputs import extract_prompt_token_ids, prepare_prompt_inputs


@dataclass
class GenerationOutput:
    completion: str
    state_path: str | None = None


class LocalCausalLM:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        prompt_format: str = "raw",
        thinking: str = "auto",
        tokenizer_path: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for real model generation. Install it with "
                "`pip install -r requirements-model.txt` or install the PyTorch "
                "build matching your CUDA/CPU environment."
            ) from exc
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.prompt_format = prompt_format
        self.thinking = thinking
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.is_multimodal = config.model_type in {"qwen3_5", "qwen3_5_moe"}
        if self.is_multimodal:
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "This model uses a multimodal conditional-generation architecture. "
                    "Install the latest Transformers release, then retry."
                ) from exc
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            self.tokenizer = self.processor.tokenizer
            model_class = AutoModelForMultimodalLM
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path or model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            model_class = AutoModelForCausalLM
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = _resolve_dtype(torch, dtype)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device `{device}`, but torch.cuda.is_available() is False.")
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
            "dtype": torch_dtype,
        }
        if device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": device}

        self.model = model_class.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.input_device = _first_parameter_device(self.model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        collect_states: bool = False,
        state_path: str | Path | None = None,
    ) -> GenerationOutput:
        torch = self.torch
        inputs = self._move_inputs(self._prepare_inputs(prompt))

        saved_state_path: str | None = None
        if collect_states and state_path is not None:
            saved_state_path = self._collect_prompt_state_from_inputs(inputs, state_path)

        do_sample = temperature > 0
        generation_config = getattr(self.model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0, prompt_len:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationOutput(completion=completion, state_path=saved_state_path)

    def collect_prompt_state(self, prompt: str, state_path: str | Path) -> str:
        inputs = self._move_inputs(self._prepare_inputs(prompt))
        return self._collect_prompt_state_from_inputs(inputs, state_path)

    def prompt_token_ids(self, prompt: str) -> list[int]:
        return extract_prompt_token_ids(self._prepare_inputs(prompt))

    def _prepare_inputs(self, prompt: str) -> dict[str, Any]:
        renderer = self.processor or self.tokenizer
        return prepare_prompt_inputs(
            self.tokenizer,
            renderer,
            prompt,
            self.prompt_format,
            self.thinking,
            return_tensors="pt",
        )

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _collect_prompt_state_from_inputs(
        self,
        inputs: dict[str, Any],
        state_path: str | Path,
    ) -> str:
        torch = self.torch
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        hidden = torch.stack(
            [layer[:, -1, :].detach().float().cpu() for layer in outputs.hidden_states],
            dim=0,
        ).squeeze(1)
        logits = outputs.logits[:, -1, :].detach().float()
        topk = min(50, logits.shape[-1])
        values, indices = torch.topk(logits, k=topk, dim=-1)
        np.savez_compressed(
            path,
            prompt_end_hidden=hidden.numpy(),
            topk_logits=values.cpu().numpy(),
            topk_token_ids=indices.cpu().numpy(),
        )
        return str(path)


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _first_parameter_device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return "cpu"
