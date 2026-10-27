"""Gated DeltaNet likelihoods using flash-linear-attention kernels.

Supports full scan, cached chunks and fused recurrent execution; check_gdn
compares the three paths.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np

from .mamba_adapter import _score_inputs, _version


DEFAULT_MODEL_ID = "linear-moe-hub/Gated-Deltanet-340M"
DEFAULT_REVISION = "c83bdada453cde56932f37be71338df22ca29b7d"
TOKENIZER_ID = DEFAULT_MODEL_ID
DEFAULT_TOKENIZER_REVISION = DEFAULT_REVISION


def remap_checkpoint_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map this checkpoint's MLP weights to the released FLA names.

    The checkpoint stores gate/up concatenated in gate_proj, plus an attn.D that
    the GDN forward does not use. Splitting leaves every used weight unchanged;
    any other unexpected key fails strict loading.
    """
    mapped, dropped = {}, {}
    for key, value in state.items():
        if key.endswith(".attn.D"):
            dropped[key] = value
        elif (
            key.endswith(".mlp.gate_proj.weight")
            and key.replace("gate_proj", "up_proj") not in state
        ):
            gate, up = value.chunk(2, 0)
            mapped[key] = gate.contiguous()
            mapped[key.replace("gate_proj", "up_proj")] = up.contiguous()
        else:
            mapped[key] = value
    return mapped, dropped


def _cached_target_losses(
    input_ids: np.ndarray,
    target_start: int,
    chunk_size: int,
    forward: Callable[[np.ndarray, Any], tuple[Any, Any]],
    readout: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Consume each prefix once, scoring only next-token-aligned target rows.

    A fresh cache starts on every call. Unlike native Mamba2's one-token cache,
    FLA supports multi-token chunks with recurrent and convolution state carry.
    The model itself advances its cache; the scheduler does not change offsets.
    """
    ids = _score_inputs(input_ids, target_start, chunk_size)
    losses = np.empty(len(ids) - target_start, dtype=np.float32)
    if not len(losses):
        return losses
    state = None
    for begin in range(0, len(ids) - 1, chunk_size):
        end = min(begin + chunk_size, len(ids) - 1)
        hidden, state = forward(ids[begin:end], state)
        first_target = max(target_start, begin + 1)
        if first_target <= end:
            losses[first_target - target_start : end + 1 - target_start] = readout(
                hidden[:, first_target - 1 - begin : end - begin],
                ids[first_target : end + 1],
            )
        del hidden
    return losses


class GDNAdapter:
    """Likelihoods of ids[target_start:] with a fresh state per call.

    Default BF16 matches the published configuration; CE/log-softmax is FP32
    over all native output classes, with no vocabulary slicing. A caller adds
    exactly one native BOS/document-boundary token; encode adds no specials.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_memory_gb: float | None = None,
        local_files_only: bool = True,
        solve_precision: str = "native",
    ) -> None:
        if solve_precision not in ("native", "ieee"):
            raise ValueError("solve_precision must be native or ieee")
        import torch
        import fla
        from fla.ops.gated_delta_rule import chunk_fwd
        from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        # FLA's fused BT64 triangular solve supplies input_precision explicitly,
        # so TRITON_F32_DEFAULT cannot override it. This optional diagnostic
        # changes only the current process, before any kernel is compiled.
        native_solve_precision = str(chunk_fwd.SOLVE_TRIL_DOT_PRECISION.value)
        if solve_precision == "ieee":
            import triton.language as tl

            chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
        effective_solve_precision = str(chunk_fwd.SOLVE_TRIL_DOT_PRECISION.value)

        self._torch, self.device = torch, device
        memory_budget = None
        if max_memory_gb is not None:
            requested = float(max_memory_gb) * 1_000_000_000
            if not np.isfinite(requested) or requested <= 0:
                raise ValueError("max_memory_gb must be finite and positive")
            index = torch.device(device).index
            if index is None:
                index = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(index).total_memory
            fraction = min(requested / total, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, device=index)
            memory_budget = {
                "requested_decimal_gb": max_memory_gb,
                "allocator_fraction": fraction,
                "scope": "PyTorch allocator; context/library allocations are additional",
            }

        def cached(filename):
            return hf_hub_download(
                model_id, filename, revision=revision, local_files_only=local_files_only
            )

        with open(cached("config.json"), encoding="utf-8") as handle:
            raw_config = json.load(handle)
        if raw_config.get("model_type") != "gated_deltanet" or raw_config.get("attn"):
            raise ValueError("GDNAdapter expects an all-Gated-DeltaNet checkpoint")
        config = GatedDeltaNetConfig(**raw_config)
        config.use_cache = False
        config.attn_mode = "chunk"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
        )
        # Build on CPU; state copying is exact before conversion to the declared
        # inference dtype.
        model = GatedDeltaNetForCausalLM(config)
        state = load_file(cached("model.safetensors"))
        stored_dtypes = sorted({str(value.dtype) for value in state.values()})
        state, dropped = remap_checkpoint_state(state)
        model.load_state_dict(state, strict=True)
        remap_info = {
            "split_gate_up_layers": sum(key.endswith(".mlp.up_proj.weight") for key in state),
            "dropped_unused_keys": sorted(dropped),
        }
        del state, dropped
        self.model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
        self.model.requires_grad_(False)
        self.base, self.head = self.model.model, self.model.get_output_embeddings()
        self.prefix_id = int(self.tokenizer.bos_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.metadata = {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": tokenizer_revision,
            "dtype": dtype,
            "stored_dtypes": stored_dtypes,
            "loss_dtype": "float32",
            "device": str(device),
            "memory_budget": memory_budget,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "backend": f"flash-linear-attention {fla.__version__} native GatedDeltaNet",
            "compiler_mode": "native fused Triton kernels; no torch.compile",
            "versions": {
                name: _version(name)
                for name in (
                    "torch",
                    "triton",
                    "flash-linear-attention",
                    "fla-core",
                    "transformers",
                )
            },
            "cuda_runtime": torch.version.cuda,
            "torch_matmul_tf32": False,
            "torch_cudnn_tf32": False,
            "triton_fp32_dot_precision_requested": os.environ.get(
                "TRITON_F32_DEFAULT", "backend default"
            ),
            "fla_fused_solve_precision": {
                "requested": solve_precision,
                "native_at_import": native_solve_precision,
                "effective": effective_solve_precision,
                "scope": "fused BT64 triangular solve; process-local constexpr override only",
            },
            "checkpoint_remap": remap_info,
            "boundary_token_id": self.prefix_id,
            "special_tokens": "encode adds none; caller supplies one native BOS per history",
            "vocabulary_normalization": "all native lm_head output classes",
            "vocabulary_classes": int(self.head.weight.shape[0]),
            "execution_modes": {
                "score_full": "full fused chunk scan; sequence-sized hidden activations; target logits chunked",
                "score": "bounded native cached chunks, including recurrent and short-conv state carry",
                "score_recurrent": "bounded <=64-token cached chunks using native fused recurrent kernels",
            },
            "validation_status": "GPU agreement and timing must be read from this run's smoke report",
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _readout(self, hidden: Any, targets: np.ndarray) -> np.ndarray:
        torch = self._torch
        logits = self.head(hidden).float().reshape(len(targets), -1)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        losses = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        return losses.cpu().numpy().astype(np.float32, copy=False)

    def score(self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256) -> np.ndarray:
        torch = self._torch

        def forward(ids, state):
            tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device)[None, :]
            output = self.base(
                input_ids=tensor, past_key_values=state, use_cache=True, return_dict=True
            )
            return output.last_hidden_state, output.past_key_values

        with torch.inference_mode():
            return _cached_target_losses(
                input_ids, target_start, chunk_size, forward, self._readout
            )

    def score_recurrent(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 64
    ) -> np.ndarray:
        """FLA switches to its fused recurrent kernel at <=64 tokens in eval."""
        return self.score(input_ids, target_start, min(chunk_size, 64))

    def score_full(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        ids = _score_inputs(input_ids, target_start, chunk_size)
        losses = np.empty(len(ids) - target_start, dtype=np.float32)
        if not len(losses):
            return losses
        torch = self._torch
        with torch.inference_mode():
            tensor = torch.as_tensor(ids[:-1], dtype=torch.long, device=self.device)[None, :]
            hidden = self.base(
                input_ids=tensor, use_cache=False, return_dict=True
            ).last_hidden_state
            for start in range(target_start, len(ids), chunk_size):
                end = min(start + chunk_size, len(ids))
                losses[start - target_start : end - target_start] = self._readout(
                    hidden[:, start - 1 : end - 1],
                    ids[start:end],
                )
        return losses
