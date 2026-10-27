"""Mamba-2 likelihoods using native scan and recurrent kernels.

``score`` uses a bounded prefill followed by one-token cached continuation.
Its ``chunk_size`` limits the prefill, not the continuation length per call.
``score_full`` uses the native fused scan, retaining sequence-sized hidden
activations and projecting only target-predicting rows to logits.

Model dependencies are loaded when the adapter is constructed.
"""

from __future__ import annotations

import json
from importlib import metadata as package_metadata
import os
from typing import Any, Callable
import warnings

import numpy as np


DEFAULT_MODEL_ID = "state-spaces/mamba2-130m"
DEFAULT_REVISION = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
DEFAULT_TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"


def _score_inputs(input_ids: np.ndarray, target_start: int, chunk_size: int) -> np.ndarray:
    ids = np.asarray(input_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("input_ids must be a one-dimensional integer array")
    if not isinstance(target_start, (int, np.integer)) or not 1 <= target_start <= len(ids):
        raise ValueError("target_start must be in [1, len(input_ids)]; supply a preceding token")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if np.any(ids < 0):
        raise ValueError("token IDs must be nonnegative")
    return ids


def _stream_target_losses(
    input_ids: np.ndarray,
    target_start: int,
    chunk_size: int,
    new_state: Callable[[int], Any],
    forward: Callable[[np.ndarray, Any], Any],
    readout: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Schedule native prefill and one-token continuation.

    ``forward`` returns [1, sequence, hidden] and updates the native state in
    place. This scheduler alone advances ``seqlen_offset``. ``readout`` receives
    only hidden rows whose next-token targets are being measured.
    """
    ids = _score_inputs(input_ids, target_start, chunk_size)
    losses = np.empty(len(ids) - target_start, dtype=np.float32)
    if not len(losses):
        return losses
    state = new_state(len(ids))
    input_count = len(ids) - 1
    prefill_end = min(chunk_size, input_count)
    hidden = forward(ids[:prefill_end], state)
    state.seqlen_offset += prefill_end
    if prefill_end >= target_start:
        count = prefill_end - target_start + 1
        losses[:count] = readout(
            hidden[:, target_start - 1 : prefill_end],
            ids[target_start : prefill_end + 1],
        )
    del hidden
    for position in range(prefill_end, input_count):
        hidden = forward(ids[position : position + 1], state)
        state.seqlen_offset += 1
        if position + 1 >= target_start:
            losses[position + 1 - target_start] = readout(hidden, ids[position + 1 : position + 2])[
                0
            ]
        del hidden
    return losses


def _version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


class MambaAdapter:
    """Mamba-2 checkpoint scored from a fresh state on every call.

    Text encoding adds no special tokens. Callers prepend ``prefix_id`` (the
    native EOS/document-start token) consistently to each intervention and
    increment target_start accordingly. This keeps documentary prefix lengths
    separate from the one boundary token.

    ``score(ids, target_start, chunk_size=256)`` and
    ``score_full(ids, target_start, chunk_size=256)`` both return float32
    negative log likelihoods for exactly ``ids[target_start:]`` in nats.
    No target token is used to predict itself.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
        device: str = "cuda",
        dtype: str = "float32",
        max_memory_gb: float | None = None,
    ) -> None:
        import torch

        memory_budget = None
        if max_memory_gb is not None:
            requested_bytes = float(max_memory_gb) * 1_000_000_000
            if not np.isfinite(requested_bytes) or requested_bytes <= 0:
                raise ValueError("max_memory_gb must be finite and positive")
            # The allocator API requires an explicit index, unlike model.to("cuda").
            budget_device = torch.device(device)
            budget_index = budget_device.index
            if budget_index is None:
                budget_index = torch.cuda.current_device()
            total_bytes = torch.cuda.get_device_properties(budget_index).total_memory
            fraction = min(requested_bytes / total_bytes, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, device=budget_index)
            memory_budget = {
                "requested_decimal_gb": float(max_memory_gb),
                "requested_bytes": int(requested_bytes),
                "device_total_bytes": total_bytes,
                "allocator_fraction": fraction,
                "cuda_device_index": budget_index,
                "scope": "PyTorch caching allocator; CUDA context/library allocations are additional",
            }
        from huggingface_hub import hf_hub_download
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.modules import mamba2
        from mamba_ssm.utils.generation import InferenceParams
        from transformers import AutoTokenizer

        self._torch = torch
        self._InferenceParams = InferenceParams
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=tokenizer_revision)
        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        with open(config_path, encoding="utf-8") as handle:
            config_data = json.load(handle)
        if config_data.get("ssm_cfg", {}).get("layer") != "Mamba2" or config_data.get(
            "attn_layer_idx"
        ):
            raise ValueError("This adapter supports all-Mamba2 checkpoints only")
        config = MambaConfig(**config_data)
        self.model = MambaLMHeadModel(config, device=device, dtype=getattr(torch, dtype))
        state = torch.load(
            hf_hub_download(model_id, "pytorch_model.bin", revision=revision),
            map_location="cpu",
            weights_only=True,
        )
        stored_dtypes = sorted({str(value.dtype) for value in state.values()})
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval()
        self.base = self.model.backbone
        self.head = self.model.lm_head
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.prefix_id = self.eos_id
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        fused_step = (
            mamba2.causal_conv1d_update is not None and mamba2.selective_state_update is not None
        )
        if not fused_step:
            warnings.warn(
                "Native Mamba2 fused step kernels are unavailable; native eager "
                "fallback will be recorded. Do not treat its timing as a fused result.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.metadata = {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": tokenizer_revision,
            "dtype": dtype,
            "stored_dtypes": stored_dtypes,
            "device": str(device),
            "memory_budget": memory_budget,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "backend": "mamba_ssm native Mamba2",
            "compiler_mode": "native CUDA/Triton kernels; Python orchestration; no torch.compile",
            "versions": {
                name: _version(name)
                for name in ("torch", "mamba-ssm", "causal-conv1d", "triton", "transformers")
            },
            "cuda_runtime": torch.version.cuda,
            "torch_matmul_tf32": False,
            "torch_cudnn_tf32": False,
            "triton_fp32_dot_precision": os.environ.get(
                "TRITON_F32_DEFAULT", "backend default; may use TF32"
            ),
            "triton_environment": {
                name: os.environ.get(name) for name in ("TRITON_F32_DEFAULT", "TRITON_CACHE_DIR")
            },
            "boundary_token_id": self.prefix_id,
            "special_tokens": "encode adds none; caller supplies one EOS boundary per history",
            "loss_dtype": "float32",
            "vocabulary_normalization": "all native lm_head output classes, including padding",
            "execution_modes": {
                "score": {
                    "prefill": "native SSD chunk scan",
                    "prefill_fused_convolution": mamba2.causal_conv1d_fn is not None,
                    "continuation": "one token per native cache step",
                    "fused_step_available": fused_step,
                    "memory": "fixed recurrent state plus bounded prefill activations/logits",
                },
                "score_full": {
                    "forward": "native fused mamba_split_conv1d_scan_combined",
                    "memory": "full-sequence hidden activations; chunked target logits",
                },
            },
            "long_context_status": (
                "Full fused scoring retains sequence-sized activations; cached scoring "
                "uses one-token continuation. Efficient 200K-1M chunked carry is not validated."
            ),
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
        """Bounded prefill followed by native one-token cached continuation."""
        torch = self._torch

        def new_state(length: int) -> Any:
            return self._InferenceParams(max_seqlen=length, max_batch_size=1)

        def forward(ids: np.ndarray, state: Any) -> Any:
            tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device)[None, :]
            return self.base(tensor, inference_params=state)

        with torch.inference_mode():
            return _stream_target_losses(
                input_ids, target_start, chunk_size, new_state, forward, self._readout
            )

    def score_full(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        """Score targets with a full fused scan; hidden memory grows with input length."""
        ids = _score_inputs(input_ids, target_start, chunk_size)
        losses = np.empty(len(ids) - target_start, dtype=np.float32)
        if not len(losses):
            return losses
        torch = self._torch
        with torch.inference_mode():
            tensor = torch.as_tensor(ids[:-1], dtype=torch.long, device=self.device)[None, :]
            hidden = self.base(tensor)
            for start in range(target_start, len(ids), chunk_size):
                end = min(start + chunk_size, len(ids))
                losses[start - target_start : end - target_start] = self._readout(
                    hidden[:, start - 1 : end - 1], ids[start:end]
                )
        return losses
