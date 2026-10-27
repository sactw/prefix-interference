"""RWKV-7 Pile likelihoods through the official RWKV runtime.

The native CUDA FP32 recurrence carries all three state pieces per layer.
Prefix chunks request only their final logits; target chunks request every
logit. Every call starts from zero state; check_rwkv compares the execution
paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .hashing import sha256_file
from .mamba_adapter import _score_inputs, _version


DEFAULT_MODEL_ID = "BlinkDL/rwkv-7-pile"
DEFAULT_REVISION = "dac041b037030e33db4ccaccffb67b3fd6d63ff4"
DEFAULT_FILENAME = "RWKV-x070-Pile-421M-20241127-ctx4096.pth"
TOKENIZER_ID = "BlinkDL/ChatRWKV/20B_tokenizer.json"
DEFAULT_TOKENIZER_REVISION = "2e2bb1cd390cefbedae0a89f4a343f6f754d6621"
DEFAULT_RUNTIME_REVISION = DEFAULT_TOKENIZER_REVISION


def _chunked_target_losses(
    input_ids: np.ndarray,
    target_start: int,
    chunk_size: int,
    prefill_chunk_size: int,
    forward: Callable[[np.ndarray, Any, bool], tuple[Any, Any]],
    readout: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Consume each token once; row t predicts token t+1, including seams.

    The prefix ends before the token predicting the first target, so no pending
    boundary logit needs to be carried separately. ``forward`` owns native state
    updates and receives full_output=True only for scored rows.
    """
    ids = _score_inputs(input_ids, target_start, chunk_size)
    if not isinstance(prefill_chunk_size, (int, np.integer)) or prefill_chunk_size < 1:
        raise ValueError("prefill_chunk_size must be a positive integer")
    losses = np.empty(len(ids) - target_start, dtype=np.float32)
    if not len(losses):
        return losses
    state = None
    prefix_end = target_start - 1
    for begin in range(0, prefix_end, prefill_chunk_size):
        logits, state = forward(
            ids[begin : min(begin + prefill_chunk_size, prefix_end)], state, False
        )
        del logits
    for start in range(target_start, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        logits, state = forward(ids[start - 1 : end - 1], state, True)
        losses[start - target_start : end - target_start] = readout(logits, ids[start:end])
        del logits
    return losses


class RWKVAdapter:
    """FP32 all-vocabulary NLL of input_ids[target_start:], fresh state/call.

    Set LCD_RWKV_TOKENIZER to the pinned official 20B_tokenizer.json, or pass
    tokenizer_path. The Pile checkpoint uses this tokenizer, not RWKV World.
    Native sequence kernels are JIT compiled by the installed official package;
    Source hashes and runtime versions are recorded with each result.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
        device: str = "cuda",
        dtype: str = "float32",
        max_memory_gb: float | None = None,
        local_files_only: bool = True,
        *,
        filename: str = DEFAULT_FILENAME,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        prefill_chunk_size: int = 4096,
        cuda_on: bool = True,
    ) -> None:
        if dtype != "float32":
            raise ValueError("This study's RWKV adapter scores in float32")
        if prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be positive")
        import torch
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._torch, self.device = torch, str(device)
        self.prefill_chunk_size = int(prefill_chunk_size)
        memory_budget = None
        if max_memory_gb is not None and torch.device(device).type == "cuda":
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

        # The official module selects its architecture/kernel branches at import.
        # Distinct CUDA/eager smoke modes therefore run in separate processes.
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1" if cuda_on else "0"
        os.environ["RWKV_DE_VERSION"] = "0"
        import rwkv.model as native

        # Official import enables TF32; turn it off before loading/folding the
        # embedding layer normalization and before any model matmul executes.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_file = (
            Path(model_path)
            if model_path is not None
            else Path(
                hf_hub_download(
                    model_id, filename, revision=revision, local_files_only=local_files_only
                )
            )
        )
        requested_tokenizer = tokenizer_path or os.environ.get("LCD_RWKV_TOKENIZER")
        if requested_tokenizer is None:
            raise ValueError(
                "Pass tokenizer_path or LCD_RWKV_TOKENIZER for official pinned 20B_tokenizer.json"
            )
        token_file = Path(requested_tokenizer)
        self.tokenizer = Tokenizer.from_file(str(token_file))
        self.prefix_id = self.eos_id = 0
        # Official loader appends '.pth', including when loading a cached HF path.
        native_path = str(model_file)
        if native_path.endswith(".pth"):
            native_path = native_path[:-4]
        self.model = native.RWKV(model=native_path, strategy=f"{device} fp32")
        self.model.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        vocab_classes = int(self.model.z["head.weight"].shape[-1])
        source = Path(native.__file__)
        source_paths = [
            source,
            source.parent / "cuda/rwkv7.cu",
            source.parent / "cuda/rwkv7_op.cpp",
        ]
        self.metadata = {
            "model_id": model_id,
            "revision": revision,
            "filename": filename,
            "model_file_sha256": sha256_file(model_file),
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_file_sha256": sha256_file(token_file),
            "dtype": "float32",
            "loss_dtype": "float32",
            "state_dtype": "float32",
            "device": str(device),
            "memory_budget": memory_budget,
            "parameters": sum(tensor.numel() for tensor in self.model.z.values()),
            "layers": int(self.model.n_layer),
            "width": int(self.model.n_embd),
            "head_size": int(self.model.head_size),
            "backend": "official rwkv RWKV_x070 native CUDA FP32"
            if cuda_on
            else "official rwkv RWKV_x070 TorchScript eager recurrence",
            "selected_execution": "native_fused_chunked_recurrence"
            if cuda_on
            else "native_torchscript_chunked_recurrence",
            "compiler_mode": "official TorchScript plus native CUDA extension"
            if cuda_on
            else "official TorchScript; no CUDA recurrence extension",
            "runtime_reference_revision": DEFAULT_RUNTIME_REVISION,
            "runtime_source_sha256": {
                str(path.relative_to(source.parent)): sha256_file(path)
                for path in source_paths
                if path.is_file()
            },
            "versions": {
                name: _version(name) for name in ("torch", "rwkv", "tokenizers", "huggingface-hub")
            },
            "cuda_runtime": torch.version.cuda,
            "native_cuda_fast_math": bool(cuda_on),
            "torch_matmul_tf32": False,
            "torch_cudnn_tf32": False,
            "boundary_token_id": self.prefix_id,
            "special_tokens": "encode adds none; caller supplies one native Pile document boundary ID0",
            "vocabulary_normalization": "all native head output classes, including any padded classes",
            "vocabulary_classes": vocab_classes,
            "tokenizer_vocabulary": int(self.tokenizer.get_vocab_size()),
            "prefill_chunk_size": self.prefill_chunk_size,
            "state": "native attention previous-x, matrix state and FFN previous-x in every layer",
            "execution_modes": {
                "score_full": "bounded native fused sequence prefill, followed by all-logit target chunks",
                "score": "same native recurrence, caller-sized chunks for both prefix and targets",
                "score_reference": "single native full-output sequence; short numerical reference only",
                "score_prefill_reference": "single native prefix call with final output only, then target chunks",
                "score_one": "official one-token TorchScript recurrence; short numerical reference only",
            },
            "validation_status": "GPU parity/timing must be read from this run's smoke report",
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _forward(self, ids, state, full_output):
        return self.model.forward(ids.tolist(), state, full_output=full_output)

    def _readout(self, logits: Any, targets: np.ndarray) -> np.ndarray:
        torch = self._torch
        logits = logits.float().reshape(len(targets), -1)
        labels = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        return (
            torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def score_full(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        with self._torch.inference_mode():
            return _chunked_target_losses(
                input_ids,
                target_start,
                chunk_size,
                self.prefill_chunk_size,
                self._forward,
                self._readout,
            )

    def score(self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256) -> np.ndarray:
        with self._torch.inference_mode():
            return _chunked_target_losses(
                input_ids, target_start, chunk_size, chunk_size, self._forward, self._readout
            )

    def score_one(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        with self._torch.inference_mode():
            return _chunked_target_losses(
                input_ids, target_start, 1, 1, self._forward, self._readout
            )

    def score_prefill_reference(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        """Single prefix call for long-context chunk-boundary agreement.

        This retains sequence-sized prefix activations, but only the last prefix
        vocabulary projection. Target logits stay bounded by chunk_size. Unlike
        score_reference, it does not allocate a prefix-length by vocabulary tensor.
        """
        with self._torch.inference_mode():
            return _chunked_target_losses(
                input_ids,
                target_start,
                chunk_size,
                max(1, target_start - 1),
                self._forward,
                self._readout,
            )

    def score_reference(
        self, input_ids: np.ndarray, target_start: int, chunk_size: int = 256
    ) -> np.ndarray:
        ids = _score_inputs(input_ids, target_start, chunk_size)
        if target_start == len(ids):
            return np.empty(0, dtype=np.float32)
        with self._torch.inference_mode():
            logits, _ = self._forward(ids[:-1], None, True)
            logits = logits.reshape(len(ids) - 1, -1)
            return self._readout(logits[target_start - 1 :], ids[target_start:])
