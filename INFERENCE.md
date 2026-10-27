# Inference reproduction

Independent inference requires the public source PDFs, pinned tokenizers and checkpoints, and Linux CUDA environments. Source URLs, hashes, extraction settings, target spans, and model revisions are recorded under `results/`. The CPU quickstart in [README.md](README.md) uses saved losses instead.

## Prepare source documents

Install PyMuPDF, tokenizers, huggingface-hub, and beautifulsoup4 in a preparation environment with the CPU requirements. Use the PyMuPDF version recorded in each `source_report.json` and obtain the tokenizers at their recorded revisions.

For example, with the pinned Mamba tokenizer saved as `tokenizers/mamba/tokenizer.json`:

```sh
python -m long_context_diagnostics.prepare_gao_panel \
  --tokenizer tokenizers/mamba/tokenizer.json --output-dir inputs/gao
python -m long_context_diagnostics.prepare_fresh_panel \
  --sources results/fresh_panel/sources.json \
  --tokenizer tokenizers/mamba/tokenizer.json \
  --donor-panel inputs/gao/panel.jsonl --output-dir inputs/fresh
```

The first command prepares the two GAO donors. The second extracts the fresh-panel candidates, reapplies native PDF block order to the donors, and writes `inputs/fresh/panel_donor1.jsonl` and `panel_donor2.jsonl`. Compare source and extracted-text hashes with the retained metadata. Use the recorded plans for the actual cohort and target spans; `sources.json` also includes unselected candidates.

The remaining preparation modules expose their inputs through `--help`:

| Module under `long_context_diagnostics` | Purpose |
| --- | --- |
| `prepare_matched_panel` | Build the original eight-report matched-text plans |
| `prepare_second_donor` | Build the second-donor plans for the same eight reports |
| `prepare_replication_sources` | Extract the additional reports using the original donor |
| `prepare_replication_panel` | Build matched Mamba/Gated DeltaNet plans for selected documents |
| `prepare_rwkv_panel` | Add RWKV plans with the existing Mamba target IDs and boundaries |
| `prepare_state_attenuation` | Write the state-attenuation scoring commands |

For example, run `python -m long_context_diagnostics.prepare_replication_panel --help`. The shared `matched_text.plan_matched_text` function constructs common target boundaries from source-character spans, and the preparation scripts record a scoring command for each plan. Reconstruct full input-plan NPZ files from the documents and tokenizers; saved result NPZ files are not inference plans.

## Model environments

Use separate environments matching each adapter's recorded runtime:

| Adapter | Measured runtime |
| --- | --- |
| Mamba-2 | torch 2.7.1+cu128; mamba-ssm 2.2.6.post3; causal-conv1d 1.5.3.post1; Triton 3.3.1; transformers 4.44.2 |
| Gated DeltaNet | torch 2.11.0+cu128; FLA/fla-core 0.5.2; Triton 3.6.0; transformers 4.56.2 |
| RWKV-7 | torch 2.7.1+cu128; rwkv 0.8.32; tokenizers 0.19.1; native CUDA FP32 recurrence |

Checkpoint and tokenizer revisions are defined in `mamba_adapter.py`, `gdn_adapter.py`, and `rwkv_adapter.py`. Gated DeltaNet uses the native TF32 triangular solve; its original training runtime is unknown. RWKV uses native CUDA fast-math with Torch TF32 disabled. Record runtime changes with new measurements.

## Check and score

Inspect the numerical-check options for the selected adapter:

```sh
python -m long_context_diagnostics.check_mamba --help
python -m long_context_diagnostics.check_gdn --help
python -m long_context_diagnostics.check_rwkv --help
```

After constructing an input plan and running the relevant check on the target hardware, score it with:

```sh
python -m long_context_diagnostics.score_matched_text \
  --plan path/to/prepared-plan.npz --output new-results/result.npz \
  --solve-precision native --repeat --max-memory-gb 14 \
  --code-revision your-recorded-revision
```

The scorer selects the model from plan metadata, starts each history from fresh state, and normalizes over the complete native output vocabulary. Write new measurements to a separate output directory. The native-token Mamba experiments use `score_native_panel`, `score_local_control` and `score_permutation`, and the state intervention uses `score_state_attenuation`; each lists its options with `--help`.

The complete download-to-inference workflow has not been rerun from this repository. The tested reproduction path is the CPU analysis of the supplied losses. Recomputing extraction and repetition labels requires the original source bytes. Third-party documents and checkpoints retain their respective terms.
