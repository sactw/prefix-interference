# Prefix interference in recurrent language models

Code and numerical results for **Separating Distant-Content Benefits from Prefix Interference in Recurrent Language Models**, by Sotirios Sakaretsios (Codewire), October 2026.

The study compares next-token losses under original, truncated, and foreign histories in public Mamba-2, Gated DeltaNet, and RWKV-7 checkpoints. This repository includes model adapters, preparation tools, per-target losses, target IDs and positions, repetition labels, and source metadata.

## Quickstart

Use Python 3.11 or newer. From the repository root:

```sh
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then run:

```sh
python -m pip install -r requirements-cpu.txt
python reproduce.py --output reproduced
```

This recomputes every reported numerical panel from the included losses, using 2,000 paired document-bootstrap resamples and seed 20261020. After installing NumPy, it requires no network access, model weights, or GPU.

To regenerate the three paper figures:

```sh
python -m pip install -r requirements-figures.txt
python reproduce.py --output reproduced --figures
```

## Repository layout

```text
long_context_diagnostics/   Model adapters, scorers, preparation and readouts
tests/                      CPU tests
results/
  fresh_panel/             Native-token grid
  local_control/           Short-context control
  length_ladder/           Fixed-target context lengths
  permutation/             Older-block permutation
  cross_donor/             Alternate-prefix sensitivity
  matched_panel/           Original matched-text cohort
  second_donor/            Second-donor comparison
  replication/             Additional-report cohort
  state_attenuation/       Recurrent-state intervention
  rwkv_panel/              RWKV and three-model readouts
  numerical_checks/        Saved numerical checks
```

Per-target arrays are in each experiment's `losses/` directory. Experiment
dates and original artifact identifiers are retained in the metadata. The
planning records (`PLAN.json`, `inspection_notes.json`) were written before
scoring and are kept unchanged.

## Results

`reproduced/REPRODUCED.json` lists the generated reports and figures. Reports are grouped by experiment:

| Output under `reproduced/` | Contents |
| --- | --- |
| `fresh_panel/RESULTS.json` | Native-token grid |
| `length_ladder/RESULTS.json` | Fixed-target length effects |
| `permutation/RESULTS.json` | Older-block permutation effects |
| `rwkv_panel/original8_RESULTS.json` | Three-model comparison on the original eight reports |
| `rwkv_panel/replication4_RESULTS.json` | Three-model comparison on four additional reports |
| `figures/` | PDF and PNG plots when `--figures` is enabled |

The command also writes the local-control, cross-donor, second-donor, matched-text, and state-attenuation analyses. The eight-report and four-report cohorts remain separate. Matched-text effects use nats per UTF-8 target byte; native-token panels use nats per target token. The native grid averages its two fixed donor observations within each document before resampling.

## Tests

Run the CPU tests for history construction, target alignment, recurrence scheduling, and statistical readouts:

```sh
python -m unittest discover -s tests -t .
```

GPU kernel checks are separate; see [INFERENCE.md](INFERENCE.md).

## Scope and data

The quickstart recomputes statistics from saved model outputs. Independent inference requires the source PDFs, pinned tokenizers, and checkpoints described in [INFERENCE.md](INFERENCE.md). Full documents, full conditioning-token arrays, and model weights are not included. Retained target IDs encode the short scored passages.

Results concern selected checkpoints and small report cohorts, largely beyond the training context lengths. Unequal model sizes, training, and runtimes preclude an architecture ranking or a unique capacity explanation.

Source URLs, extraction settings, and source and tokenizer hashes are recorded in `results/sources_used.json` and the experiment metadata. Use [CITATION.cff](CITATION.cff) for citation metadata. The [MIT license](LICENSE) covers the original code and the author-created numerical results; third-party models, dependencies and source text, including text recoverable from the target token IDs, keep their own terms.
