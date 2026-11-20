# Reproduction guide

## 1. Exact CPU reproduction (recommended)

```bash
pip install -r requirements-analysis.txt
python analysis/reproduce_reported_results.py --write
python analysis/acl_accuracy_reanalysis.py .
python -m unittest discover -s tests -v
```

The first script reads the nine committed JSONL files, validates complete paired
conditions and unique result IDs, reconstructs the margin estimands, and repeats
the deterministic 10,000-replicate trial-cluster bootstrap. The second script
reports factorial accuracy for all four models, paired accuracy contrasts and
intervals, Gemma's CE predicted-digit distribution, and localization accuracy
for Qwen and Gemma. Both require no network, GPU, model weights, or access token.
The added script uses only existing recorded predictions: it is an analysis
extension, not a new model run.

The four main factorial targets are stored in their corresponding files under
`summaries/`. The original Qwen/Gemma and localization targets are retained in
`summaries/original_experiments_reported_results.json`.

## 2. Full GPU reruns

The four notebooks in `notebooks/factorial/` reproduce the revised four-model
factorial experiments. Each notebook loads a frozen configuration from the exact
v5.1 harness, checks its engineering or qualification audit, validates the chat
shell and factorial invariants, and writes an append-only JSONL ledger.

The v5.1 harness is available in two forms:

- installable artifact: `code/TACL_Expansion_Harness_v5_1.zip`;
- readable source: `src/tacl_factorization/`.

The retokenization dependency is vendored as `code/fragburden0.3.0.zip`.

### Runtime

All four main factorial experiments used Transformers 5.15.1, NF4 weights,
BF16 compute, SDPA attention, and one-shot/no-cache scoring on an NVIDIA A100
40GB. Exact model and tokenizer revisions are recorded in the row provenance and
engineering audits. Gemma requires accepted Hugging Face access and an
`HF_TOKEN` secret.

The historical Qwen/Gemma factorial and localization notebooks retain their
individually validated execution paths. Their raw magnitudes are interpreted
within model and are not placed on a common cross-model scale.

## 3. Provenance chain

For the new experiments, each raw row records the frozen configuration hash,
runtime fingerprint, result ID, token-sequence hash, position hash, prompt hash,
and the hash of the qualifying audit. The corresponding audit and frozen config
are committed under `audits/` and `configs/`.

GPU kernels can introduce small numeric variation. The immutable committed rows,
rather than bit-identical regeneration on arbitrary hardware, define the exact
article outputs reproduced by the CPU path.

## Excluded development material

Calibration scratch files, notebook repair versions, model downloads,
credentials, and the abandoned natural-language pilot are intentionally absent.
They are not required to reproduce any claim in the manuscript.
