# When Tokenization Effects Cancel — ACL/ARR reproducibility package

> **ACL/ARR review release candidate — not yet publicly published.**
> This is a reproducibility package for anonymous review. Share only as an
> anonymized supplementary archive or through a permitted anonymous repository.
> Do not link a personal GitHub account or release the original Git history.
> Complete the release/anonymity checks in `docs/ACL_REVIEW_RELEASE.md` first.

This repository contains the frozen configurations, engineering audits, raw
outputs, notebooks, and analysis code for:

> *When Tokenization Effects Cancel: Factorizing Representation and Position in
> Long-Context Retrieval*

The package is deliberately narrower than the development archive. It excludes
credentials, model weights, calibration scratch files, failed implementation
attempts, obsolete notebook revisions, and the abandoned natural-language pilot.

## Reproduce every reported number without a GPU

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
python analysis/reproduce_reported_results.py --write
python analysis/acl_accuracy_reanalysis.py .
python -m unittest discover -s tests -v
```

The original analysis script reconstructs all reported logit-margin estimands and
10,000-replicate semantic-trial cluster-bootstrap intervals, checking the
results against committed summaries. The ACL accuracy script separately
recomputes four-condition accuracy, paired accuracy contrasts and intervals,
Gemma's predicted-digit distribution, and six-condition localization accuracy
from the already committed predictions. Neither path requires a GPU.
The margin script finishes with:

```text
PASS: all reported numerical effects reproduce exactly from the committed raw rows.
```

## Evidence included

| Role in revised paper | Model | Raw output | GPU notebook |
|---|---|---|---|
| Main four-model factorial | Qwen3-4B | `data/qwen/qwen_harmonized_robustness_v1.jsonl` | `notebooks/factorial/qwen_harmonized_robustness.ipynb` |
| Main four-model factorial | Gemma 3 4B IT | `data/gemma/gemma_harmonized_robustness_v1.jsonl` | `notebooks/factorial/gemma_harmonized_robustness.ipynb` |
| Main four-model factorial | Phi-4-mini-instruct | `data/phi/phi_factorial_confirmatory_v1.jsonl` | `notebooks/factorial/phi_confirmatory_factorial.ipynb` |
| Main four-model factorial | Mistral-7B-Instruct-v0.3 | `data/mistral/mistral_factorial_confirmatory_v1.jsonl` | `notebooks/factorial/mistral_confirmatory_factorial.ipynb` |
| Main localization | Qwen3-4B | `data/qwen/v062_geometry_localization_4k8k.jsonl` | `notebooks/qwen/v062_localization.ipynb` |
| Main localization | Gemma 3 4B IT | `data/gemma/v072_gemma_geometry_localization_4k8k.jsonl` | `notebooks/gemma/v072_localization.ipynb` |
| Appendix: initial failed rescue | Qwen3-4B | `data/qwen/v060_mechanism_4k8k.jsonl` | `notebooks/qwen/v060_failed_rescue.ipynb` |
| Appendix: original factorial | Qwen3-4B | `data/qwen/v061_factorial_4k8k.jsonl` | `notebooks/qwen/v061_factorial.ipynb` |
| Appendix: original factorial | Gemma 3 4B IT | `data/gemma/v071_gemma_factorial_4k8k.jsonl` | `notebooks/gemma/v071_factorial.ipynb` |

The Qwen/Gemma harmonized experiments use 16 fresh semantic seeds per model
(256 rows each). The prospectively frozen Phi/Mistral experiments use 32 seeds
per model (512 rows each). The two localization inventories use 32 seeds per
model (768 rows each).

## ACL accuracy additions (no new experiments)

The revised paper adds forced-choice accuracy for all four factorial conditions,
paired Gemma accuracy contrasts with trial-cluster bootstrap intervals, and the
accuracy counterpart to the independent Qwen/Gemma localization runs. These
numbers are now reproduced by `analysis/acl_accuracy_reanalysis.py`.
See `docs/ARTICLE_RESULT_MAP.md` for the matching raw files and reported values.
The published experiment outputs, frozen configurations, audits, and harness
are unchanged from the original repository.

## Full model reruns

The notebooks are clean, output-free Colab copies. The four new factorial
notebooks use the exact `code/TACL_Expansion_Harness_v5_1.zip` artifact and the
vendored `code/fragburden0.3.0.zip` retokenization package. Frozen JSON
configurations are also readable under `configs/`, and the v5.1 harness source is
available under `src/tacl_factorization/`.

All four main factorial runs use Transformers 5.15.1, NF4 weights, BF16 compute,
SDPA, and one-shot/no-cache scoring. Gemma requires accepted Hugging Face access
and an `HF_TOKEN` secret. Recorded runs used an NVIDIA A100 40GB. Model weights
are not redistributed.

GPU-level logits may vary slightly across hardware and kernels. The committed
raw outputs are the frozen article outputs; the CPU analysis path is the exact
reproduction target for the paper's tables and confidence intervals.

## Repository structure

```text
analysis/     CPU scripts that recompute reported statistics
audits/       passed engineering/qualification gates for the new runs
code/         exact installable harness and retokenization artifacts
configs/      frozen experiment configurations
data/         immutable JSONL rows used in the manuscript
docs/         result map, data dictionary, and reproduction guide
notebooks/    clean GPU rerun notebooks
src/          readable source for the v5.1 factorial harness
summaries/    committed numerical targets and diagnostics
tests/        row-count, invariant, and exact-reproduction tests
```

## Scope and release status

The experiments concern controlled long-context associative retrieval. English
and Turkish are task-irrelevant filler regimes, not comparative language-
understanding tasks. The paper does not claim that increased positional distance
is universally beneficial or harmful.

Before any release, follow `docs/ACL_REVIEW_RELEASE.md`. Author names, personal
paths, links, notebook metadata, and Git history must be screened for anonymity.
Choose a final license and citation metadata deliberately; neither is supplied
by this draft review package. The original archived filenames retain `TACL`
for provenance, not because this package is intended for TACL submission.
After acceptance, update the public-facing repository URL and manuscript DOI.
