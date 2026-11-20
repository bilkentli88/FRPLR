# Article result map

## Main four-model factorial table

| Model | Data | Frozen config | Audit | Notebook | Summary |
|---|---|---|---|---|---|
| Qwen3-4B | `data/qwen/qwen_harmonized_robustness_v1.jsonl` | `configs/qwen_harmonized_robustness_v1.json` | `audits/qwen_harmonized_engineering_audit_v1.json` | `notebooks/factorial/qwen_harmonized_robustness.ipynb` | `summaries/qwen_harmonized_robustness_v1_summary.json` |
| Gemma 3 4B IT | `data/gemma/gemma_harmonized_robustness_v1.jsonl` | `configs/gemma_harmonized_robustness_v1.json` | `audits/gemma_harmonized_engineering_audit_v1.json` | `notebooks/factorial/gemma_harmonized_robustness.ipynb` | `summaries/gemma_harmonized_robustness_v1_summary.json` |
| Phi-4-mini-instruct | `data/phi/phi_factorial_confirmatory_v1.jsonl` | `configs/phi_factorial_confirmatory_v1.json` | `audits/phi_engineering_audit.json` | `notebooks/factorial/phi_confirmatory_factorial.ipynb` | `summaries/phi_factorial_confirmatory_v1_summary.json` |
| Mistral-7B-Instruct-v0.3 | `data/mistral/mistral_factorial_confirmatory_v1.jsonl` | `configs/mistral_factorial_confirmatory_v1.json` | `audits/mistral_engineering_audit.json` | `notebooks/factorial/mistral_confirmatory_factorial.ipynb` | `summaries/mistral_factorial_confirmatory_v1_summary.json` |

## Main localization figure

| Claim | Data | Notebook |
|---|---|---|
| Qwen query/readout localization | `data/qwen/v062_geometry_localization_4k8k.jsonl` | `notebooks/qwen/v062_localization.ipynb` |
| Gemma opposing displacement/spreading components | `data/gemma/v072_gemma_geometry_localization_4k8k.jsonl` | `notebooks/gemma/v072_localization.ipynb` |

## Appendix evidence

| Claim | Data | Notebook |
|---|---|---|
| Initial rescue hypothesis was inconclusive | `data/qwen/v060_mechanism_4k8k.jsonl` | `notebooks/qwen/v060_failed_rescue.ipynb` |
| Original Qwen factorial | `data/qwen/v061_factorial_4k8k.jsonl` | `notebooks/qwen/v061_factorial.ipynb` |
| Original Gemma factorial | `data/gemma/v071_gemma_factorial_4k8k.jsonl` | `notebooks/gemma/v071_factorial.ipynb` |

`analysis/reproduce_reported_results.py` verifies all mapped numerical effects.


## ACL revision: accuracy, prediction distributions, and confidence intervals

The original logit-margin statistics and frozen experiment outputs are unchanged.
The revised ACL paper also reports the following results, computed from the
same JSONL predictions with `python analysis/acl_accuracy_reanalysis.py .`:

| Result in revised ACL paper | Source | Main fields |
|---|---|---|
| Four-model factorial accuracy (CC/CE/FC/FE) | All four `main_factorial` files above | `factorial_code`, `correct` |
| Gemma paired CE--CC, FE--CE, FE--CC accuracy contrasts and cluster intervals | `data/gemma/gemma_harmonized_robustness_v1.jsonl` | `trial_index`, `semantic_seed`, `filler_language`, `target_chat_tokens`, `factorial_code`, `correct` |
| Gemma CE response bias (62 of 64 predictions select digit 1) | `data/gemma/gemma_harmonized_robustness_v1.jsonl` | `factorial_code`, `predicted_digit` |
| Qwen/Gemma six-condition localization accuracy | `data/qwen/v062_geometry_localization_4k8k.jsonl`, `data/gemma/v072_gemma_geometry_localization_4k8k.jsonl` | `geometry_code`, `correct` |

Gemma accuracy by factorial condition: CC 95.3125%, CE 21.875%, FC 89.0625%, FE 87.5%.
The paired Gemma contrasts (percentage points, 10,000-replicate trial-cluster
bootstrap) are CE--CC -73.4375 [-93.75, -51.5625], FE--CE +65.625
[+35.9375, +89.0625], and FE--CC -7.8125 [-18.75, 0.0]. These values
are descriptive and inferential summaries of pre-existing records, not new runs.
