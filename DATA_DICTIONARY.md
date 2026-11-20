# Data dictionary

Each JSONL row is one scored condition for one semantic instance.

## Identifiers and pairing

The pairing key is `filler_language`, `target_chat_tokens`, `semantic_seed`, and
`trial_index`. `result_id` uniquely identifies a scored condition. The four
factorial codes are:

- `CC`: canonical token IDs, source-aligned geometry;
- `CE`: canonical token IDs, expanded geometry;
- `FC`: fragmented token IDs, source-aligned geometry;
- `FE`: fragmented token IDs, expanded geometry.

Localization additionally uses `CQ` and `FQ`, which shift query/readout
coordinates while leaving filler geometry source-aligned. `CE` and `FE` then add
filler spreading.

## Outcome fields

`gold_logit_margin` is the gold-candidate score minus the strongest distractor
score. `correct` and `predicted_digit` record the forced-choice decision.
`forced_choice_nll` is retained for diagnostics and is the predeclared endpoint
only in the initial v060 rescue experiment. Mistral uses verified contextual
multi-token completions; its candidate interface is audit-bound.

## Intervention checks

`ids_hash` and `position_hash` make the same-token-ID and matched-geometry
contrasts directly auditable. Geometry diagnostics include
`evidence_query_position_gap`, `evidence_readout_position_gap`,
`filler_position_span`, `position_max`, and `realized_full_tokens`.

## Provenance

New-run rows also record configuration and audit hashes, runtime fingerprints,
resolved model metadata, scorer version, task version, and prompt/evidence/query
hashes. No prompts contain personal or participant data; all instances are
synthetically generated.
