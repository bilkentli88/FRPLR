# Stage 2: harmonized Qwen/Gemma robustness protocol

Frozen: 2026-09-13 UTC, before running seeds 9000--9015.

## Purpose

Stage 1 added Mistral and Phi under a common one-shot, no-cache BF16/NF4
path. Stage 2 addresses the remaining inference-path caveat in the submitted
manuscript by rerunning Qwen and Gemma under matched high-level inference
choices: NF4 weights, BF16 computation, SDPA, one-shot execution, no KV cache,
and visibility determined by sequential token order rather than experimental
position IDs.

The architecture-specific mask representation is not treated as an
experimental difference. Qwen consumes its validated boolean full-attention
mask; Gemma consumes the validated finite additive masks required for its full
and sliding-attention layers.

## Locked design

- Models: `Qwen/Qwen3-4B` and `google/gemma-3-4b-it`.
- Seeds: 9000--9015, unused in the submitted experiments.
- Cells: `CC`, `CE`, `FC`, and `FE`.
- Strata: English/Turkish filler by 4K/8K target length.
- Rows: 16 seeds x 2 languages x 2 lengths x 4 conditions = 256 per model.
- Metric: gold digit logit minus the strongest distractor digit logit.
- Estimation: the six frozen factorial contrasts and 10,000-replicate
  percentile intervals resampling `trial_index` clusters.
- Stopping: all 16 seeds; no optional stopping or outcome-based replacement.

The tokenizer interventions and fragmentation strengths remain those already
frozen for each model. Qwen uses the original v061 pre-token-chunk encoder
(`en=2`, `tr=5`). Gemma uses the original v071 token-offset encoder and keeps
byte-fallback/special tokens canonical (`en=1`, `tr=5`).

## Engineering preflight

Seed 8999 is reserved exclusively for engineering and excluded from analysis.
Before confirmatory rows can be written, each notebook checks the frozen
configuration, resolved model/tokenizer revisions, tokenizer and chat-shell
contracts, geometry invariants, native-versus-explicit candidate argmax,
deterministic repeat scoring, sensitivity to a fractional position change,
and finite completion of the hardest 8K `FE` path. The audit stores only gate
booleans and scalar maximum differences; it does not store or print task
predictions or candidate logits.

These gates can reject an invalid execution path, but cannot select a model,
fragmentation strength, seed, direction, or scientific result.

## Frozen interpretation

All six effects and their uncertainty intervals will be reported regardless
of direction or significance. Direction agreement with the historical runs is
descriptive robustness evidence, not a condition for inclusion. Gemma's
qualitative cancellation is additionally summarized as

`abs(FE-CC) / (abs(CE-CC) + abs(FE-CE))`.

A value at or below 0.25 is the predeclared qualitative-cancellation criterion.
Failure to meet it is retained and reported.
