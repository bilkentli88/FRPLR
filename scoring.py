"""Candidate-margin scorers with sequential causal visibility."""

from __future__ import annotations

import gc
import inspect
import time
from typing import Any, Mapping, Sequence

from .geometry import Condition


def _finish_scores(
    logits: Any,
    gold_digit: str,
    digit_ids: Mapping[str, int],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    digits = tuple("1234")
    candidate_ids = torch.tensor(
        [digit_ids[digit] for digit in digits], dtype=torch.long, device=logits.device
    )
    candidate_logits = logits.float().index_select(0, candidate_ids)
    if not torch.isfinite(candidate_logits).all():
        values = candidate_logits.detach().float().cpu().tolist()
        raise RuntimeError(f"non-finite candidate logits: {values}")
    candidate_logp = functional.log_softmax(candidate_logits, dim=-1)
    candidate_probability = candidate_logp.exp()
    gold_index = digits.index(gold_digit)
    distractors = [index for index in range(4) if index != gold_index]
    best_index = max(distractors, key=lambda index: float(candidate_logits[index].item()))
    predicted_index = int(torch.argmax(candidate_logits).item())
    full_argmax_id = int(torch.argmax(logits).item())
    full_argmax_digit = next(
        (digit for digit, token_id in digit_ids.items() if token_id == full_argmax_id), None
    )
    output = {
        "predicted_digit": digits[predicted_index],
        "correct": int(digits[predicted_index] == gold_digit),
        "gold_logit_margin": float(
            (candidate_logits[gold_index] - candidate_logits[best_index]).item()
        ),
        "gold_logit": float(candidate_logits[gold_index].item()),
        "best_distractor_digit": digits[best_index],
        "best_distractor_logit": float(candidate_logits[best_index].item()),
        "forced_choice_nll": float(-candidate_logp[gold_index].item()),
        "gold_candidate_probability": float(candidate_probability[gold_index].item()),
        "full_argmax_id": full_argmax_id,
        "full_argmax_digit": full_argmax_digit,
        "full_argmax_is_gold": int(full_argmax_digit == gold_digit),
        "candidate_logits": {
            digit: float(candidate_logits[index].item()) for index, digit in enumerate(digits)
        },
        "candidate_probabilities": {
            digit: float(candidate_probability[index].item()) for index, digit in enumerate(digits)
        },
    }
    if tokenizer is not None:
        output["full_argmax_text"] = tokenizer.decode(
            [full_argmax_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output


def score_one_shot(
    model: Any,
    condition: Condition,
    gold_digit: str,
    digit_ids: Mapping[str, int],
    tokenizer: Any | None = None,
    scorer_version: str = "one_shot_no_cache_float_positions_v1",
) -> dict[str, Any]:
    """Score one prompt without a cache; causal order follows token sequence."""

    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor(condition.ids, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(
        condition.positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    forward_parameters = inspect.signature(model.forward).parameters
    logit_kwargs: dict[str, int] = {}
    if "logits_to_keep" in forward_parameters:
        logit_kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in forward_parameters:
        logit_kwargs["num_logits_to_keep"] = 1
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **logit_kwargs,
        )
        scored = _finish_scores(output.logits[0, -1], gold_digit, digit_ids, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        scored["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    scored["seconds"] = time.perf_counter() - started
    scored["scorer_version"] = scorer_version
    del input_ids, position_ids, attention_mask, output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def score_qwen_one_shot_sequential(
    model: Any,
    condition: Condition,
    gold_digit: str,
    digit_ids: Mapping[str, int],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Qwen one-shot/no-cache scorer with explicit sequential visibility."""

    import torch

    backbone = model.model
    device = next(model.parameters()).device
    input_ids = torch.tensor(condition.ids, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(
        condition.positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    width = input_ids.shape[1]
    query = torch.arange(width, device=device)[:, None]
    key = torch.arange(width, device=device)[None, :]
    causal = (key <= query)[None, None, :, :]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = backbone(
            input_ids=input_ids,
            attention_mask={"full_attention": causal},
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()[0]
        scored = _finish_scores(logits, gold_digit, digit_ids, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        scored["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    scored["seconds"] = time.perf_counter() - started
    scored["scorer_version"] = "qwen_oneshot_nocache_sequential_mask_v1"
    del input_ids, position_ids, query, key, causal, output, logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def score_gemma3_one_shot_sequential(
    model: Any,
    condition: Condition,
    gold_digit: str,
    digit_ids: Mapping[str, int],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Gemma 3 one-shot scorer with explicit full/sliding causal masks."""

    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor(condition.ids, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(
        condition.positions, dtype=torch.float32, device=device
    ).unsqueeze(0)
    width = input_ids.shape[1]
    text_config = getattr(model.config, "text_config", model.config)
    sliding_window = int(text_config.sliding_window)
    query = torch.arange(width, device=device)[:, None]
    key = torch.arange(width, device=device)[None, :]
    full_visible = key <= query
    # This is equivalent to ``(query - key) < sliding_window`` for integer
    # indices, but avoids materializing a width-by-width int64 difference.
    sliding_visible = full_visible & (key > (query - sliding_window))
    mask_dtype = torch.bfloat16
    minimum = torch.finfo(mask_dtype).min
    full = torch.full(
        (width, width), minimum, dtype=mask_dtype, device=device
    ).masked_fill_(full_visible, 0.0)[None, None, :, :]
    local = torch.full(
        (width, width), minimum, dtype=mask_dtype, device=device
    ).masked_fill_(sliding_visible, 0.0)[None, None, :, :]
    attention_mask = {"full_attention": full, "sliding_attention": local}
    if not torch.isfinite(full).all() or not torch.isfinite(local).all():
        raise RuntimeError("non-finite Gemma sequential mask")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    forward_parameters = inspect.signature(model.forward).parameters
    logit_kwargs: dict[str, int] = {}
    if "logits_to_keep" in forward_parameters:
        logit_kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in forward_parameters:
        logit_kwargs["num_logits_to_keep"] = 1
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **logit_kwargs,
        )
        scored = _finish_scores(output.logits[0, -1], gold_digit, digit_ids, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        scored["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    scored["seconds"] = time.perf_counter() - started
    scored["scorer_version"] = "gemma3_oneshot_nocache_sequential_masks_v1"
    del (
        input_ids,
        position_ids,
        query,
        key,
        full_visible,
        sliding_visible,
        attention_mask,
        full,
        local,
        output,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def score_qwen_chunked_v061(
    model: Any,
    condition: Condition,
    gold_digit: str,
    digit_ids: Mapping[str, int],
    tokenizer: Any | None = None,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Exact historical v061 Qwen cache/chunk contract."""

    import torch
    from transformers import DynamicCache

    backbone = model.model
    device = next(model.parameters()).device
    ids_all = torch.tensor(condition.ids, dtype=torch.long, device=device)
    pos_all = torch.tensor(condition.positions, dtype=torch.float32, device=device)
    cache = DynamicCache(config=model.config)
    last_hidden = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        total = ids_all.numel()
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            query = torch.arange(start, end, device=device)[:, None]
            key = torch.arange(0, end, device=device)[None, :]
            causal = (key <= query)[None, None, :, :]
            output = backbone(
                input_ids=ids_all[start:end].unsqueeze(0),
                attention_mask={"full_attention": causal},
                position_ids=pos_all[start:end].unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            if end == total:
                last_hidden = output.last_hidden_state[:, -1, :]
        logits = model.lm_head(last_hidden).float()[0]
        scored = _finish_scores(logits, gold_digit, digit_ids, tokenizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        scored["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    scored["seconds"] = time.perf_counter() - started
    scored["scorer_version"] = "assoc_chunked_margin_v061"
    del ids_all, pos_all, cache, last_hidden
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def flatten_score(scored: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in scored.items()
        if key not in {"candidate_logits", "candidate_probabilities"}
    }
    for digit, value in scored["candidate_logits"].items():
        output[f"logit_{digit}"] = value
    for digit, value in scored["candidate_probabilities"].items():
        output[f"prob_{digit}"] = value
    return output
