"""Deterministic symbolic associative-retrieval generator.

Defaults reproduce the semantic contract of Qwen v061.  Token-length targeting
is injected as a callback so the generator itself remains tokenizer-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .determinism import stable_int, stable_rng

DIGITS = tuple("1234")
POSITION_RANK_SET = (0, 2, 5, 7)

FILLER_BANK = {
    "en": {
        "subjects": ("The archivist", "The technician", "The editor", "The visitor", "The clerk", "The gardener"),
        "verbs": ("reviewed", "checked", "organized", "recorded", "sorted", "noted"),
        "objects": ("a routine report", "the daily schedule", "a short memo", "the supply list", "a maintenance note", "the meeting agenda"),
        "settings": ("near the window", "in the quiet office", "before lunch", "during the afternoon", "at the front desk", "after the meeting"),
    },
    "tr": {
        "subjects": ("Arşiv görevlisi", "Teknisyen", "Editör", "Ziyaretçi", "Görevli", "Bahçıvan"),
        "verbs": ("inceledi", "kontrol etti", "düzenledi", "kaydetti", "sıraladı", "not etti"),
        "objects": ("sıradan bir raporu", "günlük programı", "kısa bir notu", "malzeme listesini", "bakım notunu", "toplantı gündemini"),
        "settings": ("pencerenin yanında", "sessiz ofiste", "öğle yemeğinden önce", "öğleden sonra", "ön masada", "toplantıdan sonra"),
    },
}


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class AssocInstance:
    filler_language: str
    semantic_seed: int
    trial_index: int
    target_chat_tokens: int
    user_content: str
    spans: tuple[CharSpan, ...]
    target_key: str
    gold_digit: str
    target_record_position: int
    evidence_lines: tuple[str, ...]
    filler_lines: tuple[str, ...]
    query_lines: tuple[str, ...]


def make_key(semantic_seed: int, slot: int, n_records: int = 32) -> str:
    value = stable_int("key_v061", semantic_seed, slot, n_records) % 90000 + 10000
    return f"K{value}"


def gold_for_trial(trial_index: int) -> str:
    return DIGITS[trial_index % 4]


def position_rank_for_trial(trial_index: int) -> int:
    gold_index = trial_index % 4
    replicate = trial_index // 4
    return POSITION_RANK_SET[(replicate + gold_index) % len(POSITION_RANK_SET)]


def target_position_for_trial(trial_index: int, n_records: int = 32) -> int:
    return int(round(position_rank_for_trial(trial_index) * (n_records - 1) / 7))


def filler_sentence(language: str, idx: int, semantic_seed: int) -> str:
    if language not in FILLER_BANK:
        raise ValueError(f"unsupported filler language: {language}")
    bank = FILLER_BANK[language]
    x = idx + 7919 * semantic_seed
    subject = bank["subjects"][x % len(bank["subjects"])]
    x //= len(bank["subjects"])
    verb = bank["verbs"][x % len(bank["verbs"])]
    x //= len(bank["verbs"])
    obj = bank["objects"][x % len(bank["objects"])]
    x //= len(bank["objects"])
    setting = bank["settings"][x % len(bank["settings"])]
    if language == "en":
        return f"{subject} {verb} {obj} {setting}."
    return f"{subject} {setting} {obj} {verb}."


def make_records(
    semantic_seed: int, trial_index: int, n_records: int = 32
) -> tuple[list[tuple[str, str]], str, str, int]:
    if n_records % 4:
        raise ValueError("n_records must be divisible by four")
    gold = gold_for_trial(trial_index)
    values = list(DIGITS) * (n_records // 4)
    gold_idx = values.index(gold)
    values[0], values[gold_idx] = values[gold_idx], values[0]
    records = [(make_key(semantic_seed, slot, n_records), values[slot]) for slot in range(n_records)]
    target, others = records[0], records[1:]
    stable_rng("records_v061", semantic_seed, trial_index).shuffle(others)
    position = target_position_for_trial(trial_index, n_records)
    ordered = others[:]
    ordered.insert(position, target)
    return ordered, target[0], gold, position


def base_evidence_query(
    semantic_seed: int, trial_index: int, n_records: int = 32
) -> tuple[list[str], list[str], str, str, int]:
    records, target_key, gold, position = make_records(semantic_seed, trial_index, n_records)
    evidence = [
        "Task: look up the value for the queried key.",
        "Each evidence line has the form KEY=VALUE.",
        "Return only the VALUE associated with the queried KEY.",
        "Do not explain or repeat the key.",
        "EVIDENCE:",
        *[f"{key}={value}" for key, value in records],
    ]
    query = [f"QUERY KEY: {target_key}", "Answer with one digit only:"]
    return evidence, query, target_key, gold, position


def compose_user_content(
    evidence: Sequence[str], filler: Sequence[str], query: Sequence[str]
) -> tuple[str, tuple[CharSpan, ...]]:
    evidence_text, filler_text, query_text = "\n".join(evidence), "\n".join(filler), "\n".join(query)
    user_content = evidence_text + "\n" + filler_text + "\n" + query_text
    filler_start = len(evidence_text) + 1
    query_start = filler_start + len(filler_text) + 1
    spans = (
        CharSpan(0, filler_start, "evidence"),
        CharSpan(filler_start, query_start, "filler"),
        CharSpan(query_start, len(user_content), "query"),
    )
    return user_content, spans


def build_instance(
    filler_language: str,
    semantic_seed: int,
    trial_index: int,
    target_chat_tokens: int,
    count_chat_tokens: Callable[[str], int],
    verify_chat_shell: Callable[[str], None] | None = None,
    n_records: int = 32,
    max_filler_lines: int = 5000,
) -> AssocInstance:
    evidence, query, target_key, gold, position = base_evidence_query(
        semantic_seed, trial_index, n_records
    )
    filler: list[str] = []
    while True:
        user_content, spans = compose_user_content(evidence, filler, query)
        if count_chat_tokens(user_content) >= target_chat_tokens:
            break
        filler.append(filler_sentence(filler_language, len(filler), semantic_seed))
        if len(filler) > max_filler_lines:
            raise RuntimeError("filler loop failed")
    if verify_chat_shell is not None:
        verify_chat_shell(user_content)
    return AssocInstance(
        filler_language=filler_language,
        semantic_seed=semantic_seed,
        trial_index=trial_index,
        target_chat_tokens=target_chat_tokens,
        user_content=user_content,
        spans=spans,
        target_key=target_key,
        gold_digit=gold,
        target_record_position=position,
        evidence_lines=tuple(evidence),
        filler_lines=tuple(filler),
        query_lines=tuple(query),
    )

