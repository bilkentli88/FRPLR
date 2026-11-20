"""Hugging Face chat/tokenization adapter for the frozen intervention."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..generator import AssocInstance, CharSpan
from ..geometry import UserEncoding, fractional_local_positions


def _normalize_input_ids(obj: Any) -> list[int]:
    if hasattr(obj, "keys") and "input_ids" in obj:
        obj = obj["input_ids"]
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().tolist()
    if (
        isinstance(obj, (list, tuple))
        and len(obj) == 1
        and isinstance(obj[0], (list, tuple))
    ):
        obj = obj[0]
    return [int(item) for item in obj]


def span_label(start: int, end: int, spans: Sequence[CharSpan]) -> str:
    best_label: str | None = None
    best_overlap = 0
    for span in spans:
        overlap = max(0, min(end, span.end) - max(start, span.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = span.label
    if best_label is None:
        raise ValueError(f"chunk [{start},{end}) has no semantic region")
    return best_label


@dataclass
class HFChatAdapter:
    tokenizer: Any
    base_dial: Any
    fragmented_dials: dict[str, Any]
    chat_kwargs: dict[str, Any]
    answer_prefix_ids: Sequence[int] = ()
    candidate_token_ids: Mapping[str, int] | None = None
    shell_extraction: str = "official_token_subsequence_v1"
    encoding_strategy: str = "auto"
    marker: str = "UNIQUE_USER_CONTENT_MARKER_TACL_A93F27"

    def __post_init__(self) -> None:
        if self.shell_extraction == "official_token_subsequence_v1":
            marker_ids = tuple(
                int(x)
                for x in self.tokenizer(self.marker, add_special_tokens=False)["input_ids"]
            )
            full_ids = tuple(self.apply_chat_ids(self.marker))
            if not marker_ids:
                raise RuntimeError("shell marker encoded to no tokens")
            width = len(marker_ids)
            matches = [
                index
                for index in range(len(full_ids) - width + 1)
                if full_ids[index : index + width] == marker_ids
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "could not isolate one user-content token subsequence in chat shell; "
                    f"found {len(matches)} matches"
                )
            marker_index = matches[0]
            self.prefix_ids = full_ids[:marker_index]
            self.official_suffix_ids = full_ids[marker_index + width :]
        elif self.shell_extraction == "rendered_text_boundary_v071":
            rendered = self.render_chat(self.marker)
            if rendered.count(self.marker) != 1:
                raise RuntimeError("could not isolate one marker in rendered chat shell")
            marker_index = rendered.index(self.marker)
            prefix_text = rendered[:marker_index]
            suffix_text = rendered[marker_index + len(self.marker) :]
            self.prefix_ids = tuple(
                int(x)
                for x in self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            )
            self.official_suffix_ids = tuple(
                int(x)
                for x in self.tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
            )
        else:
            raise ValueError(f"unsupported shell extraction: {self.shell_extraction}")
        self.answer_prefix_ids = tuple(int(x) for x in self.answer_prefix_ids)
        self.suffix_ids = self.official_suffix_ids + self.answer_prefix_ids
        self.digit_ids = self._digit_ids()

    def apply_chat_ids(self, user_content: str) -> list[int]:
        output = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            **self.chat_kwargs,
        )
        return _normalize_input_ids(output)

    def render_chat(self, user_content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_kwargs,
        )

    def _digit_ids(self) -> dict[str, int]:
        if self.candidate_token_ids is not None:
            if set(self.candidate_token_ids) != set("1234"):
                raise RuntimeError("configured candidate IDs must have exactly keys 1, 2, 3, 4")
            answer = {digit: int(self.candidate_token_ids[digit]) for digit in "1234"}
            if len(set(answer.values())) != 4:
                raise RuntimeError("configured candidate digit token IDs are not distinct")
            return answer
        answer: dict[str, int] = {}
        for digit in "1234":
            ids = self.tokenizer(digit, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(f"candidate {digit!r} is not one token")
            answer[digit] = int(ids[0])
        if len(set(answer.values())) != 4:
            raise RuntimeError("candidate digit token IDs are not distinct")
        return answer

    def canonical_user_ids(self, user_content: str) -> list[int]:
        tokenizer_ids = [
            int(x)
            for x in self.tokenizer(user_content, add_special_tokens=False)["input_ids"]
        ]
        dial_ids = self.base_dial.pieces_to_ids(self.base_dial.encode_pieces(user_content))
        if tokenizer_ids != dial_ids:
            raise RuntimeError("canonical dial IDs disagree with tokenizer")
        return tokenizer_ids

    def verify_chat_shell(self, user_content: str) -> None:
        reconstructed = (
            list(self.prefix_ids)
            + self.canonical_user_ids(user_content)
            + list(self.official_suffix_ids)
        )
        if reconstructed != self.apply_chat_ids(user_content):
            raise RuntimeError("chat-shell reconstruction failed")

    def count_chat_tokens(self, user_content: str) -> int:
        return len(self.apply_chat_ids(user_content))

    def encode_instance(self, instance: AssocInstance) -> UserEncoding:
        encoding_strategy = getattr(self, "encoding_strategy", "auto")
        if encoding_strategy == "canonical_token_offsets_v1":
            self.last_encoding_strategy = encoding_strategy
            return build_user_encoding_tokenwise(
                instance,
                tokenizer=self.tokenizer,
                canonical_dial=self.base_dial,
                fragmented_dial=self.fragmented_dials[instance.filler_language],
            )
        if encoding_strategy == "pretoken_chunk_v061":
            self.last_encoding_strategy = encoding_strategy
            return build_user_encoding(
                instance,
                tokenizer=self.tokenizer,
                canonical_dial=self.base_dial,
                fragmented_dial=self.fragmented_dials[instance.filler_language],
            )
        if encoding_strategy != "auto":
            raise ValueError(f"unsupported encoding strategy: {encoding_strategy}")

        required = {"evidence", "filler", "query"}
        chunk_labels = {
            span_label(start, end, instance.spans)
            for _, (start, end) in self.base_dial.pre_tokenize(instance.user_content)
        }
        if required.issubset(chunk_labels):
            self.last_encoding_strategy = "pretoken_chunk_v061"
            return build_user_encoding(
                instance,
                tokenizer=self.tokenizer,
                canonical_dial=self.base_dial,
                fragmented_dial=self.fragmented_dials[instance.filler_language],
            )

        # SentencePiece-style backends may expose the complete user message as
        # one pre-tokenization chunk.  Such a chunk cannot carry three semantic
        # region labels.  Fall back to the token-offset construction already
        # used by the frozen Gemma experiment: fragment only canonical tokens
        # lying fully inside the filler span and keep boundary tokens intact.
        self.last_encoding_strategy = "canonical_token_offsets_v1"
        return build_user_encoding_tokenwise(
            instance,
            tokenizer=self.tokenizer,
            canonical_dial=self.base_dial,
            fragmented_dial=self.fragmented_dials[instance.filler_language],
        )


def build_user_encoding(
    instance: AssocInstance,
    tokenizer: Any,
    canonical_dial: Any,
    fragmented_dial: Any,
) -> UserEncoding:
    canonical_ids: list[int] = []
    fragmented_ids: list[int] = []
    canonical_labels: list[str] = []
    fragmented_labels: list[str] = []
    fragmented_source_positions: list[float] = []
    canonical_cursor = 0

    for chunk, (start, end) in canonical_dial.pre_tokenize(instance.user_content):
        label = span_label(start, end, instance.spans)
        canonical_pieces = canonical_dial.encode_chunk_pieces(chunk)
        canonical_chunk_ids = canonical_dial.pieces_to_ids(canonical_pieces)
        canonical_ids.extend(canonical_chunk_ids)
        canonical_labels.extend([label] * len(canonical_chunk_ids))

        if label == "filler":
            fragmented_pieces = fragmented_dial.encode_chunk_pieces(chunk)
            fragmented_chunk_ids = canonical_dial.pieces_to_ids(fragmented_pieces)
            local = fractional_local_positions(canonical_pieces, fragmented_pieces)
            source_positions = [canonical_cursor + float(value) for value in local]
        else:
            fragmented_chunk_ids = canonical_chunk_ids
            source_positions = [
                float(value)
                for value in range(canonical_cursor, canonical_cursor + len(canonical_chunk_ids))
            ]
        fragmented_ids.extend(fragmented_chunk_ids)
        fragmented_labels.extend([label] * len(fragmented_chunk_ids))
        fragmented_source_positions.extend(source_positions)
        canonical_cursor += len(canonical_chunk_ids)

    whole = tokenizer(instance.user_content, add_special_tokens=False)["input_ids"]
    if canonical_ids != [int(value) for value in whole]:
        raise RuntimeError("chunkwise canonical encoding mismatch")
    decoded = tokenizer.decode(
        fragmented_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != instance.user_content:
        raise RuntimeError("fragmented IDs do not decode to identical user content")

    return UserEncoding(
        canonical_ids=tuple(canonical_ids),
        fragmented_ids=tuple(fragmented_ids),
        canonical_labels=tuple(canonical_labels),
        fragmented_labels=tuple(fragmented_labels),
        fragmented_source_positions=tuple(
            float(value) for value in np.asarray(fragmented_source_positions, dtype=np.float32)
        ),
    )


def build_user_encoding_tokenwise(
    instance: AssocInstance,
    tokenizer: Any,
    canonical_dial: Any,
    fragmented_dial: Any,
) -> UserEncoding:
    """Offset-based fallback for tokenizers without useful pre-token chunks.

    This is the package form of the token-level construction used in the
    frozen Gemma v071 experiment.  It leaves any token crossing a semantic
    boundary canonical, so the intervention cannot leak outside the filler.
    """

    encoded = tokenizer(
        instance.user_content,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    canonical_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(canonical_ids) != len(offsets):
        raise RuntimeError("canonical IDs/offsets length mismatch")

    filler_span = next(span for span in instance.spans if span.label == "filler")
    canonical_labels: list[str] = []
    fragmented_ids: list[int] = []
    fragmented_labels: list[str] = []
    fragmented_source_positions: list[float] = []

    for canonical_index, (canonical_id, (start, end)) in enumerate(
        zip(canonical_ids, offsets)
    ):
        label = span_label(start, end, instance.spans)
        canonical_labels.append(label)
        fully_inside_filler = filler_span.start <= start and end <= filler_span.end

        piece = tokenizer.convert_ids_to_tokens(canonical_id)
        if not isinstance(piece, str):
            raise RuntimeError("canonical token did not convert to one piece")
        byte_fallback = re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece) is not None
        special_token = canonical_id in set(getattr(tokenizer, "all_special_ids", ()))

        if (
            label == "filler"
            and fully_inside_filler
            and not byte_fallback
            and not special_token
        ):
            if canonical_dial.pieces_to_ids([piece]) != [canonical_id]:
                raise RuntimeError("canonical token/piece ID disagreement")
            fragmented_pieces = fragmented_dial.encode_chunk_pieces(piece)
            if "".join(fragmented_pieces) != piece:
                raise RuntimeError(
                    f"fragmented pieces do not reconstruct canonical piece {piece!r}"
                )
            token_ids = canonical_dial.pieces_to_ids(fragmented_pieces)
            width = len(token_ids)
            if width < 1:
                raise RuntimeError("empty token-level fragmentation")
            source_positions = [
                canonical_index + rank / width for rank in range(width)
            ]
        else:
            token_ids = [canonical_id]
            source_positions = [float(canonical_index)]

        fragmented_ids.extend(token_ids)
        fragmented_labels.extend([label] * len(token_ids))
        fragmented_source_positions.extend(source_positions)

    canonical_reference = [
        int(value)
        for value in tokenizer(
            instance.user_content, add_special_tokens=False
        )["input_ids"]
    ]
    if canonical_ids != canonical_reference:
        raise RuntimeError("offset and ordinary canonical encodings disagree")

    canonical_decoded = tokenizer.decode(
        canonical_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    fragmented_decoded = tokenizer.decode(
        fragmented_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if fragmented_decoded != canonical_decoded:
        raise RuntimeError(
            "tokenwise fragmented IDs do not decode identically to canonical IDs"
        )

    result = UserEncoding(
        canonical_ids=tuple(canonical_ids),
        fragmented_ids=tuple(fragmented_ids),
        canonical_labels=tuple(canonical_labels),
        fragmented_labels=tuple(fragmented_labels),
        fragmented_source_positions=tuple(
            float(value)
            for value in np.asarray(fragmented_source_positions, dtype=np.float32)
        ),
    )
    result.validate()
    return result
