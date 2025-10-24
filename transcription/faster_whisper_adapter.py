"""Adapter that exposes Faster-Whisper transcriptions in an AWS-compatible format."""

from __future__ import annotations

import contextlib
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from types import MethodType
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from faster_whisper import WhisperModel
from faster_whisper.transcribe import get_compression_ratio

# Character sets copied from Faster-Whisper defaults so that we split punctuation
# the same way that Whisper expects during decoding.
_LEADING_PUNCTUATION = set("\"'“¿([{-")
_TRAILING_PUNCTUATION = set("\"'.。,，!！?？:：”)]}、")
_TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


@dataclass
class _WordUnit:
    """Normalized representation of a word item."""

    text: str
    start: float
    end: float
    confidence: float


def _format_time(value: Optional[float]) -> str:
    return f"{float(value):.2f}" if value is not None else "0.00"


def _format_confidence(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value))):.4f}"


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exp_scores = [math.exp(v - max_val) for v in values]
    total = sum(exp_scores)
    if total == 0:
        return [0.0 for _ in exp_scores]
    return [score / total for score in exp_scores]


def _split_word_parts(token: str) -> List[Tuple[str, str]]:
    """Split a Whisper word token into punctuation and word parts."""

    stripped = token.strip()
    if not stripped:
        return []

    parts: List[Tuple[str, str]] = []
    while stripped and stripped[0] in _LEADING_PUNCTUATION:
        parts.append(("punctuation", stripped[0]))
        stripped = stripped[1:]

    trailing: List[str] = []
    while stripped and stripped[-1] in _TRAILING_PUNCTUATION:
        trailing.append(stripped[-1])
        stripped = stripped[:-1]

    if stripped:
        parts.append(("word", stripped))

    for char in reversed(trailing):
        parts.append(("punctuation", char))

    return parts


def _tokenize_words(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_PATTERN.findall(text.strip()):
        normalized = token.strip()
        if not normalized:
            continue
        if any(ch.isalnum() for ch in normalized):
            tokens.append(normalized)
    return tokens


def _normalize_word(token: str) -> str:
    text = token.strip()
    while text and text[0] in _LEADING_PUNCTUATION:
        text = text[1:]
    while text and text[-1] in _TRAILING_PUNCTUATION:
        text = text[:-1]
    return text.strip()


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int], start: int) -> Optional[int]:
    if not subsequence:
        return start
    max_start = len(sequence) - len(subsequence)
    for idx in range(start, max_start + 1):
        if list(sequence[idx : idx + len(subsequence)]) == list(subsequence):
            return idx
    return None


@contextlib.contextmanager
def _capture_beam_results(model: WhisperModel) -> Iterator[List["ctranslate2.models.WhisperGenerationResult"]]:
    """Patches ``generate_with_fallback`` so beam hypotheses are retained."""

    from faster_whisper.transcribe import TranscriptionOptions

    beam_results: List["ctranslate2.models.WhisperGenerationResult"] = []
    original_method = model.generate_with_fallback

    def patched(
        self: WhisperModel,
        encoder_output,
        prompt,
        tokenizer,
        options: TranscriptionOptions,
    ):
        final_result = None
        all_results = []
        below_cr_threshold_results = []

        temperatures = list(options.temperatures)
        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.max_length
        if max_length > self.max_length:
            raise ValueError(
                "The prompt length exceeds the Whisper model maximum length."
            )

        for temperature in temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": max(1, options.beam_size),
                    "num_hypotheses": max(1, options.beam_size),
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]
            seq_len = len(tokens)
            cumulative_logprob = result.scores[0] * (seq_len ** options.length_penalty)
            avg_logprob = cumulative_logprob / (seq_len + 1)
            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            decode_result = (result, avg_logprob, temperature, compression_ratio)
            all_results.append(decode_result)
            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = True

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False

            if not needs_fallback:
                final_result = decode_result
                break

        if final_result is None:
            fallback_choice = below_cr_threshold_results or all_results
            final_result = max(fallback_choice, key=lambda entry: entry[1])
            final_result = (
                final_result[0],
                final_result[1],
                temperatures[-1] if temperatures else 0.0,
                final_result[3],
            )

        beam_results.append(final_result[0])
        return final_result

    model.generate_with_fallback = MethodType(patched, model)
    try:
        yield beam_results
    finally:
        model.generate_with_fallback = original_method


class FasterWhisperTranscriber:
    """High-level transcription API that mimics AWS Transcribe JSON output."""

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "auto",
        compute_type: str = "float16",
        **model_kwargs,
    ) -> None:
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            **model_kwargs,
        )

    def transcribe_to_aws(
        self,
        audio_path: str,
        *,
        job_name: str = "faster-whisper-job",
        account_id: str = "000000000000",
        beam_size: int = 5,
        best_of: Optional[int] = None,
        vad_filter: bool = True,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Dict[str, object]:
        """Transcribe ``audio_path`` and return an AWS Transcribe-style dict."""

        best_of = best_of or max(1, beam_size)

        with _capture_beam_results(self.model) as beam_results:
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                word_timestamps=True,
                vad_filter=vad_filter,
            )
            segments = list(segments_iter)

        transcript_text = " ".join(seg.text.strip() for seg in segments).strip()
        tokenizer = self.model.hf_tokenizer

        segment_beam_map = self._align_segments_with_beams(segments, beam_results)
        aws_items: List[Dict[str, object]] = []

        for segment in segments:
            alignment = segment_beam_map.get(segment.id)
            if alignment is None:
                continue
            beam_index, start_token, end_token = alignment
            beam_result = beam_results[beam_index]
            word_items, placeholder_items = self._build_word_items(segment)
            if not word_items:
                aws_items.extend(placeholder_items)
                continue

            alternatives = self._compute_alternatives(
                word_items,
                beam_result,
                start_token,
                end_token,
                tokenizer,
            )

            for item in placeholder_items:
                if item["type"] != "pronunciation":
                    aws_items.append(item)
                    continue
                word_index = item.pop("word_index")
                item["alternatives"] = alternatives[word_index]
                aws_items.append(item)

        result: Dict[str, object] = {
            "jobName": job_name,
            "accountId": account_id,
            "status": "COMPLETED",
            "results": {
                "transcripts": [{"transcript": transcript_text}],
                "items": aws_items,
            },
        }

        if getattr(info, "language", None):
            result["results"]["language_code"] = info.language
        if getattr(info, "duration", None) is not None:
            result["results"]["audio_duration"] = float(info.duration)

        return result

    @staticmethod
    def _align_segments_with_beams(segments, beam_results):
        mapping: Dict[int, Tuple[int, int, int]] = {}
        if not segments or not beam_results:
            return mapping

        beam_idx = 0
        beam_tokens = beam_results[beam_idx].sequences_ids[0]
        cursor = 0

        for segment in segments:
            target_tokens = segment.tokens
            if not target_tokens:
                continue
            match_index = _find_subsequence(beam_tokens, target_tokens, cursor)
            while match_index is None:
                beam_idx += 1
                if beam_idx >= len(beam_results):
                    return mapping
                beam_tokens = beam_results[beam_idx].sequences_ids[0]
                cursor = 0
                match_index = _find_subsequence(beam_tokens, target_tokens, cursor)
            start = match_index
            end = start + len(target_tokens)
            mapping[segment.id] = (beam_idx, start, end)
            cursor = end
        return mapping

    @staticmethod
    def _build_word_items(segment) -> Tuple[List[_WordUnit], List[Dict[str, object]]]:
        word_units: List[_WordUnit] = []
        items: List[Dict[str, object]] = []

        for word in segment.words or []:
            parts = _split_word_parts(word.word)
            if not parts:
                continue
            for part_type, content in parts:
                if part_type == "punctuation":
                    items.append(
                        {
                            "type": "punctuation",
                            "alternatives": [
                                {
                                    "content": content,
                                    "confidence": _format_confidence(1.0),
                                }
                            ],
                        }
                    )
                else:
                    index = len(word_units)
                    word_units.append(
                        _WordUnit(
                            text=content,
                            start=float(word.start),
                            end=float(word.end),
                            confidence=float(getattr(word, "probability", 0.0) or 0.0),
                        )
                    )
                    items.append(
                        {
                            "type": "pronunciation",
                            "start_time": _format_time(word.start),
                            "end_time": _format_time(word.end),
                            "word_index": index,
                        }
                    )
        return word_units, items

    def _compute_alternatives(
        self,
        word_units: List[_WordUnit],
        beam_result,
        start_token: int,
        end_token: int,
        tokenizer,
    ) -> List[List[Dict[str, str]]]:
        probability_maps: List[OrderedDict[str, float]] = [
            OrderedDict({unit.text: unit.confidence}) for unit in word_units
        ]

        beam_probs = _softmax(beam_result.scores)
        for seq_index, sequence in enumerate(beam_result.sequences_ids):
            token_slice = sequence[start_token:end_token]
            if not token_slice:
                continue
            decoded = tokenizer.decode(token_slice).strip()
            candidate_words = [_normalize_word(token) for token in _tokenize_words(decoded)]
            if not candidate_words:
                continue
            for idx, unit in enumerate(word_units):
                if idx >= len(candidate_words):
                    break
                candidate = candidate_words[idx]
                if not candidate:
                    continue
                score = beam_probs[seq_index]
                existing = probability_maps[idx].get(candidate)
                if existing is None or score > existing:
                    probability_maps[idx][candidate] = score

        alternative_lists: List[List[Dict[str, str]]] = []
        for idx, unit in enumerate(word_units):
            entries = probability_maps[idx]
            base_conf = unit.confidence
            entries[unit.text] = base_conf
            sorted_entries = sorted(entries.items(), key=lambda pair: pair[1], reverse=True)

            ordered = [(unit.text, base_conf)] + [
                (word, conf)
                for word, conf in sorted_entries
                if word != unit.text and word
            ]
            alternative_lists.append(
                [
                    {"content": word, "confidence": _format_confidence(conf)}
                    for word, conf in ordered
                ]
            )

        return alternative_lists
