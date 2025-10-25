#!/usr/bin/env python3
"""Compute word error rate between reference and hypothesis transcripts."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from jiwer import wer as jiwer_wer
from torchmetrics.text import WordErrorRate


def normalize_and_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def load_transcripts(path: str, text_column: int) -> Dict[str, List[str]]:
    transcripts: Dict[str, List[str]] = {}
    missing_columns = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            if text_column >= len(row):
                missing_columns += 1
                continue
            transcripts[key] = normalize_and_tokenize(row[text_column])
    if missing_columns:
        print(
            f"Warning: {missing_columns} row(s) missing column {text_column + 1} in {path}",
            file=sys.stderr,
        )
    return transcripts


def load_hypotheses_with_candidates(path: str) -> Dict[str, List[List[str]]]:
    candidates: Dict[str, List[List[str]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            variant_tokens = [normalize_and_tokenize(value) for value in row[1:]]
            candidates[key] = variant_tokens
    return candidates


AlignmentStep = Tuple[str, Optional[int], Optional[int], Optional[str], Optional[str]]


def alignment_details(
    reference: List[str], hypothesis: List[str]
) -> Tuple[int, int, int, List[AlignmentStep]]:
    ref_len, hyp_len = len(reference), len(hypothesis)
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    backtrack = [["" for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        backtrack[i][0] = "del"
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        backtrack[0][j] = "ins"
    backtrack[0][0] = "done"

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                backtrack[i][j] = "eq"
            else:
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                substitution = dp[i - 1][j - 1] + 1
                best = min(deletion, insertion, substitution)
                dp[i][j] = best
                if best == substitution:
                    backtrack[i][j] = "sub"
                elif best == deletion:
                    backtrack[i][j] = "del"
                else:
                    backtrack[i][j] = "ins"

    substitutions = deletions = insertions = 0
    i, j = ref_len, hyp_len
    path: List[AlignmentStep] = []
    while i > 0 or j > 0:
        action = backtrack[i][j]
        if action == "eq":
            path.append((action, i - 1, j - 1, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif action == "sub":
            substitutions += 1
            path.append((action, i - 1, j - 1, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif action == "del":
            deletions += 1
            path.append((action, i - 1, None, reference[i - 1], None))
            i -= 1
        elif action == "ins":
            insertions += 1
            path.append((action, None, j - 1, None, hypothesis[j - 1]))
            j -= 1
        else:
            break
    path.reverse()

    return substitutions, deletions, insertions, path


def build_alternative_sets(
    reference_tokens: Sequence[str],
    candidate_sequences: Sequence[Sequence[str]],
) -> List[Set[str]]:
    alternative_sets: List[Set[str]] = [set() for _ in reference_tokens]
    if not reference_tokens:
        return alternative_sets

    for sequence in candidate_sequences:
        if not sequence:
            continue
        _, _, _, path = alignment_details(list(reference_tokens), list(sequence))
        for action, ref_index, _, _, hyp_word in path:
            if action == "ins" or ref_index is None:
                continue
            if hyp_word is not None:
                alternative_sets[ref_index].add(hyp_word)
    return alternative_sets


def alt_hits_for_errors(
    alignment_path: Sequence[AlignmentStep], alternative_sets: Sequence[Set[str]]
) -> Tuple[int, int]:
    false_hits = 0
    false_total = 0
    for action, ref_index, _, ref_word, _ in alignment_path:
        if action != "sub" or ref_index is None or not ref_word:
            continue
        false_total += 1
        if ref_word in alternative_sets[ref_index]:
            false_hits += 1
    return false_hits, false_total


def compute_wer(
    reference_map: Dict[str, List[str]],
    hypothesis_candidates: Dict[str, List[List[str]]],
    hyp_column_index: int,
) -> Tuple[int, int, int, int, float, int, int, Set[str]]:
    substitutions = deletions = insertions = 0
    reference_words = 0
    matched_keys: Set[str] = set()
    total_alt_hits = 0
    total_alt_words = 0
    wer_metric = WordErrorRate()

    for key, ref_tokens in reference_map.items():
        variants = hypothesis_candidates.get(key)
        if not variants or hyp_column_index >= len(variants):
            continue
        hyp_tokens = variants[hyp_column_index]
        s, d, i, alignment_path = alignment_details(ref_tokens, hyp_tokens)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += len(ref_tokens)
        matched_keys.add(key)

        reference_text = " ".join(ref_tokens)
        hypothesis_text = " ".join(hyp_tokens)
        wer_metric.update([hypothesis_text], [reference_text])

        alternative_sets = build_alternative_sets(ref_tokens, variants)
        alt_hits, alt_total = alt_hits_for_errors(alignment_path, alternative_sets)
        total_alt_hits += alt_hits
        total_alt_words += alt_total

    wer_value = float(wer_metric.compute()) if matched_keys else 0.0

    return (
        substitutions,
        deletions,
        insertions,
        reference_words,
        wer_value,
        total_alt_hits,
        total_alt_words,
        matched_keys,
    )


def main(args: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute word error rate between a reference transcript TSV and an AWS Transcribe TSV."
        )
    )
    parser.add_argument("reference", help="TSV file with reference transcripts (path, text)")
    parser.add_argument("hypothesis", help="TSV file with hypothesis transcripts (path, text(s))")
    parser.add_argument(
        "--hyp-column",
        type=int,
        default=1,
        help=(
            "1-based index of the hypothesis text column in the hypothesis file; "
            "defaults to 1 (first transcript after the path)."
        ),
    )

    parsed = parser.parse_args(list(args) if args is not None else None)

    hyp_column_index = parsed.hyp_column - 1
    if hyp_column_index < 0:
        parser.error("--hyp-column must be >= 1")

    reference_map = load_transcripts(parsed.reference, text_column=1)
    hypothesis_candidates = load_hypotheses_with_candidates(parsed.hypothesis)

    (
        substitutions,
        deletions,
        insertions,
        reference_words,
        wer_value,
        total_alt_hits,
        total_alt_words,
        matched_keys,
    ) = compute_wer(reference_map, hypothesis_candidates, hyp_column_index)

    missing = set(reference_map) - matched_keys

    print(f"Utterances compared: {len(reference_map) - len(missing)}")
    print(f"Missing hypotheses: {len(missing)}")
    print(f"Reference words: {reference_words}")
    if reference_words:
        print(f"WER: {wer_value:.2%}")
        print(f"Substitutions: {substitutions}")
        print(f"Deletions: {deletions}")
        print(f"Insertions: {insertions}")
        if total_alt_words:
            alt_hit_rate = total_alt_hits / total_alt_words
            print(f"Alt-hit rate: {alt_hit_rate:.2%}")
            print(f"Alt-hit successes: {total_alt_hits}")
            print(f"Alt-hit opportunities: {total_alt_words}")
        else:
            print("Alt-hit rate: n/a")
    else:
        print("No reference words to compare.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
