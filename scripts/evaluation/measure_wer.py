#!/usr/bin/env python3
"""Compute word error rate between reference and hypothesis transcripts."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from typing import Dict, Iterable, List, Tuple


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


def alignment_stats(reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int]:
    ref_len, hyp_len = len(reference), len(hypothesis)
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    backtrack = [["" for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        backtrack[i][0] = "del"
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        backtrack[0][j] = "ins"

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
    while i > 0 or j > 0:
        action = backtrack[i][j]
        if action == "eq":
            i -= 1
            j -= 1
        elif action == "sub":
            substitutions += 1
            i -= 1
            j -= 1
        elif action == "del":
            deletions += 1
            i -= 1
        elif action == "ins":
            insertions += 1
            j -= 1
        else:
            break

    return substitutions, deletions, insertions


def compute_wer(
    reference_map: Dict[str, List[str]],
    hypothesis_map: Dict[str, List[str]],
) -> Tuple[int, int, int, int]:
    substitutions = deletions = insertions = 0
    reference_words = 0
    matched_items = 0

    for key, ref_tokens in reference_map.items():
        hyp_tokens = hypothesis_map.get(key)
        if hyp_tokens is None:
            continue
        s, d, i = alignment_stats(ref_tokens, hyp_tokens)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += len(ref_tokens)
        matched_items += 1

    return substitutions, deletions, insertions, reference_words


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
    hypothesis_map = load_transcripts(parsed.hypothesis, text_column=hyp_column_index + 1)

    substitutions, deletions, insertions, reference_words = compute_wer(
        reference_map, hypothesis_map
    )

    missing = set(reference_map) - set(hypothesis_map)

    print(f"Utterances compared: {len(reference_map) - len(missing)}")
    print(f"Missing hypotheses: {len(missing)}")
    print(f"Reference words: {reference_words}")
    if reference_words:
        wer = (substitutions + deletions + insertions) / reference_words
        print(f"WER: {wer:.2%}")
        print(f"Substitutions: {substitutions}")
        print(f"Deletions: {deletions}")
        print(f"Insertions: {insertions}")
    else:
        print("No reference words to compare.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
