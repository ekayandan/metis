"""Utilities for acoustically expanding alternative tokens."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List, Sequence

import warnings

import numpy as np
import pronouncing
from g2p_en import G2p

warnings.filterwarnings("ignore", category=RuntimeWarning, module="g2p_en")

# Regex matches letters and apostrophes so we keep contractions like "I'm".
_NORMALIZE_PATTERN = re.compile(r"[^a-z']+")
_PHONETIC_MAX_DISTANCE = 1
_PHONETIC_MAX_NEIGHBORS = 5
_MAX_HOMOPHONE_TOTAL = 6
_MAX_PHONETIC_TOTAL = 6
_MAX_TOTAL_OPTIONS = 12
_CANDIDATE_PATTERN = re.compile(r"^[a-z][a-z']*$")

_G2P = G2p()
pronouncing.init_cmu()

# Build a lightweight index of pronouncing vocabulary to narrow search space.
_PRONOUNCING_VOCAB: tuple[str, ...] = tuple(sorted(pronouncing.lookup.keys()))
_VOCAB_BY_INITIAL: dict[str, tuple[str, ...]] = {}
for _word in _PRONOUNCING_VOCAB:
    if not _word:
        continue
    initial = _word[0].lower()
    _VOCAB_BY_INITIAL.setdefault(initial, []).append(_word)
_VOCAB_BY_INITIAL = {k: tuple(v) for k, v in _VOCAB_BY_INITIAL.items()}


def _normalize(word: str) -> str:
    """Lowercase and strip non-speech characters for dictionary lookup."""
    return _NORMALIZE_PATTERN.sub("", word.lower())


def _is_valid_candidate(word: str) -> bool:
    if not word:
        return False
    if not re.fullmatch(r"[A-Za-z']+", word):
        return False
    normalized = _normalize(word)
    if not normalized:
        return False
    return bool(_CANDIDATE_PATTERN.fullmatch(normalized))


@lru_cache(maxsize=65536)
def _g2p_phonemes(word: str) -> tuple[str, ...]:
    return tuple(p for p in _G2P(word) if p != " ")


@lru_cache(maxsize=65536)
def _phonemes_for_word(word: str) -> tuple[str, ...]:
    normalized = _normalize(word)
    if not normalized:
        return ()
    phones = pronouncing.phones_for_word(normalized)
    if phones:
        return tuple(phones[0].split())
    return _g2p_phonemes(normalized)


def get_homophones(word: str) -> List[str]:
    """Return homophones for ``word`` using the CMU pronunciation dictionary."""
    normalized = _normalize(word)
    if not normalized:
        return []
    phones = pronouncing.phones_for_word(normalized)
    if not phones:
        return []
    matches = pronouncing.search(f"^{phones[0]}$")
    return matches


def phoneme_edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=float)
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(a)][len(b)]


def _candidate_vocab(word: str) -> Sequence[str]:
    if not word:
        return _PRONOUNCING_VOCAB
    return _VOCAB_BY_INITIAL.get(word[0].lower(), _PRONOUNCING_VOCAB)


@lru_cache(maxsize=65536)
def _letter_edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(a)][len(b)]


def _within_letter_distance(base: str, candidate: str) -> bool:
    norm_base = _normalize(base)
    norm_candidate = _normalize(candidate)
    if not norm_base or not norm_candidate:
        return True
    base_len = len(norm_base)
    max_len = max(base_len, len(norm_candidate))
    limit = 1 if max_len <= 5 else 2
    if base_len <= 4:
        if norm_base[0] != norm_candidate[0] or norm_base[-1] != norm_candidate[-1]:
            return False
    return _letter_edit_distance(norm_base, norm_candidate) <= limit


def get_phonetic_neighbors(word: str, max_distance: int = _PHONETIC_MAX_DISTANCE) -> List[str]:
    """Return vocabulary items whose phonemes are within ``max_distance`` edits."""
    base_phonemes = _phonemes_for_word(word)
    if not base_phonemes:
        return []

    neighbors: List[str] = []
    for candidate in _candidate_vocab(word):
        if not _is_valid_candidate(candidate):
            continue
        if candidate.lower() == word.lower():
            continue
        candidate_phonemes = _phonemes_for_word(candidate)
        if not candidate_phonemes:
            continue
        if not _within_letter_distance(word, candidate):
            continue
        distance = phoneme_edit_distance(base_phonemes, candidate_phonemes)
        if distance <= max_distance:
            neighbors.append(candidate)
            if len(neighbors) >= _PHONETIC_MAX_NEIGHBORS:
                break
    return neighbors


def expand_with_homophones(alternatives: Sequence[str]) -> List[str]:
    """Augment ``alternatives`` with homophones and nearby phonetic variants."""
    seen: set[str] = set()
    expanded: List[str] = []

    def add_options(options: Iterable[str], *, allow_all: bool = False) -> None:
        for option in options:
            if not option:
                continue
            if not allow_all and not _is_valid_candidate(option):
                continue
            key = option.lower()
            if key in seen:
                continue
            seen.add(key)
            expanded.append(option)

    add_options(alternatives, allow_all=True)
    primary_option = alternatives[0] if alternatives else ""
    neighbor_seeds: List[str] = []

    if primary_option and _is_valid_candidate(primary_option):
        neighbor_seeds.append(primary_option)

    homophone_added = 0
    for option in list(expanded):
        if homophone_added >= _MAX_HOMOPHONE_TOTAL:
            break
        for homophone in get_homophones(option):
            if homophone_added >= _MAX_HOMOPHONE_TOTAL:
                break
            prev_len = len(expanded)
            add_options([homophone])
            if len(expanded) > prev_len:
                homophone_added += 1
                if option == primary_option and _is_valid_candidate(homophone):
                    neighbor_seeds.append(homophone)

    if neighbor_seeds:
        neighbor_seeds = list(dict.fromkeys(neighbor_seeds))

    phonetic_added = 0
    for option in neighbor_seeds:
        if phonetic_added >= _MAX_PHONETIC_TOTAL:
            break
        if len(_normalize(option)) <= 4:
            continue
        neighbors = get_phonetic_neighbors(option)
        for neighbor in neighbors:
            if phonetic_added >= _MAX_PHONETIC_TOTAL:
                break
            prev_len = len(expanded)
            add_options([neighbor])
            if len(expanded) > prev_len:
                phonetic_added += 1

    if len(expanded) > _MAX_TOTAL_OPTIONS:
        expanded = expanded[:_MAX_TOTAL_OPTIONS]
    return expanded


def filter_by_phonetic_distance(
    base_word: str,
    candidates: Sequence[str],
    *,
    max_distance: int = _PHONETIC_MAX_DISTANCE,
) -> List[str]:
    """Keep only candidates whose phoneme distance from ``base_word`` is within ``max_distance``.

    If phonemes cannot be derived for the base word or a candidate, the candidate is retained to
    avoid false negatives. Always preserves the original ordering of the input sequence.
    """

    if not candidates:
        return []

    base_phonemes = _phonemes_for_word(base_word)
    if not base_phonemes:
        return list(candidates)

    filtered: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.lower() == base_word.lower():
            filtered.append(candidate)
            continue
        cand_phonemes = _phonemes_for_word(candidate)
        if not cand_phonemes:
            filtered.append(candidate)
            continue
        distance = phoneme_edit_distance(base_phonemes, cand_phonemes)
        if distance <= max_distance:
            filtered.append(candidate)

    if not filtered:
        # Ensure the best ASR guess is preserved even if other options were filtered out.
        filtered.append(candidates[0])

    return filtered
