"""Candidate generation modules for ASR correction.

This package contains various candidate generators that augment ASR N-best
alternatives with phonetic neighbors, normalizations, word-boundary edits, etc.
"""

from .phonetic import PhoneticNeighborGenerator, generate_phonetic_neighbors
from .normalization import generate_normalization_candidates
from .word_boundary import (
    generate_word_boundary_candidates,
    generate_multi_word_boundary_candidates,
    split_word,
    merge_words,
)

__all__ = [
    'PhoneticNeighborGenerator',
    'generate_phonetic_neighbors',
    'generate_normalization_candidates',
    'generate_word_boundary_candidates',
    'generate_multi_word_boundary_candidates',
    'split_word',
    'merge_words',
]

