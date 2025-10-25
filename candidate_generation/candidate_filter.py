"""Shared filters and deduplication for candidate generation.

Provides filtering logic to cap candidate slates and ensure quality:
- Deduplication while preserving order
- Frequency thresholds
- Length constraints
- Phonetic similarity checks
"""

import re
from typing import List, Set, Dict
from collections import OrderedDict
import logging

try:
    from metaphone import doublemetaphone
except ImportError:
    doublemetaphone = None

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None

from .util import is_capitalized, contains_digit, is_numeric

logger = logging.getLogger(__name__)


def dedup_keep_order(items: List[str]) -> List[str]:
    """Deduplicate list while preserving order.
    
    Args:
        items: List of strings
    
    Returns:
        Deduplicated list
    """
    return list(OrderedDict.fromkeys(items).keys())


def frequency_ok(
    w: str,
    min_zipf: float = 3.0,
    capitalized_ok: bool = True,
) -> bool:
    """Check if word meets frequency threshold.
    
    Args:
        w: Word to check
        min_zipf: Minimum Zipf frequency
        capitalized_ok: Allow capitalized words (proper nouns) regardless of frequency
    
    Returns:
        True if word passes frequency check
    """
    if zipf_frequency is None:
        # Without wordfreq, accept all
        return True
    
    # Always allow capitalized words (proper nouns)
    if capitalized_ok and is_capitalized(w):
        return True
    
    # Check frequency
    zipf = zipf_frequency(w.lower(), 'en')
    return zipf >= min_zipf


def metaphone_close(a: str, b: str) -> bool:
    """Check if two words have similar metaphone codes.
    
    Args:
        a: First word
        b: Second word
    
    Returns:
        True if metaphone codes match
    """
    if doublemetaphone is None:
        return False
    
    try:
        a1, a2 = doublemetaphone(a.lower())
        b1, b2 = doublemetaphone(b.lower())
        return (a1 and a1 == b1) or (a1 and a1 == b2) or (a2 and a2 == b1)
    except Exception:
        return False


def filter_candidates(
    source_token: str,
    cands: List[str],
    max_out: int = 12,
    min_zipf: float = 3.0,
    max_length_diff: int = 3,  # Relaxed from 2 to 3
) -> List[str]:
    """Filter and cap candidate list.
    
    Args:
        source_token: Original token
        cands: List of candidate strings
        max_out: Maximum candidates to return
        min_zipf: Minimum Zipf frequency
        max_length_diff: Maximum length difference from source
    
    Returns:
        Filtered and capped candidate list
    """
    src_len = len(source_token)
    src_lower = source_token.lower().strip('.,!?;:\'"')
    src_is_numeric = is_numeric(source_token)
    
    out = []
    seen = set()
    
    for c in cands:
        c_lower = c.lower().strip('.,!?;:\'"')
        
        # Skip duplicates (case-insensitive)
        if c_lower in seen:
            continue
        
        # NOTE: Do NOT skip if same as source - case changes are valid corrections!
        # E.g., "Hitai" (capitalized wrong) -> "hitachi" (correct lowercase proper noun)
        
        # Length constraint (use cleaned lengths)
        if abs(len(c_lower) - len(src_lower)) > max_length_diff:
            continue
        
        # Numeric constraint: reject candidates with digits unless source is numeric
        if not src_is_numeric and contains_digit(c):
            continue
        
        # Frequency check
        if not frequency_ok(c, min_zipf=min_zipf):
            continue
        
        out.append(c)
        seen.add(c_lower)
        
        if len(out) >= max_out:
            break
    
    return out


def merge_and_filter_candidates(
    source_token: str,
    aws_nbest: List[str],
    phonetic: List[str],
    normalization: List[str],
    word_boundary: List[str],
    gazetteer: List[str],
    grapheme: List[str],
    char_speller: List[str],
    max_total: int = 12,
) -> Dict[str, List[str]]:
    """Merge candidates from all sources, filter, and cap.
    
    Args:
        source_token: Original token
        aws_nbest: AWS N-best alternatives
        phonetic: Phonetic neighbors
        normalization: Normalization candidates
        word_boundary: Word-boundary edits
        gazetteer: Gazetteer candidates
        grapheme: Grapheme confusion candidates
        char_speller: Char-speller candidates
        max_total: Maximum total candidates
    
    Returns:
        Dict with 'raw' (per-source) and 'final' (filtered) candidates
    """
    raw = {
        'aws_nbest': aws_nbest,
        'phonetic': phonetic,
        'normalization': normalization,
        'word_boundary': word_boundary,
        'gazetteer': gazetteer,
        'grapheme': grapheme,
        'char_speller': char_speller,
    }
    
    # Merge in priority order (AWS first, then others)
    merged = []
    merged.extend(aws_nbest)
    merged.extend(phonetic)
    merged.extend(normalization)
    merged.extend(word_boundary)
    merged.extend(gazetteer)
    merged.extend(grapheme)
    merged.extend(char_speller)
    
    # Deduplicate
    merged = dedup_keep_order(merged)
    
    # Filter and cap
    filtered = filter_candidates(source_token, merged, max_out=max_total)
    
    return {
        'raw': raw,
        'final': filtered,
    }

