"""Word-boundary candidate generator.

Generates candidate corrections by splitting or merging adjacent tokens:
- Split: "microsoft" → "micro soft"
- Merge: "micro soft" → "microsoft"
- Merge with context: "Lager Cove" → "Bootleggers Cove" (when considering adjacent tokens)
"""

from typing import List, Optional, Tuple
import logging

try:
    from wordfreq import word_frequency
except ImportError:
    word_frequency = None


logger = logging.getLogger(__name__)


def split_word(word: str, min_length: int = 2) -> List[Tuple[str, str]]:
    """Generate split candidates for a word.
    
    Args:
        word: Word to split
        min_length: Minimum length for each part
    
    Returns:
        List of (left, right) split tuples
    """
    if len(word) < min_length * 2:
        return []
    
    candidates = []
    word_lower = word.lower().strip('.,!?;:')
    
    # Try all possible split points
    for i in range(min_length, len(word_lower) - min_length + 1):
        left = word_lower[:i]
        right = word_lower[i:]
        
        # Check if both parts look like words (basic heuristic)
        if len(left) >= min_length and len(right) >= min_length:
            # If wordfreq is available, check if both parts are real words
            if word_frequency is not None:
                left_freq = word_frequency(left, 'en')
                right_freq = word_frequency(right, 'en')
                # Only keep if both parts have some frequency
                if left_freq > 1e-8 and right_freq > 1e-8:
                    candidates.append((left, right))
            else:
                # Without wordfreq, accept all splits
                candidates.append((left, right))
    
    return candidates


def merge_words(word1: str, word2: str) -> List[str]:
    """Generate merge candidates for two adjacent words.
    
    Args:
        word1: First word
        word2: Second word
    
    Returns:
        List of merged word candidates
    """
    candidates = []
    
    # Clean words
    word1_clean = word1.lower().strip('.,!?;:')
    word2_clean = word2.lower().strip('.,!?;:')
    
    # Simple concatenation
    merged = word1_clean + word2_clean
    
    # Check if merged word exists
    if word_frequency is not None:
        freq = word_frequency(merged, 'en')
        if freq > 1e-8:
            candidates.append(merged)
            # Also add capitalized version
            if word1[0].isupper():
                candidates.append(merged.capitalize())
    else:
        # Without wordfreq, return the merge
        candidates.append(merged)
        if word1[0].isupper():
            candidates.append(merged.capitalize())
    
    return candidates


def generate_word_boundary_candidates(
    word: str,
    prev_word: Optional[str] = None,
    next_word: Optional[str] = None,
) -> List[str]:
    """Generate word-boundary candidates for a word.
    
    Args:
        word: Current word
        prev_word: Previous word (for merge-left)
        next_word: Next word (for merge-right)
    
    Returns:
        List of candidate strings
    """
    candidates = []
    
    # Generate split candidates
    splits = split_word(word, min_length=2)
    for left, right in splits:
        # Return as space-separated (caller will handle multi-token replacements)
        candidates.append(f"{left} {right}")
    
    # Generate merge-left candidates
    if prev_word:
        merges = merge_words(prev_word, word)
        candidates.extend(merges)
    
    # Generate merge-right candidates
    if next_word:
        merges = merge_words(word, next_word)
        candidates.extend(merges)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for cand in candidates:
        if cand.lower() not in seen and cand.lower() != word.lower():
            seen.add(cand.lower())
            unique_candidates.append(cand)
    
    return unique_candidates


def generate_multi_word_boundary_candidates(
    words: List[str],
    target_index: int,
) -> List[str]:
    """Generate word-boundary candidates considering multiple adjacent words.
    
    This is useful for cases like "Lager Cove" where we want to suggest
    "Bootleggers" as a replacement for both words.
    
    Args:
        words: List of words in sequence
        target_index: Index of the target word
    
    Returns:
        List of candidate strings (may be multi-token)
    """
    if target_index < 0 or target_index >= len(words):
        return []
    
    candidates = []
    target_word = words[target_index]
    
    # Get adjacent words
    prev_word = words[target_index - 1] if target_index > 0 else None
    next_word = words[target_index + 1] if target_index < len(words) - 1 else None
    
    # Generate single-word boundary candidates
    candidates.extend(generate_word_boundary_candidates(target_word, prev_word, next_word))
    
    # Try merging with both neighbors (for cases like "a b c" -> "abc")
    if prev_word and next_word:
        # Try three-way merge
        merged = prev_word.lower().strip('.,!?;:') + \
                 target_word.lower().strip('.,!?;:') + \
                 next_word.lower().strip('.,!?;:')
        
        if word_frequency is not None:
            freq = word_frequency(merged, 'en')
            if freq > 1e-8:
                candidates.append(merged)
                if prev_word[0].isupper():
                    candidates.append(merged.capitalize())
    
    return candidates

