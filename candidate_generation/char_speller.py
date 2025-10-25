"""Character-level speller for OOV candidates.

Generates plausible spelling variants for out-of-vocabulary words using
character-level edit operations guided by frequency heuristics.

Note: This is a simplified implementation using heuristic scoring rather than
a full neural character-LM. A production system would train a small LSTM/GRU
on a large word corpus.
"""

from typing import List, Set, Callable, Optional
from heapq import heappush, heappop
import logging

try:
    from wordfreq import word_frequency
except ImportError:
    word_frequency = None

logger = logging.getLogger(__name__)


# Common character frequencies in English (for heuristic scoring)
CHAR_FREQ = {
    'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
    'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043,
    'l': 0.040, 'c': 0.028, 'u': 0.028, 'm': 0.024, 'w': 0.024,
    'f': 0.022, 'g': 0.020, 'y': 0.020, 'p': 0.019, 'b': 0.015,
    'v': 0.010, 'k': 0.008, 'j': 0.002, 'x': 0.002, 'q': 0.001,
    'z': 0.001,
}


def char_score(ch: str) -> float:
    """Get heuristic score for a character (higher = more common)."""
    return CHAR_FREQ.get(ch.lower(), 0.001)


def word_score(word: str) -> float:
    """Heuristic score for a word based on character frequencies.
    
    This is a simple proxy for a character-LM probability.
    """
    if not word:
        return -100.0
    
    # Average character frequency
    score = sum(char_score(ch) for ch in word) / len(word)
    
    # Bonus for real words in wordfreq
    if word_frequency is not None:
        freq = word_frequency(word.lower(), 'en')
        if freq > 1e-7:
            score += 10.0  # Strong bonus for known words
        elif freq > 1e-8:
            score += 5.0   # Moderate bonus
    
    return score


def beam_speller(
    seed: str,
    beam: int = 8,
    max_steps: int = 2,
    allow: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """Generate spelling variants using beam search.
    
    Args:
        seed: Source word
        beam: Beam size
        max_steps: Maximum edit operations
        allow: Optional filter function (returns True if candidate is allowed)
    
    Returns:
        List of candidate strings
    """
    if not seed:
        return []
    
    if allow is None:
        allow = lambda s: True
    
    # Priority queue: (negative_score, num_edits, string)
    hypos = [(0.0, 0, seed)]
    seen = {seed}
    
    for step in range(max_steps):
        new_hypos = []
        
        for score, edits, s in hypos:
            # Try substitutions at each position
            for i in range(len(s)):
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    if ch == s[i].lower():
                        continue
                    
                    # Preserve case
                    if s[i].isupper():
                        ch = ch.upper()
                    
                    cand = s[:i] + ch + s[i+1:]
                    
                    if cand not in seen and allow(cand):
                        seen.add(cand)
                        cand_score = word_score(cand)
                        new_hypos.append((-cand_score, edits + 1, cand))
            
            # Try insertions
            for i in range(len(s) + 1):
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    # Determine case based on position
                    if i == 0 and s and s[0].isupper():
                        ch = ch.upper()
                    
                    cand = s[:i] + ch + s[i:]
                    
                    if cand not in seen and allow(cand):
                        seen.add(cand)
                        cand_score = word_score(cand)
                        new_hypos.append((-cand_score, edits + 1, cand))
            
            # Try deletions
            for i in range(len(s)):
                cand = s[:i] + s[i+1:]
                
                if cand and cand not in seen and allow(cand):
                    seen.add(cand)
                    cand_score = word_score(cand)
                    new_hypos.append((-cand_score, edits + 1, cand))
        
        # Keep top beam candidates
        hypos = sorted(new_hypos, key=lambda x: x[0])[:beam]
        
        if not hypos:
            break
    
    # Return unique strings (excluding seed)
    results = [s for _, _, s in hypos if s != seed]
    
    # Sort by score (best first)
    results.sort(key=lambda s: -word_score(s))
    
    return results


def generate_char_speller_candidates(
    token: str,
    beam: int = 8,
    max_steps: int = 2,
    max_out: int = 8,
) -> List[str]:
    """Convenience function to generate char-speller candidates.
    
    Args:
        token: Source token
        beam: Beam size
        max_steps: Maximum edit steps
        max_out: Maximum candidates to return
    
    Returns:
        List of candidate strings
    """
    # Define allow function: length within ±2, same first letter if capitalized
    def allow(s: str) -> bool:
        if not s:
            return False
        if abs(len(s) - len(token)) > 2:
            return False
        # If source is capitalized, require same first letter
        if token and token[0].isupper():
            if not s or s[0].lower() != token[0].lower():
                return False
        # Only letters and basic punctuation
        if not all(c.isalpha() or c in "'-." for c in s):
            return False
        return True
    
    candidates = beam_speller(token, beam=beam, max_steps=max_steps, allow=allow)
    return candidates[:max_out]

