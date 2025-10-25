"""Weighted grapheme confusion candidate generator.

Generates candidates by applying learned character-level substitutions
based on common ASR confusion patterns (e.g., t↔ch, c↔k, s↔z).
"""

from typing import List, Dict, Tuple, Set, Optional
from heapq import heappush, heappop
import math
import logging

logger = logging.getLogger(__name__)


# Default confusion table (common ASR errors)
# Format: (source_char, target_char) -> probability
DEFAULT_CONFUSIONS = {
    # Consonant confusions
    ('t', 'ch'): 0.3, ('ch', 't'): 0.3,
    ('c', 'k'): 0.4, ('k', 'c'): 0.4,
    ('s', 'z'): 0.3, ('z', 's'): 0.3,
    ('f', 'ph'): 0.3, ('ph', 'f'): 0.3,
    ('f', 'v'): 0.2, ('v', 'f'): 0.2,
    ('b', 'p'): 0.2, ('p', 'b'): 0.2,
    ('d', 't'): 0.2, ('t', 'd'): 0.2,
    ('g', 'k'): 0.2, ('k', 'g'): 0.2,
    
    # Vowel confusions
    ('i', 'y'): 0.3, ('y', 'i'): 0.3,
    ('e', 'i'): 0.2, ('i', 'e'): 0.2,
    ('a', 'e'): 0.2, ('e', 'a'): 0.2,
    ('o', 'u'): 0.2, ('u', 'o'): 0.2,
    ('oo', 'u'): 0.3, ('u', 'oo'): 0.3,
    
    # Common endings
    ('er', 'or'): 0.3, ('or', 'er'): 0.3,
    ('le', 'el'): 0.2, ('el', 'le'): 0.2,
    ('tion', 'sion'): 0.2, ('sion', 'tion'): 0.2,
}


class WeightedLevenshtein:
    """Weighted Levenshtein distance with learned character confusions."""
    
    def __init__(self, sub_probs: Optional[Dict[Tuple[str, str], float]] = None):
        """Initialize with substitution probabilities.
        
        Args:
            sub_probs: Dictionary mapping (source, target) -> probability
        """
        if sub_probs is None:
            sub_probs = DEFAULT_CONFUSIONS
        
        self.sub_cost = {}
        # cost = -log P(target|source); fall back to default_cost if unseen
        for (a, b), p in sub_probs.items():
            self.sub_cost[(a, b)] = -math.log(max(p, 1e-5))
        
        self.default_cost = 2.0  # Cost for unseen substitutions
        self.insert_cost = 1.0
        self.delete_cost = 1.0
    
    def sub_cost_fn(self, a: str, b: str) -> float:
        """Get substitution cost for replacing a with b."""
        # Check exact match
        if (a, b) in self.sub_cost:
            return self.sub_cost[(a, b)]
        
        # Check if a or b is multi-char and matches
        for (src, tgt), cost in self.sub_cost.items():
            if len(src) > 1 or len(tgt) > 1:
                # Multi-char substitution (e.g., 'ph' -> 'f')
                if a == src and b == tgt:
                    return cost
        
        return self.default_cost
    
    def distance(self, s1: str, s2: str) -> float:
        """Calculate weighted edit distance between two strings."""
        m, n = len(s1), len(s2)
        
        # Create DP table
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        
        # Initialize base cases
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.delete_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.insert_cost
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Try substitution
                    sub_cost = self.sub_cost_fn(s1[i-1], s2[j-1])
                    dp[i][j] = min(
                        dp[i-1][j] + self.delete_cost,     # deletion
                        dp[i][j-1] + self.insert_cost,     # insertion
                        dp[i-1][j-1] + sub_cost,           # substitution
                    )
                    
                    # Try multi-char substitutions (e.g., 'ph' -> 'f')
                    for (src, tgt), cost in self.sub_cost.items():
                        if len(src) > 1 and i >= len(src):
                            if s1[i-len(src):i] == src and j > 0 and s2[j-1] == tgt:
                                dp[i][j] = min(dp[i][j], dp[i-len(src)][j-1] + cost)
                        if len(tgt) > 1 and j >= len(tgt):
                            if s2[j-len(tgt):j] == tgt and i > 0 and s1[i-1] == src:
                                dp[i][j] = min(dp[i][j], dp[i-1][j-len(tgt)] + cost)
        
        return dp[m][n]


def generate_variants(
    token: str,
    wl: WeightedLevenshtein,
    max_edits_cost: float = 3.0,
    max_out: int = 8,
) -> List[str]:
    """Generate spelling variants using weighted edit operations.
    
    Uses A* search to explore edit operations bounded by cumulative cost.
    
    Args:
        token: Source token
        wl: WeightedLevenshtein instance with confusion costs
        max_edits_cost: Maximum cumulative edit cost
        max_out: Maximum variants to generate
    
    Returns:
        List of variant strings
    """
    if not token:
        return []
    
    # Priority queue: (cost, string)
    Q = []
    seen = set()
    out = set()
    
    heappush(Q, (0.0, token))
    
    while Q and len(out) < max_out:
        cost, s = heappop(Q)
        
        if s in seen:
            continue
        seen.add(s)
        
        # Add to output if different from source
        if s != token:
            out.add(s)
        
        # Stop expanding if cost too high
        if cost >= max_edits_cost:
            continue
        
        # Try substitutions at each position
        for i in range(len(s)):
            ch = s[i]
            
            # Single-char substitutions
            for (a, b), sub_cost in wl.sub_cost.items():
                if len(a) == 1 and len(b) == 1 and a == ch:
                    ns = s[:i] + b + s[i+1:]
                    nc = cost + sub_cost
                    if nc <= max_edits_cost and ns not in seen:
                        heappush(Q, (nc, ns))
            
            # Multi-char substitutions (e.g., 'ph' -> 'f')
            for (a, b), sub_cost in wl.sub_cost.items():
                if len(a) > 1 and i + len(a) <= len(s):
                    if s[i:i+len(a)] == a:
                        ns = s[:i] + b + s[i+len(a):]
                        nc = cost + sub_cost
                        if nc <= max_edits_cost and ns not in seen:
                            heappush(Q, (nc, ns))
    
    return list(out)


def generate_grapheme_candidates(
    token: str,
    max_edits_cost: float = 3.0,
    max_out: int = 8,
) -> List[str]:
    """Convenience function to generate grapheme confusion candidates.
    
    Args:
        token: Source token
        max_edits_cost: Maximum edit cost
        max_out: Maximum candidates
    
    Returns:
        List of candidate strings
    """
    wl = WeightedLevenshtein()
    return generate_variants(token, wl, max_edits_cost, max_out)

