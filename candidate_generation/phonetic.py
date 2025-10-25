"""Phonetic neighbor candidate generator.

Generates candidate corrections by finding words with similar phonetic representations
using G2P (grapheme-to-phoneme) conversion and phoneme edit distance.
"""

from typing import List, Set, Optional
import logging
from dataclasses import dataclass

try:
    from g2p_en import G2p
except ImportError:
    G2p = None

try:
    from wordfreq import word_frequency, top_n_list
except ImportError:
    word_frequency = None
    top_n_list = None

try:
    from metaphone import doublemetaphone
except ImportError:
    doublemetaphone = None


logger = logging.getLogger(__name__)


@dataclass
class PhoneticCandidate:
    """A candidate word with its phonetic distance from the source."""
    word: str
    phoneme_distance: int
    frequency: float
    source: str  # 'g2p' or 'metaphone'


class PhoneticNeighborGenerator:
    """Generate phonetically similar word candidates."""
    
    def __init__(
        self,
        lexicon_size: Optional[int] = None,
        max_phoneme_distance: int = 2,
        min_frequency: float = 1e-8,
        language: str = 'en',
    ):
        """Initialize the phonetic neighbor generator.
        
        Args:
            lexicon_size: Number of most frequent words to consider (None = all words)
            max_phoneme_distance: Maximum phoneme edit distance for candidates
            min_frequency: Minimum word frequency threshold
            language: Language code for wordfreq
        """
        self.lexicon_size = lexicon_size
        self.max_phoneme_distance = max_phoneme_distance
        self.min_frequency = min_frequency
        self.language = language
        
        # Initialize G2P converter
        if G2p is None:
            logger.warning("g2p_en not installed. Phonetic generation via G2P disabled.")
            self.g2p = None
        else:
            self.g2p = G2p()
        
        # Check metaphone availability
        if doublemetaphone is None:
            logger.warning("metaphone not installed. Metaphone-based generation disabled.")
        
        # Build lexicon
        self._build_lexicon()
    
    def _build_lexicon(self):
        """Build the word frequency lexicon."""
        if word_frequency is None or top_n_list is None:
            logger.warning("wordfreq not installed. Using empty lexicon.")
            self.lexicon = {}
            self.phoneme_index = {}
            self.metaphone_index = {}
            return
        
        # Get words from wordfreq
        if self.lexicon_size is None:
            logger.info(f"Building lexicon with all available words...")
            # Use a large number to get all words
            words = top_n_list(self.language, 1000000)
        else:
            logger.info(f"Building lexicon with top {self.lexicon_size} words...")
            words = top_n_list(self.language, self.lexicon_size)
        
        # Build frequency map
        self.lexicon = {
            word: word_frequency(word, self.language)
            for word in words
            if word_frequency(word, self.language) >= self.min_frequency
        }
        
        # Build phoneme index (word -> phonemes)
        self.phoneme_index = {}
        if self.g2p:
            for word in self.lexicon:
                try:
                    phonemes = self.g2p(word)
                    # Filter out non-phoneme tokens (spaces, etc.)
                    phonemes = [p for p in phonemes if p.strip() and not p.isspace()]
                    self.phoneme_index[word.lower()] = tuple(phonemes)
                except Exception as e:
                    logger.debug(f"Failed to convert '{word}' to phonemes: {e}")
        
        # Build metaphone index
        self.metaphone_index = {}
        if doublemetaphone:
            for word in self.lexicon:
                try:
                    primary, secondary = doublemetaphone(word)
                    if primary:
                        self.metaphone_index[word.lower()] = (primary, secondary)
                except Exception as e:
                    logger.debug(f"Failed to get metaphone for '{word}': {e}")
        
        logger.info(
            f"Lexicon built: {len(self.lexicon)} words, "
            f"{len(self.phoneme_index)} with phonemes, "
            f"{len(self.metaphone_index)} with metaphones"
        )
    
    def _phoneme_edit_distance(self, phonemes1: tuple, phonemes2: tuple) -> int:
        """Calculate edit distance between two phoneme sequences.
        
        Uses dynamic programming (Levenshtein distance).
        """
        if not phonemes1:
            return len(phonemes2)
        if not phonemes2:
            return len(phonemes1)
        
        # Create DP table
        m, n = len(phonemes1), len(phonemes2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if phonemes1[i-1] == phonemes2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def generate_candidates(
        self,
        source_word: str,
        max_candidates: int = 20,
        include_variants: bool = True,
    ) -> List[PhoneticCandidate]:
        """Generate phonetically similar candidates for a source word.
        
        Args:
            source_word: The word to find phonetic neighbors for
            max_candidates: Maximum number of candidates to return
            include_variants: Include casing/morphological variants
        
        Returns:
            List of PhoneticCandidate objects sorted by distance then frequency
        """
        if not self.lexicon:
            return []
        
        candidates = []
        source_lower = source_word.lower().strip()
        
        # Get source phonemes
        source_phonemes = None
        if self.g2p and source_lower in self.phoneme_index:
            source_phonemes = self.phoneme_index[source_lower]
        elif self.g2p:
            try:
                phonemes = self.g2p(source_lower)
                source_phonemes = tuple(p for p in phonemes if p.strip() and not p.isspace())
            except Exception as e:
                logger.debug(f"Failed to convert source '{source_word}' to phonemes: {e}")
        
        # Get source metaphone
        source_metaphone = None
        if doublemetaphone:
            try:
                primary, secondary = doublemetaphone(source_lower)
                source_metaphone = (primary, secondary)
            except Exception:
                pass
        
        # Search via G2P phoneme distance
        if source_phonemes:
            for word, phonemes in self.phoneme_index.items():
                if word == source_lower:
                    continue
                
                distance = self._phoneme_edit_distance(source_phonemes, phonemes)
                if distance <= self.max_phoneme_distance:
                    freq = self.lexicon.get(word, 0.0)
                    candidates.append(PhoneticCandidate(
                        word=word,
                        phoneme_distance=distance,
                        frequency=freq,
                        source='g2p'
                    ))
        
        # Search via metaphone matching
        if source_metaphone and doublemetaphone:
            source_primary, source_secondary = source_metaphone
            for word, (primary, secondary) in self.metaphone_index.items():
                if word == source_lower:
                    continue
                
                # Check if metaphones match
                if (primary and primary == source_primary) or \
                   (secondary and secondary == source_primary) or \
                   (primary and primary == source_secondary):
                    # Only add if not already in candidates from G2P
                    if not any(c.word == word for c in candidates):
                        freq = self.lexicon.get(word, 0.0)
                        candidates.append(PhoneticCandidate(
                            word=word,
                            phoneme_distance=0,  # Metaphone match = close
                            frequency=freq,
                            source='metaphone'
                        ))
        
        # Add casing variants if requested
        if include_variants and candidates:
            variants = []
            for cand in candidates:
                # Add capitalized version
                if cand.word[0].islower() and source_word[0].isupper():
                    variants.append(PhoneticCandidate(
                        word=cand.word.capitalize(),
                        phoneme_distance=cand.phoneme_distance,
                        frequency=cand.frequency,
                        source=cand.source
                    ))
                # Add possessive if source has it
                if source_word.endswith("'s") and not cand.word.endswith("'s"):
                    variants.append(PhoneticCandidate(
                        word=cand.word + "'s",
                        phoneme_distance=cand.phoneme_distance,
                        frequency=cand.frequency * 0.5,  # Slightly penalize
                        source=cand.source
                    ))
            candidates.extend(variants)
        
        # Sort by distance (ascending) then frequency (descending)
        candidates.sort(key=lambda c: (c.phoneme_distance, -c.frequency))
        
        # Deduplicate and limit
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand.word.lower() not in seen:
                seen.add(cand.word.lower())
                unique_candidates.append(cand)
                if len(unique_candidates) >= max_candidates:
                    break
        
        return unique_candidates


def generate_phonetic_neighbors(
    word: str,
    max_candidates: int = 20,
    max_distance: int = 2,
    lexicon_size: Optional[int] = None,
) -> List[str]:
    """Convenience function to generate phonetic neighbors.
    
    Args:
        word: Source word
        max_candidates: Maximum number of candidates
        max_distance: Maximum phoneme edit distance
        lexicon_size: Size of frequency lexicon (None = all words)
    
    Returns:
        List of candidate words (strings only)
    """
    generator = PhoneticNeighborGenerator(
        lexicon_size=lexicon_size,
        max_phoneme_distance=max_distance,
    )
    candidates = generator.generate_candidates(word, max_candidates=max_candidates)
    return [c.word for c in candidates]

