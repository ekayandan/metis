"""End-to-end span candidate generation.

Integrates all candidate sources (AWS N-best, phonetic, normalization,
word-boundary, gazetteer, grapheme, char-speller) and applies filters.
"""

from typing import List, Dict, Optional
import logging

from .phonetic import PhoneticNeighborGenerator
from .normalization import generate_normalization_candidates
from .word_boundary import generate_word_boundary_candidates
from .gazetteer import Gazetteer, gazetteer_candidates
from .grapheme_confusions import generate_grapheme_candidates
from .char_speller import generate_char_speller_candidates
from .candidate_filter import merge_and_filter_candidates

logger = logging.getLogger(__name__)


class MultiSourceCandidateGenerator:
    """Multi-source candidate generator for ASR correction."""
    
    def __init__(
        self,
        phonetic_lexicon_size: Optional[int] = None,
        gazetteer_min_zipf: float = 3.5,
    ):
        """Initialize all candidate generators.
        
        Args:
            phonetic_lexicon_size: Size of phonetic lexicon (None = all words)
            gazetteer_min_zipf: Minimum Zipf for gazetteer entries
        """
        logger.info("Initializing multi-source candidate generator...")
        
        # Initialize phonetic generator
        self.phonetic_gen = PhoneticNeighborGenerator(
            lexicon_size=phonetic_lexicon_size,
            max_phoneme_distance=2,
        )
        
        # Initialize gazetteer
        self.gazetteer = Gazetteer(min_zipf=gazetteer_min_zipf)
        
        logger.info(
            f"Initialized: "
            f"phonetic lexicon={len(self.phonetic_gen.lexicon)}, "
            f"gazetteer={len(self.gazetteer.entries)}"
        )
    
    def generate_candidates(
        self,
        span_text: str,
        aws_nbest: Optional[List[str]] = None,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None,
        max_total: int = 12,
    ) -> Dict[str, any]:
        """Generate candidates from all sources for a span.
        
        Args:
            span_text: Source text
            aws_nbest: AWS N-best alternatives (if available)
            prev_word: Previous word for word-boundary edits
            next_word: Next word for word-boundary edits
            max_total: Maximum total candidates after filtering
        
        Returns:
            Dict with 'raw' (per-source candidates) and 'final' (filtered slate)
        """
        if not span_text:
            return {'raw': {}, 'final': []}
        
        # Generate from each source
        aws_candidates = aws_nbest if aws_nbest else []
        
        phonetic_candidates = [
            c.word for c in self.phonetic_gen.generate_candidates(span_text, max_candidates=20)
        ]
        
        norm_candidates = generate_normalization_candidates(span_text)
        
        wb_candidates = generate_word_boundary_candidates(span_text, prev_word, next_word)
        
        gaz_candidates = gazetteer_candidates(span_text, self.gazetteer, max_out=5)
        
        grapheme_candidates = generate_grapheme_candidates(span_text, max_edits_cost=3.0, max_out=8)
        
        speller_candidates = generate_char_speller_candidates(span_text, beam=8, max_steps=2, max_out=8)
        
        # Merge and filter
        result = merge_and_filter_candidates(
            span_text,
            aws_candidates,
            phonetic_candidates,
            norm_candidates,
            wb_candidates,
            gaz_candidates,
            grapheme_candidates,
            speller_candidates,
            max_total=max_total,
        )
        
        return result


def generate_span_candidates(
    span_text: str,
    generator: MultiSourceCandidateGenerator,
    aws_nbest: Optional[List[str]] = None,
    prev_word: Optional[str] = None,
    next_word: Optional[str] = None,
    max_total: int = 12,
) -> Dict[str, any]:
    """Convenience function to generate candidates for a span.
    
    Args:
        span_text: Source text
        generator: MultiSourceCandidateGenerator instance
        aws_nbest: AWS N-best alternatives
        prev_word: Previous word
        next_word: Next word
        max_total: Maximum candidates
    
    Returns:
        Dict with 'raw' and 'final' candidates
    """
    return generator.generate_candidates(
        span_text,
        aws_nbest=aws_nbest,
        prev_word=prev_word,
        next_word=next_word,
        max_total=max_total,
    )

