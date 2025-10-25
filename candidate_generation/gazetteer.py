"""Gazetteer for global proper nouns.

Indexes proper nouns (people, places, brands) from general sources like
Wikipedia/Wikidata for phonetic and signature-based lookup.
"""

from collections import defaultdict
from typing import List, Dict, Iterable, Optional
import logging

try:
    from metaphone import doublemetaphone
except ImportError:
    doublemetaphone = None

try:
    from wordfreq import word_frequency, zipf_frequency, iter_wordlist
except ImportError:
    word_frequency = None
    zipf_frequency = None
    iter_wordlist = None


logger = logging.getLogger(__name__)


class Gazetteer:
    """Index of proper nouns with phonetic and signature-based lookup."""
    
    def __init__(self, entries: Optional[Iterable[dict]] = None, min_zipf: float = 3.5):
        """Initialize gazetteer.
        
        Args:
            entries: Optional list of entries with 'surface' and 'zipf' keys
            min_zipf: Minimum Zipf frequency for auto-generated entries
        """
        self.entries = []
        self.by_meta = defaultdict(list)  # metaphone -> list of entry indices
        self.by_sig = defaultdict(list)   # (first_char, length) -> list of entry indices
        self.min_zipf = min_zipf
        
        if entries is not None:
            self._load_entries(entries)
        else:
            self._build_from_wordfreq()
    
    def _build_from_wordfreq(self):
        """Build gazetteer from wordfreq capitalized words."""
        if word_frequency is None:
            logger.warning("wordfreq not installed. Gazetteer will be empty.")
            return
        
        if doublemetaphone is None:
            logger.warning("metaphone not installed. Phonetic indexing disabled.")
        
        logger.info(f"Building gazetteer from wordfreq (min_zipf={self.min_zipf})...")
        
        # Use wordfreq's available_languages and word lists
        # We'll check common proper nouns by trying capitalized versions
        from wordfreq import top_n_list
        
        # Get top words and check capitalized versions
        base_words = top_n_list('en', 500000)
        count = 0
        
        for word in base_words:
            # Try capitalized version
            if word and word[0].islower():
                cap_word = word.capitalize()
                zipf = zipf_frequency(cap_word, 'en')
                
                # If capitalized version has decent frequency, it's likely a proper noun
                if zipf >= self.min_zipf:
                    self._add_entry({
                        'surface': cap_word,
                        'zipf': zipf,
                    })
                    count += 1
                    
                    # Also add all-caps version for acronyms
                    if len(word) <= 5:
                        upper_word = word.upper()
                        zipf_upper = zipf_frequency(upper_word, 'en')
                        if zipf_upper >= self.min_zipf:
                            self._add_entry({
                                'surface': upper_word,
                                'zipf': zipf_upper,
                            })
            
            # Also check if the word itself is capitalized in wordfreq
            if word and word[0].isupper():
                zipf = zipf_frequency(word, 'en')
                if zipf >= self.min_zipf:
                    self._add_entry({
                        'surface': word,
                        'zipf': zipf,
                    })
                    count += 1
        
        logger.info(f"Gazetteer built: {len(self.entries)} entries")
    
    def _load_entries(self, entries: Iterable[dict]):
        """Load entries from provided list."""
        for entry in entries:
            self._add_entry(entry)
        
        logger.info(f"Gazetteer loaded: {len(self.entries)} entries")
    
    def _add_entry(self, entry: dict):
        """Add an entry to the gazetteer and update indices."""
        surface = entry['surface']
        low = surface.lower()
        zipf = entry.get('zipf', 4.0)
        
        # Get metaphone codes
        m1, m2 = None, None
        if doublemetaphone:
            try:
                m1, m2 = doublemetaphone(low)
            except Exception as e:
                logger.debug(f"Failed to get metaphone for '{surface}': {e}")
        
        # Create signature (first char, length)
        sig = (low[0] if low else '', len(low))
        
        # Store entry
        idx = len(self.entries)
        rec = {
            'surface': surface,
            'low': low,
            'zipf': zipf,
            'm1': m1,
            'm2': m2,
            'sig': sig,
        }
        self.entries.append(rec)
        
        # Update indices
        if m1:
            self.by_meta[m1].append(idx)
        if m2 and m2 != m1:
            self.by_meta[m2].append(idx)
        
        # Add to signature index with ±1 length tolerance
        for L in (len(low) - 1, len(low), len(low) + 1):
            if L > 0:
                self.by_sig[(low[0] if low else '', L)].append(idx)
    
    def lookup(
        self,
        token: str,
        top_k_meta: int = 100,
        top_k_sig: int = 100,
    ) -> List[dict]:
        """Look up candidates for a token.
        
        Args:
            token: Source token
            top_k_meta: Max candidates from metaphone index
            top_k_sig: Max candidates from signature index
        
        Returns:
            List of entry dicts with 'surface', 'zipf', etc.
        """
        low = token.lower()
        out_ids = set()
        
        # Search by metaphone
        if doublemetaphone:
            try:
                m1, m2 = doublemetaphone(low)
                for m in (m1, m2):
                    if m and m in self.by_meta:
                        out_ids.update(self.by_meta[m][:top_k_meta])
            except Exception:
                pass
        
        # Search by signature (first char, length ±1)
        for L in (len(low) - 1, len(low), len(low) + 1):
            if L > 0:
                key = (low[0] if low else '', L)
                if key in self.by_sig:
                    ids = self.by_sig[key][:top_k_sig]
                    out_ids.update(ids)
        
        # Return unique entries
        return [self.entries[i] for i in out_ids]


def gazetteer_candidates(
    span_text: str,
    gaz: Gazetteer,
    max_out: int = 5,
) -> List[str]:
    """Generate gazetteer candidates for a span.
    
    Args:
        span_text: Source text
        gaz: Gazetteer instance
        max_out: Maximum candidates to return
    
    Returns:
        List of candidate surface forms
    """
    if not span_text:
        return []
    
    cands = []
    entries = gaz.lookup(span_text)
    
    # Sort by Zipf frequency (descending)
    entries.sort(key=lambda e: e['zipf'], reverse=True)
    
    for e in entries:
        surface = e['surface']
        # Skip if same as source
        if surface.lower() == span_text.lower():
            continue
        cands.append(surface)
        if len(cands) >= max_out:
            break
    
    return cands

