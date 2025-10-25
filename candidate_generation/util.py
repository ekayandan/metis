"""Shared utilities for candidate generation.

Provides normalization, guards, and helper functions used across all generators.
"""

import unicodedata
import re
from typing import Optional

# Common stopwords that should not be corrected
STOPWORDS = {"the", "and", "to", "of", "a", "in", "is", "it", "for", "on", "with", "as", "at"}


def norm_text(s: str) -> str:
    """Normalize text to canonical form.
    
    Args:
        s: Input string
    
    Returns:
        Normalized string (NFKC, stripped)
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    return s


def is_capitalized(token: str) -> bool:
    """Check if token is properly capitalized (first letter uppercase).
    
    Args:
        token: Input token
    
    Returns:
        True if capitalized (e.g., "Hitachi", "Wi-Fi")
    """
    return bool(re.match(r"^[A-Z][\w\-\''']*$", token))


def length_class(s: str) -> int:
    """Get length class of a string.
    
    Args:
        s: Input string
    
    Returns:
        Length of string
    """
    return len(s)


def is_stopword(token: str) -> bool:
    """Check if token is a common stopword.
    
    Args:
        token: Input token
    
    Returns:
        True if token is a stopword
    """
    return token.lower().strip() in STOPWORDS


def clean_token(token: str) -> str:
    """Clean token by removing punctuation from edges.
    
    Args:
        token: Input token
    
    Returns:
        Cleaned token
    """
    return token.strip('.,!?;:\'"')


def same_first_letter(a: str, b: str) -> bool:
    """Check if two strings start with the same letter (case-insensitive).
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        True if both start with same letter
    """
    if not a or not b:
        return False
    return a[0].lower() == b[0].lower()


def contains_digit(s: str) -> bool:
    """Check if string contains any digits.
    
    Args:
        s: Input string
    
    Returns:
        True if contains digits
    """
    return bool(re.search(r'\d', s))


def is_numeric(s: str) -> bool:
    """Check if string is primarily numeric.
    
    Args:
        s: Input string
    
    Returns:
        True if mostly digits
    """
    clean = s.strip('.,!?;:\'"')
    if not clean:
        return False
    digit_count = sum(1 for c in clean if c.isdigit())
    return digit_count / len(clean) > 0.5

