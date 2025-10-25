"""Normalization candidate generator.

Generates candidate corrections by normalizing between different representations:
- Numbers: "two hundred five" ↔ "205", "2nd" ↔ "second"
- Acronyms: "u s a" ↔ "USA", "wifi" ↔ "Wi-Fi"
- Contractions: "we're" ↔ "we are", "it's" ↔ "it is"
"""

from typing import List, Set
import re


# Number words to digits
NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'million': 1000000,
}

# Digits to number words
DIGIT_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

# Ordinals
ORDINAL_WORDS = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th',
    'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
    'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th',
    'fifteenth': '15th', 'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th', 'thirtieth': '30th', 'fortieth': '40th',
    'fiftieth': '50th', 'sixtieth': '60th', 'seventieth': '70th', 'eightieth': '80th',
    'ninetieth': '90th', 'hundredth': '100th', 'thousandth': '1000th',
}

# Reverse ordinals
ORDINAL_DIGITS = {v: k for k, v in ORDINAL_WORDS.items()}

# Common contractions
CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "we'd": "we would",
    "they'd": "they would",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "can't": "cannot",
    "mightn't": "might not",
    "mustn't": "must not",
}

# Reverse contractions
EXPANDED_TO_CONTRACTION = {v: k for k, v in CONTRACTIONS.items()}

# Common acronyms (lowercase for matching)
ACRONYMS = {
    'usa': 'USA',
    'uk': 'UK',
    'eu': 'EU',
    'un': 'UN',
    'nato': 'NATO',
    'fbi': 'FBI',
    'cia': 'CIA',
    'nsa': 'NSA',
    'nasa': 'NASA',
    'wifi': 'Wi-Fi',
    'atm': 'ATM',
    'gps': 'GPS',
    'dna': 'DNA',
    'rna': 'RNA',
    'hiv': 'HIV',
    'aids': 'AIDS',
    'ceo': 'CEO',
    'cfo': 'CFO',
    'cto': 'CTO',
    'phd': 'PhD',
    'mba': 'MBA',
    'asap': 'ASAP',
    'rsvp': 'RSVP',
    'etc': 'etc.',
    'vs': 'vs.',
    'mr': 'Mr.',
    'mrs': 'Mrs.',
    'ms': 'Ms.',
    'dr': 'Dr.',
}


def normalize_number_word_to_digit(word: str) -> List[str]:
    """Convert number words to digit representations.
    
    Args:
        word: A number word like "five", "second", "1st"
    
    Returns:
        List of digit representations
    """
    candidates = []
    word_lower = word.lower().strip('.,!?;:')
    
    # Simple single number words
    if word_lower in NUMBER_WORDS:
        num = NUMBER_WORDS[word_lower]
        candidates.append(str(num))
    
    # Ordinal words to digits
    if word_lower in ORDINAL_WORDS:
        candidates.append(ORDINAL_WORDS[word_lower])
    
    # Already a digit - try ordinal
    if word_lower.isdigit():
        num = int(word_lower)
        if 1 <= num <= 100:
            # Add ordinal suffix
            if num % 10 == 1 and num != 11:
                candidates.append(f"{num}st")
            elif num % 10 == 2 and num != 12:
                candidates.append(f"{num}nd")
            elif num % 10 == 3 and num != 13:
                candidates.append(f"{num}rd")
            else:
                candidates.append(f"{num}th")
    
    return candidates


def normalize_digit_to_number_word(word: str) -> List[str]:
    """Convert digits to number word representations.
    
    Args:
        word: A digit like "5", "2nd"
    
    Returns:
        List of number word representations
    """
    candidates = []
    word_lower = word.lower().strip('.,!?;:')
    
    # Single digit to word
    if word_lower in DIGIT_TO_WORD:
        candidates.append(DIGIT_TO_WORD[word_lower])
    
    # Ordinal digit to word
    if word_lower in ORDINAL_DIGITS:
        candidates.append(ORDINAL_DIGITS[word_lower])
    
    # Multi-digit number to word (simple cases only)
    if word_lower.isdigit():
        num = int(word_lower)
        if num in [v for v in NUMBER_WORDS.values()]:
            # Find the word for this number
            for word_key, num_val in NUMBER_WORDS.items():
                if num_val == num:
                    candidates.append(word_key)
    
    return candidates


def normalize_contraction(word: str) -> List[str]:
    """Normalize contractions to expanded form and vice versa.
    
    Args:
        word: A word that might be a contraction
    
    Returns:
        List of normalized forms
    """
    candidates = []
    word_lower = word.lower().strip('.,!?;:')
    
    # Contraction to expanded
    if word_lower in CONTRACTIONS:
        expanded = CONTRACTIONS[word_lower]
        # Return as single token (will be split by caller if needed)
        candidates.append(expanded)
        # Also add capitalized version if original was capitalized
        if word[0].isupper():
            candidates.append(expanded.capitalize())
    
    # Expanded to contraction (check if word is part of an expansion)
    for contraction, expansion in CONTRACTIONS.items():
        if word_lower in expansion.split():
            # This word might be part of an expanded contraction
            # Return the contraction
            if word[0].isupper():
                candidates.append(contraction.capitalize())
            else:
                candidates.append(contraction)
    
    return candidates


def normalize_acronym(word: str) -> List[str]:
    """Normalize acronyms to standard forms.
    
    Args:
        word: A word that might be an acronym
    
    Returns:
        List of normalized forms
    """
    candidates = []
    word_lower = word.lower().strip('.,!?;:')
    
    # Check if it's a known acronym
    if word_lower in ACRONYMS:
        candidates.append(ACRONYMS[word_lower])
    
    # Check if it's spaced out (e.g., "u s a" -> "USA")
    if len(word_lower) == 1 and word_lower.isalpha():
        # Single letter - might be part of spaced acronym
        # Return uppercase version
        candidates.append(word_lower.upper())
    
    # If it's all caps, try lowercase
    if word.isupper() and len(word) > 1:
        candidates.append(word.lower())
        # Also try with periods (e.g., "USA" -> "U.S.A.")
        candidates.append('.'.join(word) + '.')
    
    return candidates


def generate_normalization_candidates(word: str) -> List[str]:
    """Generate all normalization candidates for a word.
    
    Args:
        word: Source word
    
    Returns:
        List of normalized candidate words
    """
    candidates = []
    
    # Try number normalizations
    candidates.extend(normalize_number_word_to_digit(word))
    candidates.extend(normalize_digit_to_number_word(word))
    
    # Try contraction normalizations
    candidates.extend(normalize_contraction(word))
    
    # Try acronym normalizations
    candidates.extend(normalize_acronym(word))
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for cand in candidates:
        if cand.lower() not in seen and cand.lower() != word.lower():
            seen.add(cand.lower())
            unique_candidates.append(cand)
    
    return unique_candidates

