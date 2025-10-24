"""AWS Transcribe transcript corrector using span-infilling model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Sequence

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "google/flan-t5-base"
CONFIDENCE_THRESHOLD = 0.75
MAX_CONTEXT_TOKENS = 200
MAX_ALTERNATIVES = 3
GENERATION_MAX_NEW_TOKENS = 8


@dataclass
class Token:
    text: str
    is_punctuation: bool
    confidence: float | None
    alternatives: List[str]


def load_items(path: Path) -> List[Token]:
    data = json.loads(path.read_text())
    results = data.get("results", {})
    items = results.get("items", [])
    tokens: List[Token] = []
    for item in items:
        alternatives = item.get("alternatives", [])
        if not alternatives:
            continue
        top_alt = alternatives[0]
        content = top_alt.get("content", "")
        confidence = float(top_alt.get("confidence")) if "confidence" in top_alt else None
        is_punctuation = item.get("type") == "punctuation"
        alt_words = [alt.get("content", "") for alt in alternatives]
        tokens.append(
            Token(
                text=content,
                is_punctuation=is_punctuation,
                confidence=confidence,
                alternatives=alt_words,
            )
        )
    return tokens


def format_tokens(tokens: Sequence[str], punct_flags: Sequence[bool]) -> str:
    parts: List[str] = []
    for tok, is_punct in zip(tokens, punct_flags):
        if not parts:
            parts.append(tok)
        elif is_punct or tok in {"'s", "'re", "'ve", "'ll", "'d", "n't"}:
            parts[-1] += tok
        else:
            parts.append(tok)
    return " ".join(parts)


def build_context(tokens: List[Token], index: int) -> str:
    start = max(0, index - MAX_CONTEXT_TOKENS // 2)
    end = min(len(tokens), start + MAX_CONTEXT_TOKENS)
    if end - start < MAX_CONTEXT_TOKENS:
        start = max(0, end - MAX_CONTEXT_TOKENS)
    context_tokens: List[str] = []
    punct_flags: List[bool] = []
    for i in range(start, end):
        token_text = "<mask>" if i == index else tokens[i].text
        context_tokens.append(token_text)
        punct_flags.append(tokens[i].is_punctuation)
    return format_tokens(context_tokens, punct_flags)


def select_replacement(generator, context: str, options: Sequence[str]) -> str | None:
    prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Options: " + ", ".join(options) + "\n"
        "Choose the best replacement for <mask> from the options. Respond with only the chosen word."
    )
    result = generator(
        prompt,
        max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        num_return_sequences=1,
    )[0]["generated_text"].strip()
    normalized = result.strip().strip("\"'")
    for option in options:
        if normalized.lower() == option.lower():
            return option
    return None


@lru_cache(maxsize=1)
def _load_generator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)


def correct_transcript(tokens: List[Token], generator=None) -> List[Token]:
    if generator is None:
        generator = _load_generator()

    for idx, token in enumerate(tokens):
        if token.is_punctuation or token.confidence is None:
            continue
        if token.confidence >= CONFIDENCE_THRESHOLD:
            continue
        options = token.alternatives[:MAX_ALTERNATIVES]
        if not options:
            continue
        context = build_context(tokens, idx)
        replacement = select_replacement(generator, context, options)
        if replacement and replacement in options:
            token.text = replacement
    return tokens


def tokens_to_text(tokens: Sequence[Token]) -> str:
    words = [token.text for token in tokens]
    punct_flags = [token.is_punctuation for token in tokens]
    return format_tokens(words, punct_flags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to AWS Transcribe JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write corrected transcript; prints to stdout if omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = load_items(args.input)
    corrected_tokens = correct_transcript(tokens)
    text = tokens_to_text(corrected_tokens)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
