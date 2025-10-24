"""Minimal AWS Transcribe post-correction pipeline."""
from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.correction.acoustic_alternative_expander import expand_with_homophones

# -------------------- Configuration -------------------- #
CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.9,
    "MAX_CONTEXT_TOKENS": 200,
    "MAX_ALTERNATIVES": 3,
    "MODEL_NAME": "google/flan-t5-base",
    "GENERATION_MAX_NEW_TOKENS": 6,
    "GENERATION_NUM_BEAMS": 4,
    "DEFAULT_INPUT_PATH": "transcribe_output.json",
    "DEFAULT_OUTPUT_PATH": "corrected_transcript.txt",
}
# ------------------------------------------------------- #

NO_SPACE_BEFORE = {
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    ")",
    "]",
    "}",
    "'",
    '"',
    "''",
    "n't",
    "'s",
    "'re",
    "'ve",
    "'m",
    "'d",
    "'ll",
}
NO_SPACE_AFTER = {"(", "[", "{", "``"}


@dataclass
class Token:
    index: int
    text: str
    confidence: Optional[float]
    alternatives: List[str]
    type: str
    is_punctuation: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct AWS Transcribe transcripts using a span-infilling model.")
    parser.add_argument("--input", default=CONFIG["DEFAULT_INPUT_PATH"], help="Path to AWS Transcribe JSON file.")
    parser.add_argument(
        "--output",
        default=CONFIG["DEFAULT_OUTPUT_PATH"],
        help="Where to write the corrected transcript (plain text).",
    )
    parser.add_argument(
        "--model",
        default=CONFIG["MODEL_NAME"],
        help="Hugging Face model name for span infilling.",
    )
    return parser.parse_args()


def load_transcribe_tokens(path: Path) -> List[Token]:
    data = json.loads(path.read_text())
    results = data.get("results", {})
    items = results.get("items", [])
    segments = results.get("segments", [])

    # AWS exposes richer alternatives at the segment level; gather these so the
    # correction model can consider them alongside the best token.
    by_time: dict[tuple[str, str], set[str]] = {}
    for segment in segments:
        for alt in segment.get("alternatives", []):
            for word in alt.get("items", []):
                if word.get("type") != "pronunciation":
                    continue
                content = word.get("content")
                if not content:
                    continue
                start = word.get("start_time") or ""
                end = word.get("end_time") or ""
                by_time.setdefault((start, end), set()).add(content)

    tokens: List[Token] = []
    for idx, item in enumerate(items):
        item_type = item.get("type", "")
        alternatives = item.get("alternatives", [])
        alt_contents = [alt.get("content", "") for alt in alternatives if alt.get("content")]
        start = item.get("start_time") or ""
        end = item.get("end_time") or ""
        extras = by_time.get((start, end))
        if extras:
            for extra in extras:
                if extra and extra not in alt_contents:
                    alt_contents.append(extra)
        if not alt_contents:
            continue
        content = alt_contents[0]
        conf_value: Optional[float] = None
        if alternatives:
            best_alt = alternatives[0]
            primary_content = best_alt.get("content")
            if primary_content:
                content = primary_content
            confidence = best_alt.get("confidence")
            if confidence is not None:
                conf_value = float(confidence)
        tokens.append(
            Token(
                index=len(tokens),
                text=content,
                confidence=conf_value,
                alternatives=alt_contents,
                type=item_type,
                is_punctuation=item_type == "punctuation",
            )
        )
    return tokens


def tokens_to_text(tokens: Sequence[Token], masked_index: Optional[int] = None) -> str:
    pieces: List[str] = []
    prev_text = ""
    for token in tokens:
        text = "<mask>" if masked_index is not None and token.index == masked_index else token.text
        if not text:
            continue
        if not pieces:
            pieces.append(text)
        else:
            if text in NO_SPACE_BEFORE or prev_text in NO_SPACE_AFTER:
                pieces[-1] += text
            else:
                pieces.append(" " + text)
        prev_text = text
    return "".join(pieces)


def build_context(tokens: Sequence[Token], focus_index: int) -> str:
    window = max(1, CONFIG["MAX_CONTEXT_TOKENS"])
    half_window = max(1, window // 2)
    start = max(0, focus_index - half_window)
    end = min(len(tokens), focus_index + half_window + 1)
    span = tokens[start:end]
    return tokens_to_text(span, masked_index=focus_index)


def select_alternative(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    context: str,
    options: Sequence[str],
) -> Optional[str]:
    if not options:
        return None
    option_list = ", ".join(options)
    prompt = (
        "Choose the best replacement for <mask> in the transcript. "
        "Return only the chosen word.\n"
        f"Context: {context}\n"
        f"Options: {option_list}"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=CONFIG["GENERATION_MAX_NEW_TOKENS"],
            num_beams=CONFIG["GENERATION_NUM_BEAMS"],
            early_stopping=True,
        )
    output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    normalized = {opt.lower(): opt for opt in options}
    cleaned = output.strip().lower().strip(string.punctuation)
    if cleaned in normalized:
        return normalized[cleaned]
    tokens = re.split(r"[^\w'’.-]+", output.lower())
    for token in tokens:
        token = token.strip(string.punctuation)
        if token in normalized:
            return normalized[token]
    for key, original in normalized.items():
        if key in output.lower():
            return original
    return None


def apply_corrections(tokens: List[Token], tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> None:
    threshold = CONFIG["CONFIDENCE_THRESHOLD"]
    for token in tokens:
        if token.type != "pronunciation":
            continue
        if token.confidence is None or token.confidence >= threshold:
            continue
        options = token.alternatives[: CONFIG["MAX_ALTERNATIVES"]]
        options = expand_with_homophones(options)
        if not options:
            continue
        context = build_context(tokens, token.index)
        choice = select_alternative(tokenizer, model, context, options)
        if choice and choice in options:
            token.text = choice


def save_transcript(tokens: Sequence[Token], output_path: Path) -> str:
    transcript = tokens_to_text(tokens)
    output_path.write_text(transcript)
    return transcript


def load_model(name: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.to(torch.device("cpu"))
    return tokenizer, model


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {input_path}")
    tokens = load_transcribe_tokens(input_path)
    if not tokens:
        raise ValueError("No tokens parsed from transcript.")
    tokenizer, model = load_model(args.model)
    apply_corrections(tokens, tokenizer, model)
    output_path = Path(args.output)
    transcript = save_transcript(tokens, output_path)
    print(transcript)


if __name__ == "__main__":
    main()
