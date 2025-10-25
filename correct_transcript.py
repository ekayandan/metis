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

from scripts.correction.acoustic_alternative_expander import (
    expand_with_homophones,
    filter_by_phonetic_distance,
)
from scripts.transcribe.aws_transcribe_batch import extract_nbest

# -------------------- Configuration -------------------- #
CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.97,
    "MAX_CONTEXT_TOKENS": 200,
    "MAX_ALTERNATIVES": 10,
    "MODEL_NAME": "google/flan-t5-base",
    "GENERATION_MAX_NEW_TOKENS": 6,
    "GENERATION_NUM_BEAMS": 4,
    "DEFAULT_INPUT_PATH": "artifacts/whisper_transcribe_output.json",
    "DEFAULT_OUTPUT_PATH": "artifacts/whisper_corrected.txt",
    "ENABLE_HOMOPHONE_EXPANSION": True,
    "ENABLE_PHONETIC_FILTER": False,
    "PHONETIC_MAX_DISTANCE": 0,
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


@dataclass
class Segment:
    """Represents an AWS Transcribe segment with multiple alternative transcriptions."""
    start_time: Optional[float]
    end_time: Optional[float]
    alternatives: List[str]  # List of alternative transcripts
    primary_text: str  # The primary/best transcript (updated after correction)
    original_text: str  # The original primary text before correction
    items: List[dict]  # Token-level items for each alternative
    segment_index: int
    needs_correction: bool = False


def _normalize_for_comparison(text: str) -> str:
    return text.lower().strip(string.punctuation)


def is_degenerate_choice(choice: str, token_index: int, tokens: Sequence[Token], original: str) -> bool:
    """Reject replacements that duplicate neighbours or introduce whitespace splits."""

    if not choice:
        return True

    if " " in choice:
        return True

    candidate_norm = _normalize_for_comparison(choice)
    if not candidate_norm:
        return True

    original_norm = _normalize_for_comparison(original)
    if candidate_norm == original_norm:
        return True

    if token_index > 0:
        left_norm = _normalize_for_comparison(tokens[token_index - 1].text)
        if candidate_norm == left_norm:
            return True

    if token_index + 1 < len(tokens):
        right_norm = _normalize_for_comparison(tokens[token_index + 1].text)
        if candidate_norm == right_norm:
            return True

    return False


def _extract_alternatives_with_alignment(
    primary_tokens: list[str],
    candidate_sequences: list[list[str]],
) -> dict[int, list[str]]:
    """Extract alternatives using dynamic alignment (like WER measurement does).
    
    Maps each reference token position to alternatives found across N-best sequences.
    Unlike position-based matching, this handles insertions/deletions correctly.
    """
    alternatives_by_index: dict[int, list[str]] = {}
    
    for sequence in candidate_sequences:
        if not sequence:
            continue
        # Perform dynamic alignment to handle insertions/deletions
        ref_len = len(primary_tokens)
        hyp_len = len(sequence)
        dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
        backtrack = [["" for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]
        
        # Initialize base cases
        for i in range(1, ref_len + 1):
            dp[i][0] = i
            backtrack[i][0] = "del"
        for j in range(1, hyp_len + 1):
            dp[0][j] = j
            backtrack[0][j] = "ins"
        
        # Fill DP table
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if primary_tokens[i - 1].lower() == sequence[j - 1].lower():
                    dp[i][j] = dp[i - 1][j - 1]
                    backtrack[i][j] = "eq"
                else:
                    deletion = dp[i - 1][j] + 1
                    insertion = dp[i][j - 1] + 1
                    substitution = dp[i - 1][j - 1] + 1
                    best = min(deletion, insertion, substitution)
                    dp[i][j] = best
                    if best == substitution:
                        backtrack[i][j] = "sub"
                    elif best == deletion:
                        backtrack[i][j] = "del"
                    else:
                        backtrack[i][j] = "ins"
        
        # Backtrack to find alignments
        i, j = ref_len, hyp_len
        while i > 0 or j > 0:
            action = backtrack[i][j]
            if action in {"eq", "sub"}:
                ref_idx = i - 1
                hyp_word = sequence[j - 1]
                if hyp_word:
                    bucket = alternatives_by_index.setdefault(ref_idx, [])
                    if hyp_word not in bucket:
                        bucket.append(hyp_word)
                i -= 1
                j -= 1
            elif action == "del":
                i -= 1
            elif action == "ins":
                j -= 1
            else:
                break
    
    return alternatives_by_index


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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override confidence threshold (lower to consider more words).",
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=None,
        help="Override number of token alternatives to consider per word.",
    )
    parser.add_argument(
        "--enable-homophones",
        action="store_true",
        help="Expand alternatives with homophones during correction.",
    )
    parser.add_argument(
        "--disable-phonetic-filter",
        action="store_true",
        help="Skip phonetic-distance filtering when gathering alternatives.",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help="Optional JSONL file to record model prompts, options, and decisions.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional TSV with reference transcript (audio\ttext) for tracing mismatches.",
    )
    return parser.parse_args()


def load_transcribe_segments(path: Path) -> List[Segment]:
    """Load AWS Transcribe segments with their N-best alternatives."""
    print("\n" + "="*80)
    print("🔍 LOADING SEGMENTS FROM AWS JSON")
    print("="*80)
    data = json.loads(path.read_text())
    results = data.get("results", {})
    segments = results.get("segments", [])
    print(f"📊 Total segments in JSON: {len(segments)}")

    segments_data: List[Segment] = []
    for i, segment in enumerate(segments):
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")

        # Get all alternatives for this segment
        alternatives_data = segment.get("alternatives", [])
        if not alternatives_data:
            continue

        print(f"\n--- Segment #{i} ---")
        
        # Primary text is the first alternative (best hypothesis)
        primary_text = alternatives_data[0].get("transcript", "")
        print(f"📝 Primary: '{primary_text}'")

        # Get all alternative transcripts
        alternative_transcripts = [
            alt.get("transcript", "")
            for alt in alternatives_data
            if alt.get("transcript")
        ]
        print(f"🔢 AWS provided {len(alternative_transcripts)} alternatives:")
        for j, alt in enumerate(alternative_transcripts):
            print(f"   Alt {j}: '{alt}'")

        # Determine if this segment needs correction based on confidence and alternative quality
        needs_correction = False
        if alternatives_data:
            primary_items = alternatives_data[0].get("items", [])
            min_confidence = 1.0

            # Find minimum confidence in primary
            for item in primary_items:
                if item.get("type") == "pronunciation":
                    confidence = float(item.get("confidence", "1.0"))
                    min_confidence = min(min_confidence, confidence)

            # Check if any token is below threshold (traditional approach)
            if min_confidence < CONFIG["CONFIDENCE_THRESHOLD"]:
                needs_correction = True
                print(f"✅ Needs correction: LOW confidence ({min_confidence:.4f} < {CONFIG['CONFIDENCE_THRESHOLD']})")
            else:
                print(f"⚠️  High confidence ({min_confidence:.4f}), checking alternatives...")
                # For high-confidence segments, also consider if alternatives have similar confidence
                # This captures cases where AWS is uncertain between equally good options
                primary_total_conf = 0
                primary_count = 0
                for item in primary_items:
                    if item.get("type") == "pronunciation":
                        primary_total_conf += float(item.get("confidence", "1.0"))
                        primary_count += 1

                if primary_count > 0:
                    primary_avg_conf = primary_total_conf / primary_count
                    print(f"   Primary avg confidence: {primary_avg_conf:.4f}")

                    # Check if any alternative has similar average confidence (±0.01)
                    for alt_idx, alt_data in enumerate(alternatives_data[1:], 1):  # Skip primary
                        alt_items = alt_data.get("items", [])
                        alt_total_conf = 0
                        alt_count = 0
                        for item in alt_items:
                            if item.get("type") == "pronunciation":
                                alt_total_conf += float(item.get("confidence", "1.0"))
                                alt_count += 1

                        if alt_count > 0:
                            alt_avg_conf = alt_total_conf / alt_count
                            conf_diff = abs(alt_avg_conf - primary_avg_conf)
                            print(f"   Alt {alt_idx} avg confidence: {alt_avg_conf:.4f} (diff: {conf_diff:.4f})")
                            # If alternative has very similar confidence to primary (±0.005), consider for correction
                            # This is more conservative to avoid over-correction
                            if conf_diff <= 0.005:
                                needs_correction = True
                                print(f"✅ Needs correction: Alternative has similar confidence (diff={conf_diff:.4f})")
                break

        if not needs_correction:
            print(f"❌ Skipping: No correction needed")

        segments_data.append(Segment(
            start_time=float(start_time) if start_time else None,
            end_time=float(end_time) if end_time else None,
            alternatives=alternative_transcripts,
            primary_text=primary_text,
            original_text=primary_text,  # Initially same as primary_text
            items=alternatives_data,
            segment_index=i,
            needs_correction=needs_correction,
        ))

    correction_count = sum(1 for s in segments_data if s.needs_correction)
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY: {len(segments_data)} segments loaded, {correction_count} need correction")
    print(f"{'='*80}\n")
    return segments_data


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


def segments_to_text(segments: Sequence[Segment]) -> str:
    """Convert segments to final transcript text."""
    pieces: List[str] = []
    for segment in segments:
        text = segment.primary_text
        if not text:
            continue
        if not pieces:
            pieces.append(text)
        else:
            pieces.append(" " + text)
    return "".join(pieces)


def build_context(tokens: Sequence[Token], focus_index: int) -> str:
    window = max(1, CONFIG["MAX_CONTEXT_TOKENS"])
    half_window = max(1, window // 2)
    start = max(0, focus_index - half_window)
    end = min(len(tokens), focus_index + half_window + 1)
    span = tokens[start:end]
    return tokens_to_text(span, masked_index=focus_index)


def build_segment_context(segments: Sequence[Segment], focus_index: int) -> str:
    """Build context around a segment for LLM prompting."""
    window = max(1, CONFIG["MAX_CONTEXT_TOKENS"] // 50)  # Segments are larger than tokens
    half_window = max(1, window // 2)
    start = max(0, focus_index - half_window)
    end = min(len(segments), focus_index + half_window + 1)
    span = segments[start:end]

    # Convert segment span to text
    context_segments = []
    for segment in span:
        context_segments.append(segment.primary_text)

    return " ".join(context_segments)


def select_alternative(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    context: str,
    options: Sequence[str],
) -> tuple[Optional[str], str, str]:
    if not options:
        return None, "", ""
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
        return normalized[cleaned], output, prompt
    tokens = re.split(r"[^\w'’.-]+", output.lower())
    for token in tokens:
        token = token.strip(string.punctuation)
        if token in normalized:
            return normalized[token], output, prompt
    for key, original in normalized.items():
        if key in output.lower():
            return original, output, prompt
    return None, output, prompt


def select_segment_alternative(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    context: str,
    options: Sequence[str],
) -> tuple[Optional[str], str, str]:
    """Select the best alternative for a segment (phrase-level correction)."""
    if not options:
        return None, "", ""

    # For segment correction, we need a larger generation limit since segments can be longer
    max_tokens = max(50, CONFIG["GENERATION_MAX_NEW_TOKENS"] * 10)

    option_list = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    prompt = (
        "Choose the best transcription for this segment of the transcript. "
        "The segment needs correction due to low confidence in the original transcription. "
        "Return only the chosen transcription, exactly as it appears in the options.\n\n"
        f"Context: {context}\n\n"
        f"Segment options:\n{option_list}\n\n"
        "Choose the best option (1, 2, 3, etc.) or return the text if none seem correct:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=CONFIG["GENERATION_NUM_BEAMS"],
            early_stopping=True,
        )

    output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    # First try exact match with options
    normalized = {opt.lower().strip(): opt for opt in options}
    cleaned = output.lower().strip()
    if cleaned in normalized:
        return normalized[cleaned], output, prompt

    # Try to find option numbers (1, 2, 3, etc.)
    import re
    number_match = re.search(r'(\d+)', output.strip())
    if number_match:
        try:
            option_num = int(number_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= option_num < len(options):
                return options[option_num], output, prompt
        except (ValueError, IndexError):
            pass

    # Try partial matches within the output
    for option in options:
        if option.lower().strip() in cleaned or cleaned in option.lower().strip():
            return option, output, prompt

    return None, output, prompt


def validate_correction_effectiveness(
    segments: List[Segment],
    reference_tokens: Optional[List[str]] = None,
) -> dict[str, float]:
    """Validate how often we provide correct alternatives and select them."""
    if not reference_tokens:
        return {"error": "No reference provided"}

    reference_text = " ".join(reference_tokens)
    ref_lower = reference_text.lower()

    print("\n" + "="*80)
    print("🔍 VALIDATING CORRECTION EFFECTIVENESS")
    print("="*80)

    stats = {
        "total_corrections": 0,
        "had_correct_option": 0,
        "chose_correct_option": 0,
        "missed_opportunities": 0,
        "no_reference_match": 0,
    }

    for segment in segments:
        if not segment.needs_correction or len(segment.alternatives) <= 1:
            continue

        stats["total_corrections"] += 1
        print(f"\n--- Validating Segment #{segment.segment_index} ---")

        # Check if final result matches reference
        final_lower = segment.primary_text.lower()
        pos = ref_lower.find(final_lower[:50])
        if pos == -1:
            print(f"⚠️  Final text not found in reference")
            print(f"   Final: '{segment.primary_text[:80]}'")
            stats["no_reference_match"] += 1
            continue

        # Get reference context
        start = max(0, pos - 20)
        end = min(len(reference_text), pos + len(segment.primary_text) + 20)
        ref_context = reference_text[start:end]
        
        print(f"📝 Original: '{segment.original_text}'")
        print(f"📝 Final: '{segment.primary_text}'")
        print(f"📚 Reference: '{ref_context}'")
        
        # Check which alternatives match the reference
        matching_alts = []
        for alt in segment.alternatives:
            if alt.lower().strip() in ref_context.lower():
                matching_alts.append(alt)
        
        if matching_alts:
            print(f"✅ Correct alternatives available: {matching_alts}")
        else:
            print(f"❌ No correct alternatives in list")

        # Check if final result matches reference
        if segment.primary_text.lower().strip() in ref_context.lower():
            if segment.original_text.lower().strip() != segment.primary_text.lower().strip():
                # We made a correction that was correct
                stats["chose_correct_option"] += 1
                stats["had_correct_option"] += 1  # We had the correct option and chose it
                print(f"✅ Made correct correction!")
            else:
                # Original was already correct
                stats["chose_correct_option"] += 1
                stats["had_correct_option"] += 1
                print(f"✅ Original was already correct")
        else:
            # Final result doesn't match reference
            if segment.original_text.lower().strip() in ref_context.lower():
                # Original was correct but we changed it (missed opportunity)
                stats["missed_opportunities"] += 1
                stats["had_correct_option"] += 1
                print(f"❌ Original was correct but we changed it!")
            else:
                # Neither original nor final match reference
                stats["total_corrections"] -= 1  # Don't count this as a correction attempt
                print(f"⚠️  Neither original nor final match reference")


    # Calculate rates
    if stats["total_corrections"] > 0:
        stats["correct_option_rate"] = stats["had_correct_option"] / stats["total_corrections"]
        stats["success_rate"] = stats["chose_correct_option"] / stats["total_corrections"]
        stats["missed_rate"] = stats["missed_opportunities"] / stats["total_corrections"]
        if stats["had_correct_option"] > 0:
            stats["llm_accuracy"] = stats["chose_correct_option"] / stats["had_correct_option"]

    return stats


def apply_word_level_corrections(
    segments: List[Segment],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    trace_path: Optional[Path] = None,
) -> List[dict[str, object]]:
    """Apply word-level corrections using N-best token alternatives.
    
    This is Pass 1 of two-level correction: fix individual low-confidence words
    by selecting from N-best alternatives at the same position.
    """
    print("\n" + "="*80)
    print("🔧 PASS 1: WORD-LEVEL CORRECTION")
    print("="*80)
    
    trace_handle = None
    if trace_path is not None:
        word_trace_path = trace_path.parent / (trace_path.stem + "_word" + trace_path.suffix)
        if word_trace_path.parent:
            word_trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = word_trace_path.open("w", encoding="utf-8")
    
    applied_changes: List[dict[str, object]] = []
    threshold = CONFIG["CONFIDENCE_THRESHOLD"]
    
    # Extract N-best token sequences from segments
    # Build a token-level view with alternatives
    all_tokens = []
    for seg in segments:
        # Get all alternatives for this segment
        seg_alternatives = seg.alternatives if seg.alternatives else [seg.primary_text]
        
        # Tokenize each alternative
        alt_token_lists = [alt.split() for alt in seg_alternatives]
        
        # For each token position in primary, collect alternatives
        primary_tokens = seg.primary_text.split()
        for token_idx, primary_token in enumerate(primary_tokens):
            # Collect alternatives at this position from all segment alternatives
            token_alternatives = [primary_token]  # Start with primary
            
            for alt_tokens in alt_token_lists[1:]:  # Skip primary (already added)
                if token_idx < len(alt_tokens):
                    alt_token = alt_tokens[token_idx]
                    if alt_token not in token_alternatives:
                        token_alternatives.append(alt_token)
            
            # Get confidence for this token (from primary segment items)
            confidence = 1.0
            if seg.items and len(seg.items) > 0:
                primary_items = seg.items[0].get('items', []) if isinstance(seg.items[0], dict) else []
                # Find token in items
                token_count = 0
                for item in primary_items:
                    if item.get('type') == 'pronunciation':
                        if token_count == token_idx:
                            confidence = float(item.get('confidence', 1.0))
                            break
                        token_count += 1
            
            all_tokens.append({
                'segment_idx': seg.segment_index,
                'token_idx': token_idx,
                'token': primary_token,
                'alternatives': token_alternatives,
                'confidence': confidence,
            })
    
    print(f"📊 Extracted {len(all_tokens)} tokens from {len(segments)} segments")
    
    # Apply corrections to low-confidence tokens
    corrections_made = 0
    for token_info in all_tokens:
        if token_info['confidence'] >= threshold:
            continue  # High confidence, skip
        
        if len(token_info['alternatives']) <= 1:
            continue  # No alternatives
        
        # Use LLM to select best alternative for this token
        alternatives = token_info['alternatives']
        
        if len(alternatives) <= 1:
            continue
        
        # Build context: get surrounding tokens from the segment
        seg_idx = token_info['segment_idx']
        tok_idx = token_info['token_idx']
        
        # Find the segment
        context_segment = None
        for seg in segments:
            if seg.segment_index == seg_idx:
                context_segment = seg
                break
        
        if not context_segment:
            continue
        
        # Build context with the token masked
        seg_tokens = context_segment.primary_text.split()
        context_before = ' '.join(seg_tokens[max(0, tok_idx-5):tok_idx])
        context_after = ' '.join(seg_tokens[tok_idx+1:min(len(seg_tokens), tok_idx+6)])
        
        # Prompt LLM to choose best word (or suggest a better one)
        option_list = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(alternatives)])
        prompt = (
            f"Fill in the blank with the most appropriate word based on context.\n\n"
            f"Context: {context_before} _____ {context_after}\n\n"
            f"Suggested options:\n{option_list}\n\n"
            f"You may choose one of these options (by number) or suggest a better word if none fit well. "
            f"Return only the word or number:"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
                early_stopping=True,
            )
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        
        # Parse LLM output - try multiple strategies
        chosen = None
        import re
        
        # Strategy 1: Check if output is a number (selecting from options)
        number_match = re.search(r'^(\d+)', output)
        if number_match:
            try:
                option_num = int(number_match.group(1)) - 1
                if 0 <= option_num < len(alternatives):
                    chosen = alternatives[option_num]
            except (ValueError, IndexError):
                pass
        
        # Strategy 2: Check if output is a direct word suggestion
        if not chosen:
            # Clean the output (remove punctuation for comparison)
            output_clean = output.strip('.,!?;:\'"').lower()
            
            # First check if it matches any alternative (case-insensitive)
            for alt in alternatives:
                if alt.strip('.,!?;:\'"').lower() == output_clean:
                    chosen = alt
                    break
            
            # If not in alternatives, accept the LLM's suggestion if it's a single word
            if not chosen and output_clean and ' ' not in output_clean and len(output_clean) > 1:
                # Use the LLM's suggestion, preserving punctuation from original
                original_token = token_info['token']
                # Try to preserve punctuation
                if original_token and original_token[-1] in '.,!?;:':
                    chosen = output_clean + original_token[-1]
                else:
                    chosen = output_clean
        
        # If LLM didn't pick or picked the primary, skip
        if not chosen or chosen.strip('.,!?;:\'"').lower() == token_info['token'].strip('.,!?;:\'"').lower():
            continue
        
        # Update the token in the segment
        seg_idx = token_info['segment_idx']
        tok_idx = token_info['token_idx']
        
        # Find the segment and update its primary_text
        for seg in segments:
            if seg.segment_index == seg_idx:
                tokens = seg.primary_text.split()
                if tok_idx < len(tokens):
                    old_token = tokens[tok_idx]
                    tokens[tok_idx] = chosen
                    seg.primary_text = ' '.join(tokens)
                    
                    corrections_made += 1
                    applied_changes.append({
                        'segment_idx': seg_idx,
                        'token_idx': tok_idx,
                        'original': old_token,
                        'replacement': chosen,
                        'confidence': token_info['confidence'],
                    })
                    
                    if trace_handle:
                        json.dump({
                            'segment_idx': seg_idx,
                            'token_idx': tok_idx,
                            'original': old_token,
                            'replacement': chosen,
                            'confidence': token_info['confidence'],
                            'alternatives': token_info['alternatives'],
                            'llm_output': output,
                            'prompt': prompt,
                        }, trace_handle)
                        trace_handle.write("\n")
                break
    
    if trace_handle:
        trace_handle.close()
    
    print(f"\n{'='*80}")
    print(f"📊 WORD-LEVEL SUMMARY: {corrections_made} tokens corrected")
    print(f"{'='*80}\n")
    
    return applied_changes


def apply_segment_corrections(
    segments: List[Segment],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    trace_path: Optional[Path] = None,
    reference_text: Optional[str] = None,
) -> List[dict[str, object]]:
    """Apply corrections at the segment level instead of token level.
    
    This is Pass 2 of two-level correction: improve phrasing by selecting
    best segment alternatives using LLM.
    """
    print("\n" + "="*80)
    print("🔧 PASS 2: SEGMENT-LEVEL CORRECTION")
    print("="*80)
    
    threshold = CONFIG["CONFIDENCE_THRESHOLD"]
    trace_handle = None
    if trace_path is not None:
        if trace_path.parent:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_path.open("w", encoding="utf-8")

    applied_changes: List[dict[str, object]] = []
    correction_candidates = [s for s in segments if s.needs_correction]
    print(f"📊 Processing {len(correction_candidates)} correction candidates\n")

    for segment in segments:
        # Skip segments that don't need correction (high confidence)
        if not segment.needs_correction:
            continue

        # Skip segments with only one alternative (no choice to make)
        if len(segment.alternatives) <= 1:
            print(f"⚠️  Segment #{segment.segment_index}: Only 1 alternative, skipping")
            continue

        print(f"\n{'─'*80}")
        print(f"🔍 Processing Segment #{segment.segment_index}")
        print(f"{'─'*80}")

        original_text = segment.primary_text
        options = segment.alternatives
        
        print(f"📝 Original: '{original_text}'")
        print(f"🔢 Initial alternatives from AWS: {len(options)}")
        for i, opt in enumerate(options[:5]):  # Show first 5
            print(f"   {i+1}. '{opt}'")
        if len(options) > 5:
            print(f"   ... and {len(options)-5} more")

        # Apply homophone expansion to all alternatives
        if CONFIG.get("ENABLE_HOMOPHONE_EXPANSION", True):
            expanded_options = []
            for opt in options:
                expanded = expand_with_homophones([opt])
                expanded_options.extend(expanded)
            options = list(set(expanded_options))  # Remove duplicates
            print(f"🔄 After homophone expansion: {len(options)} alternatives")

        # Limit alternatives
        options = options[:CONFIG["MAX_ALTERNATIVES"]]
        print(f"✂️  After limiting to MAX_ALTERNATIVES ({CONFIG['MAX_ALTERNATIVES']}): {len(options)} alternatives")

        if not options:
            print(f"❌ No options available after processing")
            continue

        # Build context from surrounding segments
        context = build_segment_context(segments, segment.segment_index)
        print(f"📖 Context: '{context[:100]}...'")

        # Use LLM to select the best alternative
        print(f"🤖 Sending to LLM for selection...")
        choice, raw_output, prompt = select_segment_alternative(tokenizer, model, context, options)
        print(f"💡 LLM selected: '{choice}'" if choice else "❌ LLM made no valid choice")

        trace_entry = {
            "segment_index": segment.segment_index,
            "original": original_text,
            "alternatives": options,
            "context": context,
            "prompt": prompt,
            "model_output": raw_output,
            "selected": choice,
        }

        # Check if LLM made a valid choice
        if not choice or choice not in options:
            print(f"⚠️  LLM choice not valid or not in options list")
            if trace_handle:
                trace_entry["reason"] = "model_no_choice"
                trace_entry["applied"] = False
                json.dump(trace_entry, trace_handle)
                trace_handle.write("\n")
            continue

        # Apply the correction
        segment.original_text = segment.primary_text  # Store original before changing
        segment.primary_text = choice
        
        if choice != original_text:
            print(f"✅ APPLIED: '{original_text}' → '{choice}'")
        else:
            print(f"➡️  KEPT: LLM chose original (no change)")
        
        applied_changes.append({
            "segment_index": segment.segment_index,
            "original": original_text,
                "replacement": choice,
            "alternatives": options,
        })

        if trace_handle:
            trace_entry["reason"] = "applied"
            trace_entry["applied"] = True
            json.dump(trace_entry, trace_handle)
            trace_handle.write("\n")

    if trace_handle:
        trace_handle.close()
    
    print(f"\n{'='*80}")
    print(f"📊 CORRECTION SUMMARY: {len(applied_changes)} corrections applied")
    print(f"{'='*80}\n")
    return applied_changes


def apply_corrections(
    tokens: List[Token],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2qLM,
    trace_path: Optional[Path] = None,
    reference_tokens: Optional[Sequence[str]] = None,
) -> List[dict[str, object]]:
    """Legacy function for backward compatibility - delegates to segment-based correction."""
    # For now, return empty changes since we're moving to segment-based correction
    # This function can be removed once we fully migrate
    return []


def save_transcript(segments: Sequence[Segment], output_path: Path) -> str:
    transcript = segments_to_text(segments)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
    segments = load_transcribe_segments(input_path)
    if not segments:
        raise ValueError("No segments parsed from transcript.")

    if args.confidence_threshold is not None:
        CONFIG["CONFIDENCE_THRESHOLD"] = args.confidence_threshold
    if args.max_alternatives is not None and args.max_alternatives > 0:
        CONFIG["MAX_ALTERNATIVES"] = args.max_alternatives
    if args.enable_homophones:
        CONFIG["ENABLE_HOMOPHONE_EXPANSION"] = True
    if args.disable_phonetic_filter:
        CONFIG["ENABLE_PHONETIC_FILTER"] = False

    tokenizer, model = load_model(args.model)
    trace_path = Path(args.trace_file) if args.trace_file else None

    # TWO-LEVEL CORRECTION SYSTEM
    # Pass 1: Word-level correction (recover alt-hits)
    word_changes = apply_word_level_corrections(
        segments,
        tokenizer,
        model,
        trace_path=trace_path,
    )
    
    # Pass 2: Segment-level correction (improve phrasing) - DISABLED FOR NOW
    # segment_changes = apply_segment_corrections(
    #     segments,
    #     tokenizer,
    #     model,
    #     trace_path=trace_path,
    # )
    segment_changes = []  # Disabled
    
    # Combine changes from both passes
    changes = word_changes + segment_changes
    
    output_path = Path(args.output)
    transcript = save_transcript(segments, output_path)

    # Validate effectiveness if reference is available
    reference_tokens = None
    if args.reference:
        from scripts.evaluation.measure_wer import load_transcripts
        ref_map = load_transcripts(args.reference, 1)
        key = next(iter(ref_map))
        reference_tokens = ref_map[key]  # Already tokenized as List[str]

        validation_stats = validate_correction_effectiveness(segments, reference_tokens)
        print("\n📊 VALIDATION RESULTS:")
        print(f"  Total correction attempts: {validation_stats['total_corrections']}")
        print(f"  Had correct option: {validation_stats.get('had_correct_option', 0)} ({validation_stats.get('correct_option_rate', 0)*100:.1f}%)")
        print(f"  Chose correct option: {validation_stats.get('chose_correct_option', 0)} ({validation_stats.get('success_rate', 0)*100:.1f}%)")
        print(f"  Missed opportunities: {validation_stats.get('missed_opportunities', 0)} ({validation_stats.get('missed_rate', 0)*100:.1f}%)")
        if validation_stats.get('had_correct_option', 0) > 0:
            llm_accuracy = validation_stats['chose_correct_option'] / validation_stats['had_correct_option'] * 100
            print(f"  LLM accuracy when correct option available: {llm_accuracy:.1f}%")

    if changes:
        print("\n📝 Corrections applied:")
        print(f"  Word-level: {len(word_changes)} tokens")
        print(f"  Segment-level: {len(segment_changes)} segments")
        print()
        
        if word_changes:
            print("  Word-level changes:")
            for change in word_changes[:10]:  # Show first 10
                print(f"    Seg #{change['segment_idx']}, Token #{change['token_idx']}: '{change['original']}' -> '{change['replacement']}'")
            if len(word_changes) > 10:
                print(f"    ... and {len(word_changes) - 10} more")
        
        if segment_changes:
            print("\n  Segment-level changes:")
            for change in segment_changes[:5]:  # Show first 5
                idx = change["segment_index"]
            original = change["original"]
            replacement = change["replacement"]
                print(f"    Segment #{idx}: '{original[:60]}...' -> '{replacement[:60]}...'")
    else:
        print("No corrections applied.")
    print(f"\nFinal transcript:\n{transcript}")


if __name__ == "__main__":
    main()
