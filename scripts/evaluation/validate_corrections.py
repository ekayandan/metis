#!/usr/bin/env python3
"""Validate corrections against reference transcript.

This script aligns the corrected transcript with the reference and shows
which corrections were correct, wrong, or neutral.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.measure_wer import load_transcripts, alignment_details


def load_corrections_trace(trace_path: Path) -> List[Dict]:
    """Load corrections from trace file."""
    corrections = []
    with open(trace_path) as f:
        for line in f:
            if line.strip():
                corrections.append(json.loads(line))
    return corrections


def build_token_map(transcribe_json_path: Path) -> Tuple[List[str], Dict[int, int]]:
    """Build mapping from token index to segment index."""
    data = json.loads(transcribe_json_path.read_text())
    results = data.get('results', {})
    segments = results.get('segments', [])
    
    tokens = []
    token_to_segment = {}
    
    for seg_idx, seg in enumerate(segments):
        alts = seg.get('alternatives', [])
        if not alts:
            continue
        
        primary_items = alts[0].get('items', [])
        for item in primary_items:
            if item.get('type') == 'pronunciation':
                token = item.get('content', '')
                token_idx = len(tokens)
                tokens.append(token)
                token_to_segment[token_idx] = seg_idx
    
    return tokens, token_to_segment


def find_correction_in_alignment(
    correction: Dict,
    original_tokens: List[str],
    corrected_tokens: List[str],
    token_to_segment: Dict[int, int],
) -> Optional[int]:
    """Find the global token index for a correction."""
    seg_idx = correction['segment_idx']
    tok_idx_in_seg = correction['token_idx']
    
    # Find all tokens in this segment
    segment_token_indices = [i for i, s in token_to_segment.items() if s == seg_idx]
    
    if tok_idx_in_seg < len(segment_token_indices):
        return segment_token_indices[tok_idx_in_seg]
    
    return None


def validate_corrections(
    reference_path: Path,
    transcribe_json_path: Path,
    corrected_transcript_path: Path,
    trace_path: Path,
) -> None:
    """Validate corrections against reference."""
    
    # Load reference
    ref_map = load_transcripts(reference_path, 1)
    ref_key = next(iter(ref_map))
    ref_tokens = ref_map[ref_key]
    
    # Load original tokens
    original_tokens, token_to_segment = build_token_map(transcribe_json_path)
    
    # Load corrected transcript (use load_transcripts for consistency)
    corr_map = load_transcripts(corrected_transcript_path, 1)
    corr_key = next(iter(corr_map))
    corrected_tokens = corr_map[corr_key]
    
    # Load corrections
    corrections = load_corrections_trace(trace_path)
    
    # Get alignments
    print("="*80)
    print("ALIGNMENT ANALYSIS")
    print("="*80)
    
    # Align original with reference
    subs_orig, dels_orig, ins_orig, path_orig = alignment_details(ref_tokens, original_tokens)
    print(f"Original vs Reference: {subs_orig} S, {dels_orig} D, {ins_orig} I")
    print(f"  WER: {(subs_orig + dels_orig + ins_orig) / len(ref_tokens) * 100:.2f}%")
    
    # Align corrected with reference
    subs_corr, dels_corr, ins_corr, path_corr = alignment_details(ref_tokens, corrected_tokens)
    print(f"Corrected vs Reference: {subs_corr} S, {dels_corr} D, {ins_corr} I")
    print(f"  WER: {(subs_corr + dels_corr + ins_corr) / len(ref_tokens) * 100:.2f}%")
    
    print()
    print(f"Change: {subs_corr - subs_orig:+d} S, {dels_corr - dels_orig:+d} D, {ins_corr - ins_orig:+d} I")
    print()
    
    # Build alignment maps
    orig_to_ref = {}  # original token index -> reference token
    for action, ref_idx, hyp_idx, ref_word, hyp_word in path_orig:
        if hyp_idx is not None and ref_idx is not None:
            orig_to_ref[hyp_idx] = (ref_word, action)
    
    corr_to_ref = {}  # corrected token index -> reference token
    for action, ref_idx, hyp_idx, ref_word, hyp_word in path_corr:
        if hyp_idx is not None and ref_idx is not None:
            corr_to_ref[hyp_idx] = (ref_word, action)
    
    # Validate each correction
    print("="*80)
    print(f"CORRECTION VALIDATION ({len(corrections)} corrections)")
    print("="*80)
    print()
    
    correct_count = 0
    wrong_count = 0
    neutral_count = 0
    unknown_count = 0
    
    for i, corr in enumerate(corrections, 1):
        seg_idx = corr['segment_idx']
        tok_idx = corr['token_idx']
        original = corr['original']
        replacement = corr['replacement']
        conf = corr['confidence']
        
        # Find global token index
        global_idx = find_correction_in_alignment(corr, original_tokens, corrected_tokens, token_to_segment)
        
        print(f"{i}. Segment {seg_idx}, Token #{tok_idx}: '{original}' → '{replacement}' (conf={conf:.3f})")
        
        if global_idx is None:
            print(f"   ⚠️  Could not locate token in alignment")
            unknown_count += 1
            print()
            continue
        
        # Get reference token
        ref_info = orig_to_ref.get(global_idx)
        if not ref_info:
            print(f"   ⚠️  No alignment to reference found")
            unknown_count += 1
            print()
            continue
        
        ref_word, action = ref_info
        
        # Normalize for comparison
        original_norm = original.strip('.,!?;:\'"').lower()
        replacement_norm = replacement.strip('.,!?;:\'"').lower()
        ref_norm = ref_word.strip('.,!?;:\'"').lower()
        
        print(f"   Reference: '{ref_word}'")
        
        if replacement_norm == ref_norm and original_norm != ref_norm:
            print(f"   ✅ CORRECT: Fixed '{original}' → '{replacement}' (matches reference)")
            correct_count += 1
        elif original_norm == ref_norm and replacement_norm != ref_norm:
            print(f"   ❌ WRONG: Broke '{original}' → '{replacement}' (original was correct)")
            wrong_count += 1
        elif original_norm != ref_norm and replacement_norm != ref_norm:
            print(f"   ⚪ NEUTRAL: '{original}' → '{replacement}' (both wrong, reference is '{ref_word}')")
            neutral_count += 1
        else:
            print(f"   ⚪ NO CHANGE: Both match reference")
            neutral_count += 1
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total corrections: {len(corrections)}")
    print(f"  ✅ Correct fixes: {correct_count} ({correct_count/len(corrections)*100:.1f}%)")
    print(f"  ❌ Wrong changes: {wrong_count} ({wrong_count/len(corrections)*100:.1f}%)")
    print(f"  ⚪ Neutral: {neutral_count} ({neutral_count/len(corrections)*100:.1f}%)")
    print(f"  ⚠️  Unknown: {unknown_count} ({unknown_count/len(corrections)*100:.1f}%)")
    print()
    print(f"Net WER change: {(subs_corr + dels_corr + ins_corr) - (subs_orig + dels_orig + ins_orig):+d} errors")
    print(f"Expected from corrections: ~{correct_count - wrong_count:+d} (correct - wrong)")


def main():
    parser = argparse.ArgumentParser(description="Validate corrections against reference")
    parser.add_argument("reference", type=Path, help="Reference TSV file")
    parser.add_argument("transcribe_json", type=Path, help="Original AWS Transcribe JSON")
    parser.add_argument("corrected_transcript", type=Path, help="Corrected transcript file")
    parser.add_argument("trace", type=Path, help="Correction trace JSONL file")
    
    args = parser.parse_args()
    
    validate_corrections(
        args.reference,
        args.transcribe_json,
        args.corrected_transcript,
        args.trace,
    )


if __name__ == "__main__":
    main()

