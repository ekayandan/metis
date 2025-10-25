#!/usr/bin/env python3
"""Test candidate generation coverage on a dataset.

Measures how often the correct reference word appears in the generated candidates
from multiple sources (AWS N-best, phonetic, normalization, word-boundary).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.measure_wer import load_transcripts, alignment_details
from candidate_generation import (
    PhoneticNeighborGenerator,
    generate_normalization_candidates,
    generate_word_boundary_candidates,
)


def load_aws_alternatives(transcribe_json_path: Path, token_index: int) -> List[str]:
    """Load AWS N-best alternatives for a specific token."""
    data = json.loads(transcribe_json_path.read_text())
    results = data.get('results', {})
    segments = results.get('segments', [])
    
    # Build token list with alternatives
    token_idx = 0
    for seg in segments:
        alts = seg.get('alternatives', [])
        if not alts:
            continue
        
        # Get all segment alternatives
        seg_alternatives = [alt.get('transcript', '') for alt in alts if alt.get('transcript')]
        alt_token_lists = [alt.split() for alt in seg_alternatives]
        
        # For each token position in primary
        primary_tokens = seg_alternatives[0].split()
        for tok_pos, primary_token in enumerate(primary_tokens):
            if token_idx == token_index:
                # Collect alternatives at this position
                token_alternatives = [primary_token]
                for alt_tokens in alt_token_lists[1:]:
                    if tok_pos < len(alt_tokens):
                        alt_token = alt_tokens[tok_pos]
                        if alt_token not in token_alternatives:
                            token_alternatives.append(alt_token)
                return token_alternatives
            token_idx += 1
    
    return []


def test_coverage(
    reference_path: Path,
    transcribe_json_path: Path,
    confidence_threshold: float = 0.7,
) -> Dict:
    """Test candidate generation coverage."""
    
    # Load reference
    ref_map = load_transcripts(reference_path, 1)
    ref_key = next(iter(ref_map))
    ref_tokens = ref_map[ref_key]
    
    # Load AWS transcribe output
    data = json.loads(transcribe_json_path.read_text())
    results = data.get('results', {})
    segments = results.get('segments', [])
    
    # Build primary hypothesis tokens with confidence
    primary_tokens = []
    for seg in segments:
        alts = seg.get('alternatives', [])
        if not alts:
            continue
        
        primary_items = alts[0].get('items', [])
        for item in primary_items:
            if item.get('type') == 'pronunciation':
                token = item.get('content', '')
                conf = float(item.get('confidence', 1.0))
                primary_tokens.append({'token': token, 'confidence': conf})
    
    # Get alignment
    hyp_tokens = [t['token'] for t in primary_tokens]
    subs, dels, ins, path = alignment_details(ref_tokens, hyp_tokens)
    
    # Find low-confidence wrong tokens
    low_conf_errors = []
    for action, ref_idx, hyp_idx, ref_word, hyp_word in path:
        if action == 'sub' and hyp_idx is not None and ref_idx is not None:
            if hyp_idx < len(primary_tokens):
                conf = primary_tokens[hyp_idx]['confidence']
                if conf < confidence_threshold:
                    low_conf_errors.append({
                        'hyp_word': hyp_word,
                        'ref_word': ref_word,
                        'confidence': conf,
                        'token_index': hyp_idx,
                    })
    
    print(f"Found {len(low_conf_errors)} low-confidence errors (conf < {confidence_threshold})")
    print()
    
    # Initialize generators
    print("Initializing candidate generators...")
    phonetic_gen = PhoneticNeighborGenerator(lexicon_size=None, max_phoneme_distance=2)
    print(f"  Phonetic lexicon: {len(phonetic_gen.lexicon)} words")
    print()
    
    # Test coverage from each source
    coverage_stats = {
        'total': len(low_conf_errors),
        'aws_nbest': 0,
        'phonetic': 0,
        'normalization': 0,
        'word_boundary': 0,
        'any_source': 0,
    }
    
    for error in low_conf_errors:
        hyp = error['hyp_word']
        ref = error['ref_word']
        conf = error['confidence']
        token_idx = error['token_index']
        
        ref_clean = ref.lower().strip('.,!?;:')
        
        print(f"{hyp:15} (conf={conf:.3f}) -> ref=\"{ref}\"")
        
        # Check AWS N-best
        aws_alts = load_aws_alternatives(transcribe_json_path, token_idx)
        aws_alts_clean = [a.lower().strip('.,!?;:') for a in aws_alts]
        in_aws = ref_clean in aws_alts_clean
        if in_aws:
            coverage_stats['aws_nbest'] += 1
            print(f"  ✅ AWS N-best")
        
        # Check phonetic
        phonetic_cands = phonetic_gen.generate_candidates(hyp, max_candidates=20)
        phonetic_words = [c.word.lower().strip('.,!?;:') for c in phonetic_cands]
        in_phonetic = ref_clean in phonetic_words
        if in_phonetic:
            coverage_stats['phonetic'] += 1
            print(f"  ✅ Phonetic")
        
        # Check normalization
        norm_cands = generate_normalization_candidates(hyp)
        norm_words = [c.lower().strip('.,!?;:') for c in norm_cands]
        in_norm = ref_clean in norm_words
        if in_norm:
            coverage_stats['normalization'] += 1
            print(f"  ✅ Normalization")
        
        # Check word-boundary
        wb_cands = generate_word_boundary_candidates(hyp)
        wb_words = [c.lower().strip('.,!?;:') for c in wb_cands]
        in_wb = ref_clean in wb_words
        if in_wb:
            coverage_stats['word_boundary'] += 1
            print(f"  ✅ Word-boundary")
        
        # Check if found in any source
        if in_aws or in_phonetic or in_norm or in_wb:
            coverage_stats['any_source'] += 1
            if not in_aws:
                print(f"  🌟 NEW (not in AWS N-best)!")
        else:
            print(f"  ❌ NOT FOUND in any source")
        
        print()
    
    return coverage_stats


def main():
    parser = argparse.ArgumentParser(description="Test candidate generation coverage")
    parser.add_argument("reference", type=Path, help="Reference TSV file")
    parser.add_argument("transcribe_json", type=Path, help="AWS Transcribe JSON")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence threshold for low-confidence errors")
    
    args = parser.parse_args()
    
    stats = test_coverage(
        args.reference,
        args.transcribe_json,
        args.confidence_threshold,
    )
    
    print("="*80)
    print("COVERAGE SUMMARY")
    print("="*80)
    total = stats['total']
    if total > 0:
        print(f"Total low-confidence errors: {total}")
        print()
        print(f"AWS N-best:        {stats['aws_nbest']:3d} / {total} = {stats['aws_nbest']/total*100:5.1f}%")
        print(f"Phonetic:          {stats['phonetic']:3d} / {total} = {stats['phonetic']/total*100:5.1f}%")
        print(f"Normalization:     {stats['normalization']:3d} / {total} = {stats['normalization']/total*100:5.1f}%")
        print(f"Word-boundary:     {stats['word_boundary']:3d} / {total} = {stats['word_boundary']/total*100:5.1f}%")
        print()
        print(f"ANY SOURCE:        {stats['any_source']:3d} / {total} = {stats['any_source']/total*100:5.1f}%")
        print()
        new_coverage = stats['any_source'] - stats['aws_nbest']
        print(f"New coverage (beyond AWS): {new_coverage} errors ({new_coverage/total*100:.1f}%)")
    else:
        print("No low-confidence errors found.")


if __name__ == "__main__":
    main()

