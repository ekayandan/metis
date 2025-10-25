#!/usr/bin/env python3
"""Evaluate Coverage@K for multi-source candidate generation.

Measures how often the correct reference word appears in generated candidates
from each source and tracks latency metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.measure_wer import load_transcripts, alignment_details
from candidate_generation.generate_span import MultiSourceCandidateGenerator


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


def evaluate_coverage(
    reference_path: Path,
    transcribe_json_path: Path,
    confidence_threshold: float = 0.7,
    output_json: Optional[Path] = None,
) -> Dict:
    """Evaluate coverage from all candidate sources.
    
    Args:
        reference_path: Path to reference TSV
        transcribe_json_path: Path to AWS Transcribe JSON
        confidence_threshold: Confidence threshold for low-confidence errors
        output_json: Optional path to save detailed results
    
    Returns:
        Dict with coverage metrics
    """
    
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
    
    # Initialize multi-source generator
    print("Initializing multi-source candidate generator (this may take 1-2 minutes)...")
    init_start = time.time()
    generator = MultiSourceCandidateGenerator()
    init_time = time.time() - init_start
    print(f"Initialized in {init_time:.1f}s")
    print()
    
    # Evaluate coverage
    coverage_stats = {
        'total': len(low_conf_errors),
        'aws_nbest': 0,
        'phonetic': 0,
        'normalization': 0,
        'word_boundary': 0,
        'gazetteer': 0,
        'grapheme': 0,
        'char_speller': 0,
        'any_source': 0,
        'new_coverage': 0,  # Beyond AWS N-best
    }
    
    latencies = []
    candidate_counts = []
    detailed_results = []
    
    for error in low_conf_errors:
        hyp = error['hyp_word']
        ref = error['ref_word']
        conf = error['confidence']
        token_idx = error['token_index']
        
        ref_clean = ref.lower().strip('.,!?;:\'"')
        
        # Get AWS alternatives
        aws_alts = load_aws_alternatives(transcribe_json_path, token_idx)
        
        # Generate candidates
        gen_start = time.time()
        result = generator.generate_candidates(hyp, aws_nbest=aws_alts, max_total=12)
        gen_time = (time.time() - gen_start) * 1000  # Convert to ms
        latencies.append(gen_time)
        
        final_candidates = result['final']
        candidate_counts.append(len(final_candidates))
        
        # Check coverage per source
        found_in = []
        
        for source_name, source_cands in result['raw'].items():
            source_cands_clean = [c.lower().strip('.,!?;:\'"') for c in source_cands]
            if ref_clean in source_cands_clean:
                coverage_stats[source_name] += 1
                found_in.append(source_name)
        
        # Check if in final candidates
        final_clean = [c.lower().strip('.,!?;:\'"') for c in final_candidates]
        in_final = ref_clean in final_clean
        
        # Check if in any source
        in_any = len(found_in) > 0
        if in_any:
            coverage_stats['any_source'] += 1
            
            # Check if new (not in AWS)
            aws_clean = [a.lower().strip('.,!?;:\'"') for a in aws_alts]
            if ref_clean not in aws_clean:
                coverage_stats['new_coverage'] += 1
        
        # Store detailed result
        detailed_results.append({
            'hyp': hyp,
            'ref': ref,
            'confidence': conf,
            'found_in_sources': found_in,
            'in_final': in_final,
            'num_candidates': len(final_candidates),
            'latency_ms': gen_time,
        })
    
    # Calculate metrics
    total = coverage_stats['total']
    metrics = {
        'coverage': {},
        'latency': {},
        'candidates': {},
    }
    
    if total > 0:
        for source in ['aws_nbest', 'phonetic', 'normalization', 'word_boundary', 
                       'gazetteer', 'grapheme', 'char_speller', 'any_source', 'new_coverage']:
            metrics['coverage'][source] = {
                'count': coverage_stats[source],
                'percentage': coverage_stats[source] / total * 100,
            }
    
    if latencies:
        latencies.sort()
        metrics['latency'] = {
            'p50_ms': latencies[len(latencies) // 2],
            'p95_ms': latencies[int(len(latencies) * 0.95)],
            'mean_ms': sum(latencies) / len(latencies),
        }
    
    if candidate_counts:
        metrics['candidates'] = {
            'mean': sum(candidate_counts) / len(candidate_counts),
            'max': max(candidate_counts),
            'min': min(candidate_counts),
        }
    
    # Save detailed results if requested
    if output_json:
        output_data = {
            'metrics': metrics,
            'coverage_stats': coverage_stats,
            'detailed_results': detailed_results,
        }
        output_json.write_text(json.dumps(output_data, indent=2))
        print(f"Detailed results saved to {output_json}")
        print()
    
    return metrics, detailed_results


def print_metrics(metrics: Dict, detailed_results: List[Dict]):
    """Print coverage metrics in a readable format."""
    print("="*80)
    print("COVERAGE@K METRICS")
    print("="*80)
    print()
    
    cov = metrics['coverage']
    total = sum(1 for _ in detailed_results)
    
    print(f"Total low-confidence errors: {total}")
    print()
    
    print("Coverage by source:")
    for source in ['aws_nbest', 'phonetic', 'normalization', 'word_boundary', 
                   'gazetteer', 'grapheme', 'char_speller']:
        if source in cov:
            count = cov[source]['count']
            pct = cov[source]['percentage']
            print(f"  {source:20} {count:3d} / {total} = {pct:5.1f}%")
    
    print()
    print(f"  {'ANY SOURCE':20} {cov['any_source']['count']:3d} / {total} = {cov['any_source']['percentage']:5.1f}%")
    print(f"  {'NEW (beyond AWS)':20} {cov['new_coverage']['count']:3d} / {total} = {cov['new_coverage']['percentage']:5.1f}%")
    
    print()
    print("="*80)
    print("LATENCY METRICS")
    print("="*80)
    lat = metrics['latency']
    print(f"  p50: {lat['p50_ms']:.2f} ms")
    print(f"  p95: {lat['p95_ms']:.2f} ms")
    print(f"  mean: {lat['mean_ms']:.2f} ms")
    
    print()
    print("="*80)
    print("CANDIDATE SLATE SIZE")
    print("="*80)
    cand = metrics['candidates']
    print(f"  mean: {cand['mean']:.1f}")
    print(f"  max: {cand['max']}")
    print(f"  min: {cand['min']}")
    
    print()
    print("="*80)
    print("TOP EXAMPLES")
    print("="*80)
    
    # Show newly covered examples
    newly_covered = [r for r in detailed_results if r['in_final'] and 'aws_nbest' not in r['found_in_sources']]
    if newly_covered:
        print(f"\nNewly covered (not in AWS N-best): {len(newly_covered)}")
        for r in newly_covered[:5]:
            print(f"  '{r['hyp']}' -> '{r['ref']}' (conf={r['confidence']:.3f})")
            print(f"    Found in: {r['found_in_sources']}")
    
    # Show still missing
    still_missing = [r for r in detailed_results if not r['in_final']]
    if still_missing:
        print(f"\nStill missing: {len(still_missing)}")
        for r in still_missing[:5]:
            print(f"  '{r['hyp']}' -> '{r['ref']}' (conf={r['confidence']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate generation coverage")
    parser.add_argument("reference", type=Path, help="Reference TSV file")
    parser.add_argument("transcribe_json", type=Path, help="AWS Transcribe JSON")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence threshold for low-confidence errors")
    parser.add_argument("--output-json", type=Path, help="Save detailed results to JSON")
    
    args = parser.parse_args()
    
    metrics, detailed_results = evaluate_coverage(
        args.reference,
        args.transcribe_json,
        args.confidence_threshold,
        args.output_json,
    )
    
    print_metrics(metrics, detailed_results)


if __name__ == "__main__":
    main()

