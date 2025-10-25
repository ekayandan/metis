#!/usr/bin/env python3
"""Quick debug script to see what candidates are being generated."""

import sys
from candidate_generation.generate_span import MultiSourceCandidateGenerator

# Initialize generator
print("Initializing generator...")
gen = MultiSourceCandidateGenerator()
print(f"Phonetic lexicon size: {len(gen.phonetic_gen.lexicon)}")
print(f"Gazetteer size: {len(gen.gazetteer.entries)}")
print()

# Test case 1: "Hitai" -> should find "hitachi"
test_cases = [
    ("Hitai", ["Hitachi", "Hitai"], "hitachi"),
    ("Choate", ["choate", "Choate"], "choate"),
    ("all", ["all", "awl"], "awol"),
    ("Lager", ["Lager", "lager"], "bootleggers"),
]

for hyp, aws_alts, ref in test_cases:
    print(f"\n{'='*80}")
    print(f"Testing: '{hyp}' -> '{ref}'")
    print(f"AWS alts: {aws_alts}")
    print(f"{'='*80}")
    
    result = gen.generate_candidates(hyp, aws_nbest=aws_alts, max_total=12)
    
    ref_lower = ref.lower().strip('.,!?;:\'"')
    
    print(f"\nRaw candidates per source:")
    for source, cands in result['raw'].items():
        print(f"  {source:20} ({len(cands):2d}): {cands[:5]}")  # First 5
        # Check if ref is in this source
        cands_lower = [c.lower().strip('.,!?;:\'"') for c in cands]
        if ref_lower in cands_lower:
            print(f"    ✅ FOUND '{ref}' in {source}")
    
    print(f"\nFinal candidates ({len(result['final'])}):")
    print(f"  {result['final']}")
    
    # Check if ref is in final
    final_lower = [c.lower().strip('.,!?;:\'"') for c in result['final']]
    if ref_lower in final_lower:
        print(f"  ✅ FOUND '{ref}' in final")
    else:
        print(f"  ❌ MISSING '{ref}' from final")

