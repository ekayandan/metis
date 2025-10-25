# Repository Guidelines

## Project Structure & Module Organization
Curated corpora now live under `datasets/<name>/`, each containing only:
- `audio.flac`: the canonical audio clip.
- `reference.tsv`: a single-row TSV `audio.flac<TAB>reference transcript` for scoring runs.
- `*_mini` datasets are 60-second slices chosen from the highest-WER portions of their 10-minute parents for faster debugging passes.

Canonical AWS outputs ship under `fixtures/aws/<name>/`:
- `transcribe_output.json`: raw AWS JSON (with alternatives, confidences, timestamps).
- `aws_nbest.tsv`: N-best hypotheses extracted from the JSON.

The correction logic remains in `correct_transcript.py` (single JSON pipeline). Supporting utilities stay under `scripts/` grouped by stage (dataset, transcribe, correction, evaluation). Use `artifacts/` (git-ignored) for generated transcripts, WER dumps, or debug output. The `samples/` directory is reserved for lightweight fixtures such as `transcribe_output_dummy.json` that exercise parsing logic.

**Debugging note:** default to running analysis on the `*_mini` datasets (60-second slices) unless a task explicitly calls for the 10-minute sources. Use the full-length sets only for final validation or when longer context is required.

### Pipeline Steps & Scripts
- **Dataset prep**: `scripts/dataset/setup_from_aws_csv.sh` (bootstrap manifests from AWS metadata) and the curated sets in `datasets/`.
- **Transcribe submission (AWS)**: `scripts/transcribe/run_transcribe.sh` (shell wrapper) and `scripts/transcribe/aws_transcribe_batch.py` (queue batch jobs, manage buckets, download JSON).
- **Transcribe submission (local Whisper)**: `transcribe_audio.py` CLI backed by `transcription/` (Faster-Whisper, emits AWS-compatible JSON into `artifacts/`).
- **Inference & correction**: `correct_transcript.py` (single JSON correction) and `scripts/correction/batch_correct_transcripts.py` (manifest-driven multi-file run).
- **Evaluation**: `scripts/evaluation/measure_wer.py` (WER metrics) and `scripts/evaluation/validate_corrections.py` (ground truth validation) using `datasets/<name>/reference.tsv`.
- **Support utilities**: `samples/transcribe_output_dummy.json` for regression checks (kept small and deterministic).

## Build, Test, and Development Commands
Set up dependencies in a fresh virtual environment before touching models:
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`
Install correction extras for candidate generation:
- `pip install g2p-en metaphone wordfreq`
Install Whisper extras when you need the local transcription path:
- `pip install faster-whisper`

Run the single-file corrector with `python correct_transcript.py --input fixtures/aws/cv_en_10min/transcribe_output.json --output artifacts/corrected.txt --confidence-threshold 0.7` for real data. Batch processing uses `python scripts/correction/batch_correct_transcripts.py --manifest manifests/batch.tsv --output-dir out/`. 

Validate corrections against reference with `python scripts/evaluation/validate_corrections.py datasets/cv_en_10min/reference.tsv fixtures/aws/cv_en_10min/transcribe_output.json artifacts/corrected.tsv artifacts/trace_word.jsonl`.

Measure WER with `python scripts/evaluation/measure_wer.py datasets/<name>/reference.tsv hyp.tsv --hyp-column 1`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive, `snake_case` names for functions, variables, and files. Prefer dataclasses for structured records (see `Token` and `Segment` in `correct_transcript.py`). Type hints are expected for new public functions. Keep configuration clustered in the existing `CONFIG` dictionary and document non-obvious constants with a short comment.

## Testing Guidelines
There is no automated test harness yet; rely on deterministic scripts and sample assets. Add targeted unit tests under a future `tests/` package when expanding logic, and name files `test_<module>.py`. Before merging, run the correction pipeline on a known transcript plus `scripts/evaluation/validate_corrections.py` to confirm improvements. Record manual checks (inputs, thresholds, model revisions, coverage/precision metrics) in the PR description.

## Commit & Pull Request Guidelines
Current history (`initial`) is minimal; adopt concise, imperative commit subjects (e.g., `add phonetic candidate generator`) with optional bullet bodies describing rationale and verification. Group related changes to keep diffs reviewable. Pull requests should include: summary of changes, how to reproduce results, coverage/precision metrics (before/after), and WER improvements when quality shifts occur.

## Data & Credential Handling
Keep AWS credentials outside the repo and rely on environment profiles recognised by `boto3`. Store large audio artifacts in S3 or temporary paths, not in Git. When sharing manifests, redact personally identifiable information.

## Current Correction Architecture (Oct 25)

### Status Summary
- **Current WER**: AWS baseline 8.10% → Word-level correction 8.00% (**0.1% improvement**)
- **Bottleneck identified**: Only **10% of low-confidence errors** have the correct word in AWS N-best alternatives
- **LLM accuracy**: **100%** when correct answer is available (1/1 success: "Hitai" → "Hitachi")
- **Coverage problem**: 90% of corrections are "neutral" (replacing one wrong word with another wrong word from N-best)

### Architecture Shift: Multi-Source Candidates + Fast Reranker

**Previous approach (current baseline):**
- T5-based generative model selecting from AWS N-best only
- Limited to ~10% coverage for low-confidence errors
- Slow inference, conservative gating

**New approach (in development):**

#### 1. Candidate Generation (Target: 50-70% coverage)
- **A. Phonetic neighbors** ⭐: G2P + wordfreq lexicon (phoneme edit distance ≤ 2) → recovers "Hitai → Hitachi"
- **B. Grapheme confusions**: Learned character table {t↔ch, c↔k, s↔z, i↔y, ph↔f} + weighted Levenshtein
- **C. Word-boundary edits**: Split/merge logic (e.g., "micro soft" ↔ "microsoft", "Lager Cove" → "Bootleggers Cove")
- **D. Normalizations** ⭐⭐ High ROI: Numbers ("two hundred five" ↔ "205"), acronyms ("u s a" ↔ "USA"), contractions
- **E. Proper noun gazetteer**: Wikipedia/Wikidata entities (Zipf ≥ 4), still domain-agnostic
- **F. Character-LM speller**: Safety valve for novel OOV forms
- Cap at K=8-12 candidates per span after dedup

#### 2. Reranking (Target: 70-85% precision)
- **Model**: DeBERTa-v3-base (or MiniLM-L6 for edge) replacing T5
- **Scoring**: `Score(c) = λ₁·PLL_MLM + λ₂·log p_ASR + λ₃·FreqPrior + λ₄·PhoneticBonus - λ₅·OOVPenalty`
  - PLL_MLM: Pseudo-log-likelihood from DeBERTa (context fit)
  - p_ASR: AWS confidence if candidate is in N-best
  - FreqPrior: wordfreq Zipf score (prevent rare hallucinations)
  - PhoneticBonus: Reward tight phoneme matches
  - OOVPenalty: Penalize unseen strings
- **Gating**: Only apply if `(top - original) < γ` (margin threshold)

#### 3. Span Detection
- Merge tokens with conf < 0.7 where gaps ≤ 1 token
- Build confusion network from segment alternatives
- Joint correction of multi-token errors

#### 4. Safety Rails
- Change rate cap: ≤ 1.5 changes per 100 tokens
- Whitelist POS: prioritize proper nouns, numbers, acronyms
- Blacklist: avoid changing "the", "and", "to" unless high margin
- No-change band: if `(top - second) < δ`, keep original

### Implementation Priority

**Week 1: Quick Wins (Normalizations + Phonetics)**
1. ✅ Normalization (D): Numbers, acronyms, contractions
2. ✅ Phonetic neighbors (A): G2P + wordfreq
3. ✅ Word-boundary (C): Split/merge logic
**Expected**: Coverage 30-40%, Precision 50-60%

**Week 2: Reranker**
4. ⏳ Replace T5 with DeBERTa-v3-base PLL scorer
5. ⏳ Implement combined scoring function
6. ⏳ Add gating/margin logic
**Expected**: Precision 70-80%

**Week 3: Polish**
7. ⏳ Grapheme confusions (B)
8. ⏳ Proper noun gazetteer (E)
9. ⏳ Safety rails
10. ⏳ Span detection
**Expected**: Coverage 50-70%, Precision 70-85%, WER improvement 0.5-1.0%

### Validation Tools

**Ground truth validation script**: `scripts/evaluation/validate_corrections.py`
```bash
python scripts/evaluation/validate_corrections.py \
  datasets/cv_en_10min/reference.tsv \
  fixtures/aws/cv_en_10min/transcribe_output.json \
  artifacts/corrected.tsv \
  artifacts/trace_word.jsonl
```

Reports:
- ✅ **Correct fixes**: Changed wrong word to correct reference word
- ❌ **Wrong changes**: Changed correct word to wrong word (over-editing)
- ⚪ **Neutral**: Changed wrong word to different wrong word (both not in reference)
- ⚠️ **Unknown**: Could not align to reference

### Current Metrics (Word-Level, Threshold 0.7)

| Metric | Current | Target |
|--------|---------|--------|
| **Coverage@K** | 10% (1/10) | 50-70% |
| **Precision@1** | 10% (1/10) | 70-85% |
| **Over-edit rate** | 0% (0/10) | <0.3% |
| **WER improvement** | -0.1% | -0.5% to -1.0% |

**Breakdown of 10 corrections:**
- ✅ Correct: 1 (Hitai → Hitachi)
- ❌ Wrong: 0
- ⚪ Neutral: 8 (correct answer not in N-best)
- ⚠️ Unknown: 1

### Key Findings

1. **All 9 aligned low-confidence tokens were genuinely wrong** - threshold 0.7 correctly identifies errors
2. **Only 1 out of 10 had correct answer in AWS N-best** - this is the bottleneck
3. **LLM chose correctly when answer was available** - 100% accuracy (1/1)
4. **Multi-source candidates should increase coverage from 10% to 50-70%** - this is the path forward

### Next Agent Handoff

**Current state:**
- Word-level correction with T5 achieves 0.1% WER improvement
- Validation infrastructure in place (`validate_corrections.py`)
- Architecture documented for multi-source candidates + DeBERTa reranker

**Next steps:**
1. Implement phonetic neighbor generator (G2P + wordfreq)
2. Implement normalization module (numbers, acronyms, contractions)
3. Implement word-boundary editor (split/merge)
4. Test coverage improvement on cv_en_10min dataset
5. Replace T5 with DeBERTa-v3-base PLL scorer
6. Measure new coverage/precision metrics

**Dependencies to add:**
```bash
pip install g2p-en metaphone wordfreq
```

**Artifacts to track:**
- Coverage@K per candidate source (phonetic, normalization, etc.)
- Precision@1 with DeBERTa reranker
- WER improvement on cv_en_10min (target: 0.5-1.0%)
- Over-edit rate (target: <0.3%)

Local generated files should live under `artifacts/` (ignored by git); both `transcribe_audio.py` and `correct_transcript.py` default to writing there and create the folder automatically. Only commit fixtures under `samples/` when they exercise parsing logic and cannot be regenerated quickly.
