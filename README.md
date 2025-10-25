# Low-Cost ASR Correction Pipeline

This project implements a lightweight post-correction system for automatic speech recognition (ASR) transcripts. It improves low-confidence words by generating diverse candidates from multiple sources (phonetic neighbors, normalizations, word-boundary edits) and selecting the best option using a fast encoder-only reranker.

## Overview

1. Start from a dataset in this repository. Each dataset ships with an audio file and its reference transcript so every run begins from the same source material.
2. Generate a transcript from your ASR provider. Use AWS Transcribe batch jobs or the bundled Faster-Whisper wrapper; both emit the AWS-style JSON that the correction step expects.
3. Feed the resulting JSON into the correction stage to revise low-confidence tokens:
   - **Candidate generation**: Augment AWS N-best with phonetic neighbors, normalizations (numbers/acronyms), word-boundary edits, and proper noun gazetteers
   - **Reranking**: Score candidates using DeBERTa-v3 pseudo-log-likelihood + ASR confidence + frequency priors
   - **Gating**: Only apply changes when the margin exceeds a threshold to prevent over-editing

## Datasets

Curated corpora live under `datasets/<name>/` with a consistent layout:

- `audio.flac`: source audio clip for the sample.
- `reference.tsv`: tab-separated file with a single row `audio.flac<TAB>reference text` used by evaluation utilities.
- The `*_10min` folders contain the full 10-minute clips; matching `*_mini` folders hold the 60-second high-WER excerpts used for quick iteration.

AWS reference transcriptions for these datasets are checked into `fixtures/aws/<name>/`:

- `transcribe_output.json`: canonical AWS Transcribe JSON response (with alternatives/confidences).
- `aws_nbest.tsv`: tab-separated N-best texts extracted from the JSON (column 1 = audio key, column 2+ = ranked hypotheses).

Derived artifacts (ASR hypotheses, corrections, metrics, etc.) should be written to `artifacts/` or another git-ignored location. The repository intentionally keeps datasets minimal so that anyone can regenerate downstream products from scratch.

## Features

- **Multi-source candidate generation**: Augments ASR N-best with 6 universal sources (phonetic neighbors, grapheme confusions, word-boundary edits, normalizations, proper noun gazetteer, character-LM speller)
- **Fast encoder-only reranker**: DeBERTa-v3-base scores candidates via pseudo-log-likelihood in context (replaces slower T5 generative model)
- **Margin-based gating**: Only applies changes when confidence margin exceeds threshold, preventing over-editing
- **Span detection**: Merges adjacent low-confidence tokens for joint correction
- **Safety rails**: Change rate caps, POS whitelisting, function word blacklisting
- **Domain-agnostic**: All candidate sources use general-purpose lexicons (wordfreq, Wikipedia/Wikidata) with no customer-specific tuning
- Works with AWS Transcribe JSON out of the box and accepts Faster-Whisper output as a drop-in replacement
- Word-level timestamps, confidences, and N-best alternatives captured through beam search when using the local Faster-Whisper path
- Voice-activity detection (VAD) filtering for cleaner segment boundaries in the Faster-Whisper wrapper

## Requirements

- Python 3.8+
- `torch`
- `transformers` (DeBERTa-v3-base for reranking)
- `g2p-en` (grapheme-to-phoneme for phonetic neighbors)
- `metaphone` (alternative phonetic encoding)
- `wordfreq` (frequency-based priors and lexicon)
- `boto3` (if you submit jobs to AWS Transcribe)
- Optional (local transcription): `faster-whisper`

## Installing dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional extras
pip install faster-whisper
```

## Generating transcripts with AWS Transcribe

Use the existing helper scripts to submit and collect batch jobs:

```bash
python scripts/transcribe/aws_transcribe_batch.py --manifest manifests/batch.tsv --region us-east-1
```

Download the resulting AWS JSON into `artifacts/` (or `samples/` for curated fixtures) before running the correction stage.

## Generating transcripts with Faster-Whisper

By convention, write intermediate Whisper outputs into the local `artifacts/` folder (git-ignored). The default CLI arguments already point there, so you can simply run:

```bash
python transcribe_audio.py datasets/cv_en_10min/audio.flac
python correct_transcript.py
```

The `transcribe_audio.py` script uses Faster-Whisper to emit AWS-compatible JSON:

```bash
python transcribe_audio.py datasets/cv_en_10min/audio.flac \
  --output transcribe_output.json \
  --model base \
  --beam-size 5 \
  --compute-type float16
```

Key options:

- `--beam-size` controls the number of beams used during decoding (set > 1 to gather alternatives).
- `--best-of` overrides how many hypotheses are examined per decoding step (defaults to `beam-size`).
- `--compute-type` chooses the CTranslate2 compute type (`float16`, `int8_float16`, etc.).
- `--no-vad` disables the default voice activity detector.
- `--language` can force a language code; otherwise Faster-Whisper performs automatic language detection.

The resulting `transcribe_output.json` mirrors AWS Transcribe field names so that downstream tooling continues to work unchanged.

## Correcting transcripts

### Architecture

The correction pipeline operates in three stages:

**1. Candidate Generation** (Target: 50-70% coverage)
- Start with AWS N-best alternatives (~10% coverage for low-confidence errors)
- Add phonetic neighbors using G2P + wordfreq lexicon (phoneme edit distance ≤ 2)
- Generate grapheme confusions via learned character substitution table
- Apply word-boundary edits (split/merge adjacent tokens)
- Normalize numbers, acronyms, contractions (e.g., "two hundred five" ↔ "205", "u s a" ↔ "USA")
- Search proper noun gazetteer (Wikipedia/Wikidata entities, Zipf ≥ 4)
- Character-LM speller for novel OOV forms (safety valve)
- Cap at K=8-12 candidates per span after deduplication

**2. Reranking** (Target: 70-85% precision)
- Score each candidate using DeBERTa-v3-base pseudo-log-likelihood in context
- Combine with ASR confidence, frequency prior, phonetic bonus, OOV penalty
- Formula: `Score(c) = λ₁·PLL + λ₂·log p_ASR + λ₃·FreqPrior + λ₄·PhoneticBonus - λ₅·OOVPenalty`

**3. Gating & Safety Rails**
- Only apply change if `(top - original) < γ` (margin threshold)
- Change rate cap: ≤ 1.5 changes per 100 tokens
- Whitelist POS: prioritize proper nouns, numbers, acronyms
- Blacklist: avoid changing "the", "and", "to" unless high margin

### Usage

1. Place your AWS- or Whisper-generated `transcribe_output.json` in the project directory (or point the CLI to another path).
2. Run the correction script to produce a refined transcript:

```bash
python correct_transcript.py --input artifacts/transcribe_output.json --output artifacts/corrected.txt --confidence-threshold 0.7
```

Key options:
- `--confidence-threshold`: Only correct tokens below this confidence (default: 0.7)
- `--trace-file`: Log all correction decisions for analysis
- `--enable-homophones`: Augment candidates with homophones
- `--disable-phonetic-filter`: Skip phonetic distance filtering

Batch workflows are supported via `python scripts/correction/batch_correct_transcripts.py --manifest manifests/batch.tsv --output-dir out/`.

To measure word error rate against the curated reference, run:

```bash
python scripts/evaluation/measure_wer.py datasets/cv_en_10min/reference.tsv path/to/hypotheses.tsv --hyp-column 2
```

For the bundled AWS baselines reuse `fixtures/aws/<name>/aws_nbest.tsv`, e.g.:

```bash
python scripts/evaluation/measure_wer.py datasets/cv_en_10min/reference.tsv fixtures/aws/cv_en_10min/aws_nbest.tsv --hyp-column 2
```
