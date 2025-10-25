# Low-Cost ASR Correction Pipeline

This project implements a lightweight post-correction system for automatic speech recognition (ASR) transcripts. It improves low-confidence words by selecting better alternatives using a small span-infilling language model.

## Overview

1. Generate a transcript from your ASR provider. Use AWS Transcribe batch jobs or the bundled Faster-Whisper wrapper; both emit the AWS-style JSON that the correction step expects.
2. Feed the resulting JSON into the correction stage to revise low-confidence tokens with a local span-infilling model (e.g., Flan-T5-Base).

## Features

- Works with AWS Transcribe JSON out of the box and accepts Faster-Whisper output as a drop-in replacement.
- Word-level timestamps, confidences, and N-best alternatives captured through beam search when using the local Faster-Whisper path.
- Voice-activity detection (VAD) filtering for cleaner segment boundaries in the Faster-Whisper wrapper.
- Confidence threshold and context size are configurable in the correction stage.
- The correction model only replaces words when the chosen correction is in the alternative list.

## Requirements

- Python 3.8+
- `torch`
- `transformers`
- `sentencepiece` (for T5 models)
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
python transcribe_audio.py path/to/audio.wav
python correct_transcript.py
```

The `transcribe_audio.py` script uses Faster-Whisper to emit AWS-compatible JSON:

```bash
python transcribe_audio.py path/to/audio.wav \
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

1. Place your AWS- or Whisper-generated `transcribe_output.json` in the project directory (or point the CLI to another path).
2. Run the correction script to produce a refined transcript:

```bash
python correct_transcript.py --input artifacts/transcribe_output.json --output artifacts/corrected.txt
```

Batch workflows are supported via `python scripts/correction/batch_correct_transcripts.py --manifest manifests/batch.tsv --output-dir out/`.
