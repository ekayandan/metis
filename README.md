# Low-Cost ASR Correction Pipeline

This project implements a lightweight post-correction system for automatic speech recognition (ASR) transcripts. It improves low-confidence words by selecting better alternatives using a small span-infilling language model.

## Overview

1. Generate a transcript with the bundled Faster-Whisper wrapper. The wrapper exposes word-level timestamps, confidence scores, and alternative hypotheses while matching the AWS Transcribe JSON schema expected by the correction pipeline.
2. Feed the resulting JSON into the correction step to revise low-confidence tokens with a local language model (e.g., Flan-T5-Base).

## Features

- Drop-in replacement for AWS Transcribe output.
- Word-level timestamps, confidences, and N-best alternatives captured through beam search.
- Voice-activity detection (VAD) filtering for cleaner segment boundaries.
- Confidence threshold and context size are configurable in the correction stage.
- The correction model only replaces words when the chosen correction is in the alternative list.

## Requirements

- Python 3.8+
- `faster-whisper`
- `transformers`
- `torch`
- `sentencepiece` (for T5 models)

## Installing dependencies

```bash
pip install faster-whisper transformers torch sentencepiece
```

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

1. Place your `transcribe_output.json` in the project directory.
2. Run the correction script (not included here) to produce a refined transcript:

```bash
python correct_transcript.py
```
