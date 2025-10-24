# Low-Cost ASR Correction Pipeline

This project implements a lightweight post-correction system for automatic speech recognition (ASR) transcripts. It improves low-confidence words by selecting better alternatives using a small span-infilling language model.

## Overview

Given an AWS Transcribe JSON output:
- Low-confidence tokens are detected.
- Top N alternatives for each token are extracted.
- A local language model (e.g. Flan-T5-Base) selects the best replacement using surrounding context.
- Final corrected transcript is generated.

## Features
- Supports AWS Transcribe JSON format.
- Confidence threshold and context size are configurable.
- Model only replaces words when the chosen correction is in the alternative list.
- All parameters configurable in one section of the script.

## Requirements
- Python 3.8+
- `transformers` (Hugging Face)
- `torch`
- `sentencepiece` (for T5 models)

## Quick Start

1. Place your `transcribe_output.json` in the project directory.
2. Run:

```bash
python correct_transcript.py

