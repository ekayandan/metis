# Repository Guidelines

## Project Structure & Module Organization
The core post-correction logic lives in `correct_transcript.py`, and should stay focused on the single-file pipeline that loads AWS Transcribe JSON, infers corrections, and writes text output. Supporting utilities reside under `scripts/` by pipeline stage (dataset, transcribe, correction, evaluation). Keep sample assets such as `tedlium_10min_sample/` lightweight, stash canonical fixtures in `samples/` (see `transcribe_output_dummy.json`), and reserve top-level additions for shared configuration (`requirements.txt`, future docs).

### Pipeline Steps & Scripts
- **Dataset prep**: `scripts/dataset/setup_from_aws_csv.sh` (bootstrap manifests from AWS metadata).
- **Transcribe submission**: `scripts/transcribe/run_transcribe.sh` (shell wrapper) and `scripts/transcribe/aws_transcribe_batch.py` (queue batch jobs, manage buckets, download JSON).
- **Inference & correction**: `correct_transcript.py` (single JSON correction) and `scripts/correction/batch_correct_transcripts.py` (manifest-driven multi-file run).
- **Evaluation**: `scripts/evaluation/measure_wer.py` (WER metrics) and `aws_nbest.tsv`/`sample10min.tsv` outputs for manual review.
- **Support utilities**: `combined_10min-7756ce2a.json` as archived baseline, plus `samples/transcribe_output_dummy.json` for regression checks.

## Build, Test, and Development Commands
Set up dependencies in a fresh virtual environment before touching models:
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`
Run the single-file corrector with `python correct_transcript.py --input samples/transcribe_output_dummy.json --output corrected.txt`. Batch processing uses `python scripts/correction/batch_correct_transcripts.py --manifest manifests/batch.tsv --output-dir out/`. Validate recognition quality with `python scripts/evaluation/measure_wer.py ref.tsv hyp.tsv --hyp-column 2`. Regenerate AWS jobs via `python scripts/transcribe/aws_transcribe_batch.py --manifest manifests/batch.tsv --region us-east-1`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive, `snake_case` names for functions, variables, and files. Prefer dataclasses for structured records (see `Token` in `correct_transcript.py`). Type hints are expected for new public functions. Keep configuration clustered in the existing `CONFIG` dictionary and document non-obvious constants with a short comment.

## Testing Guidelines
There is no automated test harness yet; rely on deterministic scripts and sample assets. Add targeted unit tests under a future `tests/` package when expanding logic, and name files `test_<module>.py`. Before merging, run the correction pipeline on a known transcript plus `scripts/evaluation/measure_wer.py` to confirm WER improvements. Record manual checks (inputs, thresholds, model revisions) in the PR description.

## Commit & Pull Request Guidelines
Current history (`initial`) is minimal; adopt concise, imperative commit subjects (e.g., `add wer helper`) with optional bullet bodies describing rationale and verification. Group related changes to keep diffs reviewable. Pull requests should include: summary of changes, how to reproduce results, links to AWS job manifests if applicable, and screenshots or metrics (WER before/after) when UI or quality shifts occur.

## Data & Credential Handling
Keep AWS credentials outside the repo and rely on environment profiles recognised by `boto3`. Store large audio artifacts in S3 or temporary paths, not in Git. When sharing manifests, redact personally identifiable information and confirm that sample audio stays within the provided `tedlium_10min_sample/` folder.

## Whisper Integration Status (Oct 24)
- **Agent Goal**: Deploy a local Faster-Whisper front end that feeds the existing correction pipeline, keep WER at or below the AWS baseline, and surface safe corrections with clear audit trails while we iterate toward production.
- **Definition of Done**: Faster-Whisper transcripts slot into the correction/evaluation scripts without manual tweaks; correction runs produce WER on par with or better than the raw ASR on the 10‑minute sample; AGENTS.md documents setup, results, and outstanding issues for the next agent.
- Checked out `whisper` branch and added a Faster-Whisper front end (`transcribe_audio.py`, `transcription/` package).
- Patched `README.md` with install/run instructions for the new local transcription path.
- Installed `faster-whisper`, `pronouncing`, `g2p_en` inside `venv`; ran `transcribe_audio.py samples/cv_en_10min/cv_en_10min.flac --model base --beam-size 5 --compute-type int8`.
- Whisper JSON saved to `samples/cv_en_10min/whisper_transcribe_output.json`; raw transcript WER vs. reference (`samples/cv_en_10min/cv_en_10min_manifest.tsv`) is **16.40 %** (117 S / 16 D / 29 I).

## Correction Pipeline Tweaks
- Added CLI wrapper + adapter to emit AWS-compatible items with alternatives/confidences per token.
- Disabled homophone expansion by default (`CONFIG["ENABLE_HOMOPHONE_EXPANSION"] = False`).
- Added phonetic filtering (`filter_by_phonetic_distance`) to strip ASR alternatives that are acoustically distant; currently allow only distance 0.
- Raised correction confidence threshold to **0.97**; only very low-confidence words are eligible.
- Introduced `is_degenerate_choice` to reject replacements that repeat neighbours, split tokens, or are equivalent to the original.
- `apply_corrections` now returns a change log; CLI prints the few accepted replacements for audit.

## Current Outputs
- Latest correction run (`correct_transcript.py --input samples/cv_en_10min/whisper_transcribe_output.json`) rewrites only four tokens:
  - `Lin → Lynn` (conf 0.36)
  - `color → colour` (conf 0.37)
  - `five → 5` (conf 0.55)
  - `four → 4` (conf 0.89)
- Corrected transcript stored at `samples/cv_en_10min/whisper_corrected.txt` with TSV mirror `samples/cv_en_10min/whisper_corrected.tsv`.
- WER after corrections: **16.80 %** (121 S / 16 D / 29 I) — slightly above raw whisper baseline (16.40 %), but without catastrophic regressions.

## Remaining Issues / Next Steps
1. Decide whether numeric normalization (`five → 5`) is desirable; if not, add rule to preserve word form when surrounding context uses words.
2. Still seeing phrases like “gate skill” from Whisper itself; evaluate whether we want additional language model passes or leave to future iteration.
3. Consider re-enabling phonetic distance 1 for certain tokens (e.g. `Colouridge → Coleridge`) while keeping degenerate guardrails.
4. For cloud follow-up: replicate environment (Python 3.13 venv, `pip install -r requirements.txt` + extras above) and rerun WER checks before expanding to larger manifests.

Artifacts to sync:
- `samples/cv_en_10min/whisper_transcribe_output.json`
- `samples/cv_en_10min/whisper_corrected.txt`
- `samples/cv_en_10min/whisper_corrected.tsv`
- Updated source files under `transcription/` and `correct_transcript.py`

Local generated files should now live under `artifacts/` (ignored by git); both `transcribe_audio.py` and `correct_transcript.py` default to writing there and create the folder automatically. Keep committing curated samples under `samples/` only when they are canonical fixtures.
