# Repository Guidelines

## Project Structure & Module Organization
Curated corpora now live under `datasets/<name>/`, each containing only:
- `audio.flac`: the canonical audio clip.
- `reference.tsv`: a single-row TSV `audio.flac<TAB>reference transcript` for scoring runs.

The correction logic remains in `correct_transcript.py` (single JSON pipeline). Supporting utilities stay under `scripts/` grouped by stage (dataset, transcribe, correction, evaluation). Use `artifacts/` (git-ignored) for generated transcripts, WER dumps, or debug output. The `samples/` directory is reserved for lightweight fixtures such as `transcribe_output_dummy.json` that exercise parsing logic.

### Pipeline Steps & Scripts
- **Dataset prep**: `scripts/dataset/setup_from_aws_csv.sh` (bootstrap manifests from AWS metadata) and the curated sets in `datasets/`.
- **Transcribe submission (AWS)**: `scripts/transcribe/run_transcribe.sh` (shell wrapper) and `scripts/transcribe/aws_transcribe_batch.py` (queue batch jobs, manage buckets, download JSON).
- **Transcribe submission (local Whisper)**: `transcribe_audio.py` CLI backed by `transcription/` (Faster-Whisper, emits AWS-compatible JSON into `artifacts/`).
- **Inference & correction**: `correct_transcript.py` (single JSON correction) and `scripts/correction/batch_correct_transcripts.py` (manifest-driven multi-file run).
- **Evaluation**: `scripts/evaluation/measure_wer.py` (WER metrics) using `datasets/<name>/reference.tsv` as ground truth. Store hypotheses/metrics alongside run artefacts in `artifacts/`.
- **Support utilities**: `samples/transcribe_output_dummy.json` for regression checks (kept small and deterministic).

## Build, Test, and Development Commands
Set up dependencies in a fresh virtual environment before touching models:
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`
Install Whisper extras when you need the local transcription path:
- `pip install faster-whisper pronouncing g2p_en`
Run the single-file corrector with `python correct_transcript.py --input samples/transcribe_output_dummy.json --output corrected.txt` for smoke tests. For real data, transcribe `datasets/cv_en_10min/audio.flac` (or another dataset), then feed the resulting JSON into `correct_transcript.py`. Batch processing uses `python scripts/correction/batch_correct_transcripts.py --manifest manifests/batch.tsv --output-dir out/`. Validate recognition quality with `python scripts/evaluation/measure_wer.py datasets/<name>/reference.tsv hyp.tsv --hyp-column 2`. Regenerate AWS jobs via `python scripts/transcribe/aws_transcribe_batch.py --manifest manifests/batch.tsv --region us-east-1`.

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
- Installed `faster-whisper`, `pronouncing`, `g2p_en` inside `venv`; ran `python transcribe_audio.py datasets/cv_en_10min/audio.flac --model base --beam-size 5 --compute-type int8`.
- Whisper JSON saved to `artifacts/whisper/cv_en_10min_transcribe_output.json`; raw transcript WER vs. the curated reference (`datasets/cv_en_10min/reference.tsv`) is **16.40 %** (117 S / 16 D / 29 I).

## Correction Pipeline Tweaks
- Added CLI wrapper + adapter to emit AWS-compatible items with alternatives/confidences per token.
- Disabled homophone expansion by default (`CONFIG["ENABLE_HOMOPHONE_EXPANSION"] = False`).
- Added phonetic filtering (`filter_by_phonetic_distance`) to strip ASR alternatives that are acoustically distant; currently allow only distance 0.
- Raised correction confidence threshold to **0.97**; only very low-confidence words are eligible.
- Introduced `is_degenerate_choice` to reject replacements that repeat neighbours, split tokens, or are equivalent to the original.
- `apply_corrections` now returns a change log; CLI prints the few accepted replacements for audit.

## Current Outputs
- Latest correction run (`python correct_transcript.py --input artifacts/whisper/cv_en_10min_transcribe_output.json --output artifacts/whisper/cv_en_10min_corrected.txt`) rewrote only four tokens:
  - `Lin → Lynn` (conf 0.36)
  - `color → colour` (conf 0.37)
  - `five → 5` (conf 0.55)
  - `four → 4` (conf 0.89)
- Corrected transcript stored at `artifacts/whisper/cv_en_10min_corrected.txt` with TSV mirror `artifacts/whisper/cv_en_10min_corrected.tsv` (regenerate locally as needed; not tracked in git).
- WER after corrections: **16.80 %** (121 S / 16 D / 29 I) — slightly above raw whisper baseline (16.40 %), but without catastrophic regressions.

## Remaining Issues / Next Steps
1. Decide whether numeric normalization (`five → 5`) is desirable; if not, add rule to preserve word form when surrounding context uses words.
2. Still seeing phrases like “gate skill” from Whisper itself; evaluate whether we want additional language model passes or leave to future iteration.
3. Consider re-enabling phonetic distance 1 for certain tokens (e.g. `Colouridge → Coleridge`) while keeping degenerate guardrails.
4. For cloud follow-up: replicate environment (Python 3.13 venv, `pip install -r requirements.txt` + extras above) and rerun WER checks before expanding to larger manifests.

Artifacts to sync (store under `artifacts/`, do not commit):
- Whisper JSON/TSV pairs for recent runs (e.g. `artifacts/whisper/cv_en_10min_transcribe_output.json`).
- Corrected transcripts and audit logs (e.g. `artifacts/whisper/cv_en_10min_corrected.txt`).
- Updated source files under `transcription/` and `correct_transcript.py` already live in git.

Local generated files should live under `artifacts/` (ignored by git); both `transcribe_audio.py` and `correct_transcript.py` default to writing there and create the folder automatically. Only commit fixtures under `samples/` when they exercise parsing logic and cannot be regenerated quickly.
