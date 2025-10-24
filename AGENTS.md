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
