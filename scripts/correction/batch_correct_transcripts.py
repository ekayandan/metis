#!/usr/bin/env python3
"""Download AWS transcript JSON files and generate corrected hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.transcribe.aws_transcribe_batch import download_job_json  # noqa: E402
from scripts.correction.correct_transcript import (  # noqa: E402
    Token,
    correct_transcript,
    tokens_to_text,
)


def tokens_from_json(data: Dict) -> List[Token]:
    results = data.get("results", {})
    items = results.get("items", [])
    tokens: List[Token] = []
    for item in items:
        alternatives = item.get("alternatives", [])
        if not alternatives:
            continue
        top_alt = alternatives[0]
        content = top_alt.get("content", "")
        confidence_raw = top_alt.get("confidence")
        confidence = float(confidence_raw) if confidence_raw is not None else None
        is_punctuation = item.get("type") == "punctuation"
        alt_words = [alt.get("content", "") for alt in alternatives]
        tokens.append(
            Token(
                text=content,
                is_punctuation=is_punctuation,
                confidence=confidence,
                alternatives=alt_words,
            )
        )
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(".aws_transcribe_state.json"),
        help="Path to state file produced by aws_transcribe_batch.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aws_corrected.tsv"),
        help="Where to write corrected transcripts TSV",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("transcripts_json"),
        help="Directory to cache downloaded AWS transcript JSON files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.state.exists():
        raise SystemExit(f"State file not found: {args.state}")

    state = json.loads(args.state.read_text())
    job_to_file: Dict[str, str] = state.get("job_to_file", {})
    if not job_to_file:
        raise SystemExit("State file does not contain any jobs.")

    region = state.get("region") or "us-east-1"
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    transcribe = session.client("transcribe")

    jobs_meta: Dict[str, Dict] = state.get("jobs_meta", {})

    args.json_dir.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []

    for job_name, audio_path in sorted(job_to_file.items()):
        job = jobs_meta.get(job_name)
        if not job:
            try:
                job = transcribe.get_transcription_job(TranscriptionJobName=job_name)[
                    "TranscriptionJob"
                ]
            except ClientError as exc:
                print(f"Failed to retrieve job {job_name}: {exc}", file=sys.stderr)
                continue
        uri = job.get("Transcript", {}).get("TranscriptFileUri")
        if not uri:
            print(f"No transcript URI for job {job_name}", file=sys.stderr)
            continue
        try:
            data = download_job_json(s3, uri)
        except ClientError as exc:
            print(f"Failed to download transcript for {job_name}: {exc}", file=sys.stderr)
            continue
        json_path = args.json_dir / f"{job_name}.json"
        json_path.write_text(json.dumps(data))

        tokens = tokens_from_json(data)
        corrected_tokens = correct_transcript(tokens)
        text = tokens_to_text(corrected_tokens)
        rows.append(f"{audio_path}\t{text}")

    args.output.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"Wrote {len(rows)} corrected transcripts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
