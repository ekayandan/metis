"""Command line utility to run Faster-Whisper and emit AWS-compatible JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transcription import FasterWhisperTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--output",
        default="artifacts/whisper_transcribe_output.json",
        help="Destination path for the AWS-style JSON output.",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size or path (tiny, base, small, etc.).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference (cpu, cuda, or auto).",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type (float16, int8, int8_float16, etc.).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used during decoding.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Number of candidates evaluated per beam step. Defaults to beam size.",
    )
    parser.add_argument(
        "--job-name",
        default="faster-whisper-job",
        help="Job identifier stored in the output JSON.",
    )
    parser.add_argument(
        "--account-id",
        default="000000000000",
        help="Account identifier stored in the output JSON.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection filtering.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a specific language code (otherwise language detection is used).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper decoding task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcriber = FasterWhisperTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    result = transcriber.transcribe_to_aws(
        audio_path=args.audio,
        job_name=args.job_name,
        account_id=args.account_id,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=not args.no_vad,
        language=args.language,
        task=args.task,
    )

    output_path = Path(args.output)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote AWS-style transcript to {output_path}")


if __name__ == "__main__":
    main()
