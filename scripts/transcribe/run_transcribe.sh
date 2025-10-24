#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_transcribe.sh PHASE BUCKET [REGION] [MAX_ALT]
# Phases: upload | start | wait | collect | all
# Loads .env.aws, ensures venv, installs deps, then runs the selected phase.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 PHASE BUCKET [REGION] [MAX_ALT]" >&2
  exit 1
fi

PHASE="$1"
BUCKET="$2"
REGION="${3:-${AWS_DEFAULT_REGION:-us-east-1}}"
MAX_ALT="${4:-0}"

if [[ -f .env.aws ]]; then
  set -a; source ./.env.aws; set +a
fi

python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/aws_transcribe_batch.py" \
  --phase "$PHASE" \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --max-alt "$MAX_ALT"

echo "Phase '$PHASE' done. If 'collect', results in aws_nbest.tsv"
