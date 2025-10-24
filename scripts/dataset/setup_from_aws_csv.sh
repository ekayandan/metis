#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/setup_from_aws_csv.sh ~/Downloads/credentials.csv [region]
# Creates or overwrites ./.env.aws from an AWS access keys CSV exported
# from the IAM console (columns: Access key ID, Secret access key, ...).

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/aws-credentials.csv [region]" >&2
  exit 1
fi

CSV_PATH="$1"
REGION="${2:-us-east-1}"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  exit 1
fi

# Try to extract header-aware or simple CSV (comma separated)
ACCESS_KEY_ID=""
SECRET_ACCESS_KEY=""

line=$(tail -n +2 "$CSV_PATH" | head -n 1)
# Remove UTF-8 BOM and CRs and quotes
line=$(printf '%s' "$line" | sed 's/^\xEF\xBB\xBF//' | tr -d '\r' | sed 's/"//g')
ACCESS_KEY_ID=$(printf '%s' "$line" | awk -F, '{print $1}')
SECRET_ACCESS_KEY=$(printf '%s' "$line" | awk -F, '{print $2}')

if [[ -z "$ACCESS_KEY_ID" || -z "$SECRET_ACCESS_KEY" ]]; then
  echo "Failed to parse credentials from $CSV_PATH" >&2
  exit 2
fi

cat > .env.aws <<EOF
AWS_ACCESS_KEY_ID=${ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${REGION}
# If your CSV contains a session token (temporary creds), add it here:
# AWS_SESSION_TOKEN=
EOF

echo ".env.aws written with region ${REGION}." >&2
