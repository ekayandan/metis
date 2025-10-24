#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    print("Missing dependencies. Please install with: pip install -r requirements.txt", file=sys.stderr)
    raise


def read_manifest_tsv(path: str):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            audio = parts[0]
            text = parts[1] if len(parts) > 1 else ''
            rows.append((audio, text))
    return rows


def ensure_bucket(s3, bucket: str, region: str):
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = e.response['Error'].get('Code')
        if code in ('404', 'NoSuchBucket', 'NotFound'):
            pass
        else:
            raise
    params = {'Bucket': bucket}
    if region != 'us-east-1':
        params['CreateBucketConfiguration'] = {'LocationConstraint': region}
    s3.create_bucket(**params)


def s3_key_for(local_path: str, prefix: str):
    base = os.path.basename(local_path)
    return f"{prefix.rstrip('/')}/{base}"


def start_job(transcribe, job_name: str, media_uri: str, media_format: str, language_code: str, max_alternatives: Optional[int], output_bucket=None, output_key=None):
    settings = {
        'ShowAlternatives': True,
    }
    if max_alternatives and max_alternatives > 0:
        settings['MaxAlternatives'] = max(1, min(4, int(max_alternatives)))
    else:
        # AWS currently caps alternatives at 4; request the maximum when unlimited requested.
        settings['MaxAlternatives'] = 4
    params = {
        'TranscriptionJobName': job_name,
        'LanguageCode': language_code,
        'MediaFormat': media_format,
        'Media': {'MediaFileUri': media_uri},
        'Settings': settings,
    }
    if output_bucket:
        params['OutputBucketName'] = output_bucket
        if output_key:
            params['OutputKey'] = output_key.rstrip('/') + '/'
    return transcribe.start_transcription_job(**params)


def get_job(transcribe, name: str):
    return transcribe.get_transcription_job(TranscriptionJobName=name)['TranscriptionJob']


def wait_jobs(transcribe, names):
    pending = set(names)
    statuses = {}
    while pending:
        time.sleep(5)
        for n in list(pending):
            try:
                job = get_job(transcribe, n)
            except ClientError:
                continue
            status = job['TranscriptionJobStatus']
            if status in ('COMPLETED', 'FAILED'):
                statuses[n] = job
                pending.remove(n)
    return statuses


def download_job_json(s3_client, uri: str):
    parsed = urlparse(uri)
    bucket = None
    key = None
    if parsed.scheme == 's3':
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
    elif parsed.scheme in ('http', 'https'):
        host_parts = parsed.netloc.split('.')
        path = parsed.path.lstrip('/')
        if parsed.netloc.startswith('s3.'):
            if '/' not in path:
                raise ValueError(f"Cannot parse S3 key from URI: {uri}")
            bucket, key = path.split('/', 1)
        elif 'amazonaws.com' in parsed.netloc and host_parts[0] != 's3':
            bucket = host_parts[0]
            key = path
    if not bucket or not key:
        raise ValueError(f"Unsupported transcript URI: {uri}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj['Body'].read().decode('utf-8'))


def compose_nbest_from_segments(results: dict, limit: Optional[int]):
    segments = results.get('segments') or []
    if not segments:
        return []
    max_k = 0
    for seg in segments:
        max_k = max(max_k, len(seg.get('alternatives', [])))
    if max_k == 0:
        return []
    use_n = max_k if not limit or limit <= 0 else min(limit, max_k)
    outs = ['' for _ in range(use_n)]
    for seg in segments:
        alts = seg.get('alternatives', [])
        best = alts[0]['transcript'] if alts else ''
        for k in range(use_n):
            part = alts[k]['transcript'] if k < len(alts) else best
            outs[k] = (outs[k] + ' ' + part).strip()
    return outs


def extract_nbest(result_json: dict, limit: Optional[int]):
    res = result_json.get('results', {})
    if 'alternatives' in res and isinstance(res['alternatives'], list):
        alts = res['alternatives']
        if limit and limit > 0:
            alts = alts[:limit]
        out = []
        for a in alts:
            t = a.get('transcript') or ''
            out.append(t)
        if out:
            return out
    nb = compose_nbest_from_segments(res, limit)
    if nb:
        return nb
    t = ''
    ts = res.get('transcripts') or []
    if ts:
        t = ts[0].get('transcript', '')
    if limit and limit > 0:
        return [t] + [''] * (limit - 1)
    return [t]


def stable_job_name(basename: str, s3_key: str):
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '-' for ch in basename)[:150]
    h = hashlib.sha1(s3_key.encode('utf-8')).hexdigest()[:8]
    return f"{safe}-{h}"


def save_state(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def json_safe(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description='Batch AWS Transcribe with top-N alternatives for LibriSpeech sample.')
    ap.add_argument('--manifest', default='sample10min.tsv', help='TSV with <path>\t<text>')
    ap.add_argument('--bucket', required=True, help='S3 bucket for input (and output unless service-managed).')
    ap.add_argument('--prefix', default='transcribe-input', help='S3 prefix for uploads.')
    ap.add_argument('--region', default=os.environ.get('AWS_REGION', os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')))
    ap.add_argument('--language', default='en-US')
    ap.add_argument('--max-alt', type=int, default=0, help='Top-N alternatives to request (0 = all available).')
    ap.add_argument('--output-prefix', default='transcribe-output', help='S3 prefix for JSON results (optional)')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--result-tsv', default='aws_nbest.tsv', help='Where to write parsed N-best TSV')
    ap.add_argument('--phase', choices=['upload','start','wait','collect','all'], default='all', help='Run a single phase or all')
    ap.add_argument('--state', default='.aws_transcribe_state.json', help='Path to state file for resuming')
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    s3 = session.client('s3')
    sts = session.client('sts')
    transcribe = session.client('transcribe')

    who = sts.get_caller_identity()
    print(f"Using AWS account {who['Account']} as ARN {who['Arn']} in {args.region}")

    # Ensure bucket exists (creates if missing)
    ensure_bucket(s3, args.bucket, args.region)

    items = read_manifest_tsv(args.manifest)
    if not items:
        print(f"No rows in manifest {args.manifest}", file=sys.stderr)
        sys.exit(1)

    local_files = [p for p, _ in items]

    state = load_state(args.state)
    state.update({
        'bucket': args.bucket,
        'prefix': args.prefix,
        'output_prefix': args.output_prefix,
        'region': args.region,
        'language': args.language,
        'max_alt': args.max_alt,
    })

    if args.phase in ('upload','all'):
        if args.dry_run:
            print(f"[dry-run] Would upload {len(local_files)} files to s3://{args.bucket}/{args.prefix}")
        else:
            print(f"Uploading {len(local_files)} files to s3://{args.bucket}/{args.prefix} ...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = []
                for f in local_files:
                    futs.append(ex.submit(lambda path=f: s3.upload_file(path, args.bucket, s3_key_for(path, args.prefix))))
                for fut in concurrent.futures.as_completed(futs):
                    fut.result()
            print("Uploads complete.")

    # Start phase
    if args.phase in ('start','all'):
        job_to_file = state.get('job_to_file', {})
        for p, _ in items:
            key = s3_key_for(p, args.prefix)
            media_uri = f"s3://{args.bucket}/{key}"
            job_name = stable_job_name(os.path.splitext(os.path.basename(p))[0], key)
            out_prefix = args.output_prefix.rstrip('/') + '/' + job_name
            if args.dry_run:
                print(f"[dry-run] Would start job {job_name} for {media_uri}")
                job_to_file[job_name] = p
            else:
                try:
                    start_job(transcribe, job_name, media_uri, 'flac', args.language, args.max_alt, args.bucket, out_prefix)
                    job_to_file[job_name] = p
                except ClientError as e:
                    code = e.response['Error'].get('Code')
                    if code == 'ConflictException':
                        print(f"Job {job_name} already exists, using it...")
                        job_to_file[job_name] = p
                    else:
                        print(f"Failed to start job for {p}: {e}", file=sys.stderr)
        state['job_to_file'] = job_to_file
        save_state(args.state, state)

    if args.dry_run and args.phase != 'upload':
        print("[dry-run] Skipping wait/collect.")
        return

    # Wait phase
    if args.phase in ('wait','all'):
        job_to_file = state.get('job_to_file', {})
        if not job_to_file:
            print('No jobs in state; nothing to wait for.')
        else:
            print(f"Waiting for {len(job_to_file)} jobs to finish...")
            statuses = wait_jobs(transcribe, list(job_to_file.keys()))
            state['last_statuses'] = {k: v['TranscriptionJobStatus'] for k, v in statuses.items()}
            state['jobs_meta'] = json_safe({k: v for k, v in statuses.items()})
            save_state(args.state, state)

    # Collect phase
    if args.phase in ('collect','all'):
        job_to_file = state.get('job_to_file', {})
        if not job_to_file:
            print('No jobs in state; nothing to collect.')
            return
        # Fetch status for each job and collect completed ones
        out_lines = []
        for name, local_file in job_to_file.items():
            try:
                job = get_job(transcribe, name)
            except ClientError as e:
                print(f"Failed to get job {name}: {e}", file=sys.stderr)
                continue
            if job['TranscriptionJobStatus'] != 'COMPLETED':
                continue
            uri = job['Transcript']['TranscriptFileUri']
            try:
                data = download_job_json(s3, uri)
            except ClientError as e:
                print(f"Failed to download transcript for {name}: {e}", file=sys.stderr)
                continue
            except ValueError as e:
                print(f"{e}", file=sys.stderr)
                continue
            requested = args.max_alt if args.max_alt and args.max_alt > 0 else None
            nbest = extract_nbest(data, requested)
            cols = [local_file] + nbest
            out_lines.append('\t'.join(cols))
        with open(args.result_tsv, 'w') as w:
            for line in out_lines:
                w.write(line + '\n')
        print(f"Wrote {len(out_lines)} rows to {args.result_tsv}")


if __name__ == '__main__':
    main()
