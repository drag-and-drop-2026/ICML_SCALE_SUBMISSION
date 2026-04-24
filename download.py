"""Download data from s3://hai-drag-benchmark/v1/eval/ to a local directory."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_BUCKET = "hai-drag-benchmark"
DEFAULT_PREFIX = "v1/test/"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Local directory to download files into.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download threads.",
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="Use unsigned requests (for public buckets without credentials).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally with the same size.",
    )
    return parser.parse_args()


def make_client(anonymous: bool):
    if anonymous:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def list_objects(client, bucket: str, prefix: str) -> list[dict]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].endswith("/"):
                continue
            objects.append(obj)
    return objects


def local_path(output: Path, prefix: str, key: str) -> Path:
    rel = key[len(prefix) :] if key.startswith(prefix) else key
    return output / rel


def needs_download(dest: Path, size: int, overwrite: bool) -> bool:
    if overwrite or not dest.exists():
        return True
    return dest.stat().st_size != size


def download_one(client, bucket: str, key: str, dest: Path) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(dest)
    return key, dest.stat().st_size


def main():
    args = parse_args()
    prefix = args.prefix if args.prefix.endswith("/") or not args.prefix else args.prefix + "/"

    try:
        client = make_client(args.anonymous)
        objects = list_objects(client, args.bucket, prefix)
    except NoCredentialsError:
        print("No AWS credentials found. Re-run with --anonymous if the bucket is public.")
        raise
    except ClientError as e:
        print(f"Failed to list s3://{args.bucket}/{prefix}: {e}")
        raise

    if not objects:
        print(f"No objects found at s3://{args.bucket}/{prefix}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, int, Path]] = []
    skipped = 0
    for obj in objects:
        key = obj["Key"]
        size = obj["Size"]
        dest = local_path(args.output, prefix, key)
        if needs_download(dest, size, args.overwrite):
            pending.append((key, size, dest))
        else:
            skipped += 1

    total_bytes = sum(size for _, size, _ in pending)
    print(
        f"Found {len(objects)} objects ({skipped} up to date), "
        f"downloading {len(pending)} ({total_bytes / 1e6:.1f} MB) "
        f"to {args.output}"
    )

    if not pending:
        return

    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, client, args.bucket, key, dest): key
            for key, _, dest in pending
        }
        with tqdm(total=len(futures), unit="file", desc="download") as pbar:
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((key, repr(e)))
                pbar.update(1)

    if errors:
        print(f"\n{len(errors)} downloads failed:")
        for key, err in errors[:10]:
            print(f"  {key}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        raise SystemExit(1)

    print(f"Done. Downloaded {len(pending)} files to {args.output}")


if __name__ == "__main__":
    main()
