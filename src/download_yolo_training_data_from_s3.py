"""
Download a YOLO dataset that was previously uploaded to S3
(by the prepare/upload script) onto the local machine (or a
fresh GPU training server) so it's ready for `model.train(...)`.

Downloads everything under S3_OUTPUT_PREFIX:
    images/train/*
    images/val/*
    labels/train/*
    labels/val/*
    data.yaml

into a local folder, preserving that same structure, and rewrites
data.yaml's "path:" to the absolute local dataset path.

Install once:
    pip install boto3 python-dotenv tqdm

.env file (same folder as this script) should contain:
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_REGION=ap-south-1
    AWS_BUCKET_NAME=piefly-visionx-twin-project
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =====================================================================
# CONFIG — edit this section for each dataset you want to pull down
# =====================================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")

# Must match S3_OUTPUT_PREFIX from the upload script
S3_DATASET_PREFIX = "model_training/object_detection/terrestrial_components/training_data/phase_01/"

# Where to put the dataset locally (will be created if missing)
LOCAL_DATASET_DIR = r"/home/ubuntu/yolo_model_training/yolo-object-detection-training-project/data/terrestrial_components/phase_1"

# How many parallel download threads to use
MAX_WORKERS = 16


# =====================================================================
# Helpers
# =====================================================================

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def list_all_keys(s3, bucket, prefix):
    """Return every object key under `prefix` (recursively)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):  # skip folder markers
                keys.append(key)
    return keys


def download_one(s3, bucket, key, local_root, prefix):
    """Download a single S3 key to its mirrored local path."""
    relative_path = key[len(prefix):]  # strip the S3 dataset prefix
    dest_path = Path(local_root) / relative_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest_path))
    return relative_path


def download_all(s3, bucket, prefix, local_root, max_workers):
    keys = list_all_keys(s3, bucket, prefix)
    if not keys:
        raise SystemExit(
            f"No objects found under s3://{bucket}/{prefix} — check S3_DATASET_PREFIX."
        )

    print(f"Found {len(keys)} object(s) under s3://{bucket}/{prefix}")
    print(f"Downloading to: {local_root}\n")

    Path(local_root).mkdir(parents=True, exist_ok=True)

    failures = []
    progress = tqdm(total=len(keys), unit="file") if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(download_one, s3, bucket, key, local_root, prefix): key
            for key in keys
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((key, str(exc)))
            finally:
                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    if failures:
        print(f"\nWARNING: {len(failures)} file(s) failed to download:")
        for key, err in failures[:20]:
            print(f"   {key} -> {err}")
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more")
    else:
        print("\nAll files downloaded successfully.")


def fix_data_yaml_path(local_root):
    """
    Rewrite the 'path:' line in data.yaml to the absolute local dataset
    directory, so YOLO resolves images/train, images/val etc. correctly
    on this machine.
    """
    yaml_path = Path(local_root) / "data.yaml"
    if not yaml_path.exists():
        print("\nNo data.yaml found — skipping path fix-up.")
        return

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    abs_root = str(Path(local_root).resolve()).replace("\\", "/")

    new_lines = []
    replaced = False
    for line in lines:
        if line.strip().startswith("path:"):
            new_lines.append(f"path: {abs_root}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.insert(0, f"path: {abs_root}")

    yaml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"\nUpdated data.yaml 'path:' -> {abs_root}")


# =====================================================================
# Main
# =====================================================================

def main():
    if not S3_BUCKET:
        raise SystemExit("AWS_BUCKET_NAME not found — check your .env file.")

    s3 = get_s3_client()
    prefix = normalize_prefix(S3_DATASET_PREFIX)

    download_all(s3, S3_BUCKET, prefix, LOCAL_DATASET_DIR, MAX_WORKERS)
    fix_data_yaml_path(LOCAL_DATASET_DIR)

    print("\nDataset ready for training at:")
    print(f"  {Path(LOCAL_DATASET_DIR).resolve()}")
    print(f"  -> data.yaml: {Path(LOCAL_DATASET_DIR).resolve() / 'data.yaml'}")


if __name__ == "__main__":
    main()