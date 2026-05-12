#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


PENDING_VALUES = {"", "PENDING", "TBD", "NA", "N/A"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figshare_api_download_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc in {"figshare.com", "ndownloader.figshare.com"} and len(parts) >= 3:
        if parts[-2] == "files" and parts[-1].isdigit():
            return f"https://api.figshare.com/v2/file/download/{parts[-1]}"
    return None


def _open_url(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "SMOACS weight downloader"})
    return urllib.request.urlopen(request)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    api_url = _figshare_api_download_url(url)
    open_url = api_url or url
    with _open_url(open_url) as response, path.open("wb") as out:
        if response.status == 202 or response.headers.get("x-amzn-waf-action") == "challenge":
            raise RuntimeError(
                f"download endpoint returned {response.status} "
                f"({response.headers.get('x-amzn-waf-action')}); try Figshare API URL"
            )
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file from {open_url}")


def iter_rows(manifest: Path):
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SMOACS model weights from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/model_weights_manifest.tsv"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--only", nargs="*", help="Optional artifact_id allowlist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    only = set(args.only or [])
    failures = 0

    for row in iter_rows(args.manifest):
        artifact_id = row["artifact_id"]
        if only and artifact_id not in only:
            continue

        url = row["source_url"]
        default_path = row["default_path"]
        expected = row["expected_sha256"].strip()

        if not url.startswith(("http://", "https://")):
            print(f"skip manual artifact: {artifact_id} ({url})")
            continue
        if default_path == "external":
            print(f"skip external artifact: {artifact_id}")
            continue

        path = root / default_path
        if args.dry_run:
            print(f"would download {artifact_id}: {url} -> {path}")
            continue

        if not path.exists():
            print(f"downloading {artifact_id}: {url} -> {path}")
            try:
                download(url, path)
            except Exception as exc:
                failures += 1
                print(f"ERROR {artifact_id}: download failed: {exc}", file=sys.stderr)
                continue
        else:
            print(f"using existing {artifact_id}: {path}")

        actual = sha256_file(path)
        print(f"{artifact_id} sha256={actual}")
        if expected not in PENDING_VALUES and actual.lower() != expected.lower():
            failures += 1
            print(
                f"ERROR {artifact_id}: checksum mismatch expected={expected} actual={actual}",
                file=sys.stderr,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
