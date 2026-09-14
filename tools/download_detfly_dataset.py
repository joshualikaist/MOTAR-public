"""Download the official public Det-Fly folders through their anonymous OneDrive API.

The upstream project publishes two SharePoint folder links rather than archive URLs.  This tool
opens each public page with Chromium, extracts its short-lived anonymous read token, enumerates the
folder through SharePoint's read-only v2 API, and downloads files atomically.  Existing files with
the recorded size are skipped, so interrupted downloads can be resumed by running the same command.

No token or pre-authenticated download URL is written to disk.  The receipt retains file names,
sizes, upstream quick-XOR hashes and a digest of that stable manifest.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import requests


ANNOTATIONS_SHARE = (
    "https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/"
    "Em9kWPCMYm1KpFHnhVOTn9UBQfQ25m0lH67xIcULuYXYHw?e=AWZfiw"
)
IMAGES_SHARE = (
    "https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/"
    "EqFYguroD9lEnVDTRkyoOJQBrHvndTQGa5f8EQurGRFUqQ?e=DXPQxN"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "datasets" / "detfly"
SHARE_CONTEXTS = {}
SHARE_CONTEXT_LOCK = threading.Lock()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--chrome", default="google-chrome")
    return parser.parse_args()


def page_context(share_url, chrome):
    with tempfile.TemporaryDirectory(prefix="detfly-page-") as profile:
        command = [
            chrome, "--headless", "--disable-gpu", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}", "--dump-dom", share_url,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, timeout=60, check=True)
    html = result.stdout
    drive = re.search(r'"\.driveUrl":"([^"]+)"', html)
    token = re.search(r'"\.driveAccessToken":"([^"]+)"', html)
    if not drive or not token:
        raise RuntimeError("SharePoint anonymous drive context was not present in the rendered page")
    return drive.group(1).replace(r"\u002f", "/"), token.group(1)


def tokenized_api_url(url, token):
    # SharePoint pagination links contain an access_token parameter that is not usable for the
    # anonymous shared drive.  Preserve pagination parameters but replace that value with the
    # token extracted from the public page.
    base, separator, query = url.partition("?")
    parameters = ([part for part in query.split("&")
                   if part and not part.startswith("access_token=")]
                  if separator else [])
    return base + "?" + "&".join(parameters + [token])


def api_json(url, token, attempts=5):
    url = tokenized_api_url(url, token)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def embedded_expiry(value):
    """Return an exp claim from a JWT-like SharePoint token without retaining its contents."""
    for segment in value.split("."):
        try:
            padded = segment + "=" * (-len(segment) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            if "exp" in payload:
                return int(payload["exp"])
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return 0


def cache_share_context(share_url, drive, token):
    SHARE_CONTEXTS[share_url] = {
        "drive": drive,
        "token": token,
        "expires": embedded_expiry(token.split("=", 1)[-1]),
    }


def share_context(share_url, chrome, force=False):
    with SHARE_CONTEXT_LOCK:
        cached = SHARE_CONTEXTS.get(share_url)
        if not force and cached and cached["expires"] > time.time() + 300:
            return cached["drive"], cached["token"]
        drive, token = page_context(share_url, chrome)
        cache_share_context(share_url, drive, token)
        return drive, token


def enumerate_folder(drive, token, remote_root):
    first = drive + "/root:/" + urllib.parse.quote(remote_root, safe="/") + ":/children?$top=200"
    queue = [(remote_root, first)]
    files = []
    while queue:
        parent, url = queue.pop(0)
        while url:
            page_base = url.partition("?")[0]
            payload = api_json(url, token)
            for item in payload.get("value", []):
                relative = parent + "/" + item["name"]
                if "folder" in item:
                    child_url = drive + "/items/" + urllib.parse.quote(item["id"], safe="") + "/children?$top=200"
                    queue.append((relative, child_url))
                elif "file" in item:
                    files.append({
                        "item_id": item["id"],
                        # Python 3.8 compatibility (str.removeprefix was added in 3.9).
                        "relative": (relative[len("Det-Fly/"):]
                                     if relative.startswith("Det-Fly/") else relative),
                        "size": int(item["size"]),
                        "quick_xor_hash": item.get("file", {}).get("hashes", {}).get("quickXorHash"),
                        # Despite its name, downloadUrlNoAuth returns HTTP 401 for this public
                        # share.  The tempauth-bearing downloadUrl is the working anonymous URL.
                        "download_url": item.get("@content.downloadUrl")
                                        or item.get("@content.downloadUrlNoAuth"),
                    })
            next_link = payload.get("@odata.nextLink")
            if next_link:
                # The API returns pagination links in a legacy drives('id') URL form which
                # rejects this anonymous token.  Retain the working endpoint and copy only the
                # opaque pagination query parameters.
                next_query = urllib.parse.urlsplit(next_link).query
                next_query = "&".join(part for part in next_query.split("&")
                                      if not part.startswith("access_token="))
                url = page_base + "?" + next_query
            else:
                url = None
    return files


def stable_manifest(files):
    rows = [{key: item.get(key) for key in ("relative", "size", "quick_xor_hash")}
            for item in sorted(files, key=lambda value: value["relative"])]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(encoded).hexdigest()


DOWNLOAD_THREAD = threading.local()


def download_session():
    session = getattr(DOWNLOAD_THREAD, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "MOTAR-DetFly-Downloader/1"})
        DOWNLOAD_THREAD.session = session
    return session


def signed_url_expiry(url):
    values = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("tempauth", [])
    return embedded_expiry(values[0]) if values else 0


def refreshed_download_url(item):
    existing = item.get("download_url")
    if existing and signed_url_expiry(existing) > time.time() + 120:
        return existing
    last_error = None
    for force in (False, True):
        try:
            drive, token = share_context(item["share_url"], item["chrome"], force=force)
            metadata_url = drive + "/items/" + urllib.parse.quote(item["item_id"], safe="")
            with download_session().get(
                    tokenized_api_url(metadata_url, token), timeout=(30, 60)) as response:
                if response.status_code == 200:
                    payload = response.json()
                    url = payload.get("@content.downloadUrl")
                    if url:
                        return url
                    last_error = "metadata omitted @content.downloadUrl"
                else:
                    last_error = "metadata HTTP %s" % response.status_code
        except (requests.RequestException, ValueError) as exc:
            # Do not retain the signed request URL from the exception.
            last_error = type(exc).__name__
    raise RuntimeError("could not refresh item download URL: %s" % last_error)


def download_file(item, output, attempts=5):
    destination = output / item["relative"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == item["size"]:
        return "skipped", item["size"]
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset == item["size"]:
                os.replace(part, destination)
                return "downloaded", item["size"]
            headers = {}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            download_url = refreshed_download_url(item)
            with download_session().get(download_url, headers=headers, stream=True,
                                        timeout=(30, 120)) as response:
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with part.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
            if part.stat().st_size != item["size"]:
                raise IOError(f"size {part.stat().st_size} != {item['size']}")
            os.replace(part, destination)
            return "downloaded", item["size"]
        except (requests.RequestException, RuntimeError, TimeoutError, OSError) as exc:
            if attempt + 1 == attempts:
                # Signed URLs are credentials.  Suppress request exception text, which embeds it.
                raise RuntimeError("download failed after %s attempts (%s)" % (
                    attempts, type(exc).__name__)) from None
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main():
    args = arguments()
    if args.annotations_only and args.images_only:
        raise SystemExit("choose at most one of --annotations-only and --images-only")
    wanted = []
    if not args.images_only:
        wanted.append(("Annotations", ANNOTATIONS_SHARE))
    if not args.annotations_only:
        wanted.append(("JPEGImages", IMAGES_SHARE))

    files = []
    for kind, share in wanted:
        print(f"[detfly] opening public {kind} page", flush=True)
        drive, token = page_context(share, args.chrome)
        cache_share_context(share, drive, token)
        discovered = enumerate_folder(drive, token, f"Det-Fly/{kind}")
        for item in discovered:
            item["share_url"] = share
            item["chrome"] = args.chrome
        print(f"[detfly] {kind}: {len(discovered):,} files, "
              f"{sum(row['size'] for row in discovered) / 1e9:.3f} GB", flush=True)
        files.extend(discovered)

    rows, manifest_sha = stable_manifest(files)
    total_bytes = sum(row["size"] for row in rows)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "download_receipt.json"
    receipt = {
        "schema_version": 1,
        "source": "official Jake-WU/Det-Fly public OneDrive folders",
        "source_repository": "https://github.com/Jake-WU/Det-Fly",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": len(rows),
        "bytes": total_bytes,
        "stable_manifest_sha256": manifest_sha,
        "manifest": rows,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"[detfly] manifest {manifest_sha}; receipt {receipt_path}", flush=True)
    if args.list_only:
        return 0

    lock = threading.Lock()
    completed = completed_bytes = transferred_bytes = skipped = 0
    download_errors = []
    started = time.monotonic()
    last_report = started
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, item, output): item for item in files}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                status, size = future.result()
            except Exception as exc:
                status, size = "failed", 0
                download_errors.append({
                    "relative": item["relative"],
                    "error_type": type(exc).__name__,
                })
            with lock:
                completed += 1
                completed_bytes += size
                transferred_bytes += size if status == "downloaded" else 0
                skipped += status == "skipped"
                now = time.monotonic()
                if now - last_report >= 15 or completed == len(files):
                    rate = transferred_bytes / max(now - started, 1) / 1e6
                    eta_seconds = ((total_bytes - completed_bytes) / (rate * 1e6)
                                   if rate > 0 else float("inf"))
                    print(f"[detfly] {completed:,}/{len(files):,} files · "
                          f"{completed_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB verified · "
                          f"{rate:.1f} MB/s transferred · skipped {skipped:,} · "
                          f"failed {len(download_errors):,} · "
                          f"ETA {eta_seconds / 3600:.1f} h", flush=True)
                    last_report = now

    bad = []
    for row in rows:
        path = output / row["relative"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            bad.append(row["relative"])
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["verified_by_size"] = len(rows) - len(bad)
    receipt["verification_failures"] = bad
    receipt["download_errors"] = download_errors
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if bad:
        print(f"[detfly] FAIL: {len(bad)} files missing or wrong-sized", flush=True)
        return 2
    print(f"[detfly] PASS: {len(rows):,} files verified by recorded upstream size", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
