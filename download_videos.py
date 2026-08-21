"""
Downloads videos listed in pilot_sample_events.csv into the videos/ folder.

Downloads run concurrently via a thread pool -- this is a network-bound step
(waiting on TikTok's CDN), not a CPU-bound one, so threads are appropriate
(unlike audio_feature_extraction.py, which is CPU-bound and uses processes).

MAX_WORKERS is deliberately modest: too much concurrency against one platform
risks tripping rate limiting/blocks, which would make this slower and less
reliable, not faster. Raise it cautiously and watch for a spike in failures.

Resumable: already-downloaded videos (matched by TikTok video ID already
present in videos/) are skipped before dispatching any work, so a partial
run can just be re-invoked rather than needing a full restart. Failures are
appended to download_failures.csv (not overwritten) so a failure pattern
across multiple runs -- e.g. a spike of "video unavailable" -- is visible
without re-parsing stdout.

Use --limit N to cap how many NEW (not-yet-downloaded) videos are attempted
this run -- e.g. a smoke test on real new data before committing to the
full batch: `python download_videos.py --limit 30`.
"""
import argparse
import csv
import datetime
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yt_dlp

# Without browser impersonation, TikTok's bot detection rejects most requests
# with a misleading "Your IP address is blocked" error -- confirmed by direct
# testing: the identical URL failed without this and succeeded with it, on
# the same network, same run. Requires the curl_cffi package. yt-dlp's own
# CLI accepts a plain string ("chrome"), but the Python API used here needs
# an actual ImpersonateTarget object -- passing a string crashes with an
# AssertionError deep in yt-dlp's internals, so this is constructed once and
# reused rather than left as a string in ydl_opts, and falls back to no
# impersonation (with a warning) if curl_cffi isn't installed at all, rather
# than crashing every single download.
try:
    from yt_dlp.networking.impersonate import ImpersonateTarget
    IMPERSONATE_TARGET = ImpersonateTarget.from_str("chrome")
except Exception as e:
    print(f"WARNING: could not set up browser impersonation ({e}). "
          f"Downloads will likely fail with 'IP address is blocked' errors. "
          f"Fix: pip install -U \"yt-dlp[default]\" curl_cffi")
    IMPERSONATE_TARGET = None

MAX_WORKERS = 6

base       = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(base, "videos")
failures_path = os.path.join(base, "download_failures.csv")
os.makedirs(videos_dir, exist_ok=True)


def video_id_from_url(url):
    return url.rstrip("/").split("/")[-1]


def already_downloaded_ids(videos_dir):
    """Video IDs that already have a file in videos/ (any extension except .wav,
    which is a derived audio file produced by a later pipeline step, not a download)."""
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(videos_dir)
        if not f.endswith(".wav")
    }


def download_video(url):
    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": os.path.join(videos_dir, "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
    }
    if IMPERSONATE_TARGET is not None:
        ydl_opts["impersonate"] = IMPERSONATE_TARGET
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return url, ydl.prepare_filename(info), None
        except Exception as e:
            return url, None, str(e)


def log_failure(url, error):
    is_new = not os.path.exists(failures_path)
    with open(failures_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "url", "error"])
        writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), url, error])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max NEW videos to attempt this run (for a smoke test before the full batch).")
    args = parser.parse_args()

    pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
    unique_urls = list(pilot["url"].dropna().unique())

    done_ids = already_downloaded_ids(videos_dir)
    to_download = [u for u in unique_urls if video_id_from_url(u) not in done_ids]
    skipped = len(unique_urls) - len(to_download)
    print(f"{len(unique_urls)} unique URLs total, {skipped} already downloaded, {len(to_download)} remaining")

    if args.limit is not None:
        to_download = to_download[:args.limit]
        print(f"--limit {args.limit}: attempting {len(to_download)} of the new videos")

    if not to_download:
        print("Nothing to do.")
        return

    print(f"Downloading {len(to_download)} videos with {MAX_WORKERS} workers...")
    print_lock = threading.Lock()
    completed = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_video, url): url for url in to_download}
        for future in as_completed(futures):
            url, path, error = future.result()
            completed += 1
            with print_lock:
                if path:
                    print(f"[{completed}/{len(to_download)}] Saved: {path}")
                else:
                    n_failed += 1
                    print(f"[{completed}/{len(to_download)}] Failed: {url}\n  {error}")
                    log_failure(url, error)

    print(f"\nDone. {completed - n_failed} succeeded, {n_failed} failed"
          + (f" (see {os.path.basename(failures_path)})" if n_failed else "") + ".")


if __name__ == "__main__":
    main()
