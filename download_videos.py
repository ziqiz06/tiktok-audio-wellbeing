"""
Downloads videos listed in pilot_sample_events.csv into the videos/ folder.

Downloads run concurrently via a thread pool -- this is a network-bound step
(waiting on TikTok's CDN), not a CPU-bound one, so threads are appropriate
(unlike audio_feature_extraction.py, which is CPU-bound and uses processes).

MAX_WORKERS is deliberately modest: too much concurrency against one platform
risks tripping rate limiting/blocks, which would make this slower and less
reliable, not faster. Raise it cautiously and watch for a spike in failures.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yt_dlp

MAX_WORKERS = 6

base      = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(base, "videos")
os.makedirs(videos_dir, exist_ok=True)

pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
unique_urls = pilot["url"].dropna().unique()
print(f"Downloading {len(unique_urls)} unique URLs with {MAX_WORKERS} workers...")

_print_lock = threading.Lock()


def download_video(url):
    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": os.path.join(videos_dir, "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return url, ydl.prepare_filename(info), None
        except Exception as e:
            return url, None, str(e)


completed = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(download_video, url): url for url in unique_urls}
    for future in as_completed(futures):
        url, path, error = future.result()
        completed += 1
        with _print_lock:
            if path:
                print(f"[{completed}/{len(unique_urls)}] Saved: {path}")
            else:
                print(f"[{completed}/{len(unique_urls)}] Failed: {url}\n  {error}")

print("\nDone.")
