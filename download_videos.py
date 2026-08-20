"""
Downloads videos listed in pilot_sample_events.csv into the videos/ folder.
"""
import os
import pandas as pd
import yt_dlp

base      = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(base, "videos")
os.makedirs(videos_dir, exist_ok=True)

pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
unique_urls = pilot["url"].dropna().unique()
print(f"Downloading {len(unique_urls)} unique URLs...")


def download_video(url):
    ydl_opts = {
        "quiet": False,
        "outtmpl": os.path.join(videos_dir, "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
        except Exception as e:
            print(f"  Failed: {url}\n  {e}")
            return None


for i, url in enumerate(unique_urls):
    print(f"\n[{i+1}/{len(unique_urls)}] {url}")
    path = download_video(url)
    if path:
        print(f"  Saved: {path}")

print("\nDone.")
