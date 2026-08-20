import signal
import librosa
import numpy as np
import pandas as pd
import os

TIMEOUT_SECONDS = 60  # max time per video before skipping

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
videos_dir = os.path.join(base, "videos")
out_path = os.path.join(base, "section5.1", "df_final.csv")

# Build a map: video_id -> file path (skip .wav files)
video_files = {
    os.path.splitext(f)[0]: os.path.join(videos_dir, f)
    for f in os.listdir(videos_dir)
    if not f.endswith(".wav")
}

# Load pilot sample and filter to URLs that were actually downloaded
pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
pilot = pilot.dropna(subset=["url"])

unique_videos = [
    url for url in pilot["url"].unique()
    if url.rstrip("/").split("/")[-1] in video_files
]
print(f"Found {len(unique_videos)} pilot URLs with downloaded video files")

# ── Resume: skip URLs already in df_final.csv ────────────────────────────────
already_done = set()
if os.path.exists(out_path):
    existing = pd.read_csv(out_path)
    already_done = set(existing["url"].dropna().unique())
    print(f"Resuming: {len(already_done)} URLs already processed, skipping them")

unique_videos = [u for u in unique_videos if u not in already_done]
print(f"Remaining: {len(unique_videos)} videos to process")


# ── Timeout handler ───────────────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError()


def summarize_feature(x, prefix, features):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    for j in range(x.shape[0]):
        series = x[j]
        features[f"{prefix}_{j+1}_mean"] = float(np.mean(series))
        features[f"{prefix}_{j+1}_std"]  = float(np.std(series))
        if len(series) > 1:
            features[f"{prefix}_{j+1}_volatility"] = float(np.std(np.diff(series)))
        else:
            features[f"{prefix}_{j+1}_volatility"] = np.nan


feature_rows = []

for i, url in enumerate(unique_videos):
    video_id = url.rstrip("/").split("/")[-1]
    file_path = video_files[video_id]
    print(f"[{i+1}/{len(unique_videos)}] Processing: {url}")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        y, sr = librosa.load(file_path, sr=22050)

        rms       = librosa.feature.rms(y=y)
        zcr       = librosa.feature.zero_crossing_rate(y=y)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz    = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        non_silent_intervals = librosa.effects.split(y)
        non_silent_samples   = sum(end - start for start, end in non_silent_intervals)
        non_silence_ratio    = non_silent_samples / len(y) if len(y) > 0 else np.nan

        signal.alarm(0)  # cancel alarm

        features = {
            "url":               url,
            "tempo":             float(np.asarray(tempo).squeeze()),
            "beat_count":        int(len(beat_frames)),
            "non_silence_ratio": float(non_silence_ratio),
        }

        summarize_feature(rms,       "rms",       features)
        summarize_feature(zcr,       "zcr",       features)
        summarize_feature(centroid,  "centroid",  features)
        summarize_feature(rolloff,   "rolloff",   features)
        summarize_feature(bandwidth, "bandwidth", features)
        summarize_feature(mfcc,      "mfcc",      features)
        summarize_feature(contrast,  "contrast",  features)
        summarize_feature(chroma,    "chroma",    features)
        summarize_feature(tonnetz,   "tonnetz",   features)

        feature_rows.append(features)
        print("  Success")

    except TimeoutError:
        signal.alarm(0)
        print(f"  Timed out after {TIMEOUT_SECONDS}s — skipping")
    except Exception as e:
        signal.alarm(0)
        print(f"  Feature extraction failed: {e}")

features_df = pd.DataFrame(feature_rows)

# Merge back with pilot event info
df_new = pilot.merge(features_df, on="url", how="left")

# Append to any existing results
if already_done:
    existing = pd.read_csv(out_path)
    df_final = pd.concat([existing, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(out_path, index=False)
print(f"\nSaved {len(features_df)} new videos → section5.1/df_final.csv ({len(df_final)} total rows)")
