import sys
import os

# Add third_party/ast/src to the path (trimmed AST vendor copy, see third_party/ast/README.md)
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_base_dir = os.path.join(_root_dir, "data")  # raw, derived, and downloaded data (gitignored)
_ast_dir = os.path.join(_root_dir, "third_party", "ast")
sys.path.insert(0, _ast_dir)

from src.models import ASTModel

import subprocess
import numpy as np
import soundfile as sf
import torch
import torchaudio
import pandas as pd

# ── Device: use MPS (Apple Silicon GPU) if available ─────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ── Step 2: Load AST model ────────────────────────────────────────────────────
model = ASTModel(
    label_dim=527,
    input_tdim=1024,
    imagenet_pretrain=True,
    audioset_pretrain=True,
)
model = model.to(device)
model.eval()

# ── Step 5: Hook to extract 768-dim embedding before the classification head ──
# In forward(): x = (cls_token + dist_token) / 2  →  mlp_head(x) → logits
# We capture the input to mlp_head, which is the 768-dim representation.
_embedding_store = {}
def _hook(module, input, output):
    _embedding_store["vector"] = input[0].detach().cpu()
model.mlp_head.register_forward_hook(_hook)


# ── Step 1: Extract audio from video using ffmpeg ─────────────────────────────
def extract_audio(video_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ── Step 3: Convert wav → log-mel spectrogram, shape (1024, 128) ──────────────
def wav_to_spectrogram(wav_path, target_length=1024):
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T)  # (channels, samples)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        n_mels=128,
    )
    log_mel = torch.log(mel_transform(waveform) + 1e-7)  # (1, 128, time)

    # AudioSet normalization (from AST README)
    log_mel = (log_mel + 4.2677393) / (4.5689974 * 2)

    spec = log_mel.squeeze(0).transpose(0, 1)  # (time, 128)

    # Pad or trim to target_length
    if spec.shape[0] < target_length:
        spec = torch.nn.functional.pad(spec, (0, 0, 0, target_length - spec.shape[0]))
    else:
        spec = spec[:target_length, :]

    return spec  # (1024, 128)


# ── Steps 4 & 5: Forward pass + extract embedding ────────────────────────────
def get_embedding(wav_path):
    spec = wav_to_spectrogram(wav_path).unsqueeze(0).to(device)  # (1, 1024, 128)
    with torch.no_grad():
        _ = model(spec)
    return _embedding_store["vector"].squeeze(0)  # (768,)


# ── Single-video test ─────────────────────────────────────────────────────────
videos_dir = os.path.join(_base_dir, "videos")
all_files = [f for f in os.listdir(videos_dir) if not f.endswith(".wav")]

if not all_files:
    print("No downloaded videos found. Run pipeline/download_videos.py first.")
else:
    test_file = all_files[0]
    test_video = os.path.join(videos_dir, test_file)
    test_wav = os.path.join(videos_dir, "test.wav")

    print(f"Testing with: {test_file}")
    extract_audio(test_video, test_wav)
    emb = get_embedding(test_wav)
    print(f"Embedding shape: {emb.shape}")  # should be torch.Size([768])

    if emb.shape[0] != 768:
        raise RuntimeError(f"Unexpected embedding shape: {emb.shape}. Check the pipeline.")

    print("Single-video test passed. Running full loop...\n")

    # ── Loop over pilot sample videos ────────────────────────────────────────
    # Resumable + checkpointed: for an unattended/overnight run, a crash partway
    # through should lose at most CHECKPOINT_EVERY videos' work, not all of it.
    CHECKPOINT_EVERY = 20
    embeddings_path = os.path.join(_base_dir, "embeddings.csv")

    pilot = pd.read_csv(os.path.join(_base_dir, "sampled_events.csv"))
    unique_urls = pilot["url"].dropna().unique()

    already_done = set()
    if os.path.exists(embeddings_path):
        existing_df = pd.read_csv(embeddings_path)
        already_done = set(existing_df["url"].dropna().unique())
        print(f"Resuming: {len(already_done)} embeddings already saved, skipping them")
    else:
        existing_df = pd.DataFrame(columns=["url", "embedding"])

    remaining_urls = [u for u in unique_urls if u not in already_done]
    print(f"{len(remaining_urls)} of {len(unique_urls)} videos remaining\n")

    def save_checkpoint(existing_df, new_results):
        if not new_results:
            return existing_df
        combined = pd.concat([existing_df, pd.DataFrame(new_results)], ignore_index=True)
        combined.to_csv(embeddings_path, index=False)
        return combined

    pending = []
    n_failed = 0
    for i, url in enumerate(remaining_urls):
        print(f"[{i+1}/{len(remaining_urls)}] {url}")
        video_id = url.rstrip("/").split("/")[-1]

        matches = [f for f in os.listdir(videos_dir) if f.startswith(video_id) and not f.endswith(".wav")]
        if not matches:
            print("  Skipping — no downloaded file found.")
            continue

        video_path = os.path.join(videos_dir, matches[0])
        wav_path = os.path.join(videos_dir, f"{video_id}.wav")

        try:
            extract_audio(video_path, wav_path)
            emb = get_embedding(wav_path)
            pending.append({"url": url, "embedding": emb.numpy().tolist()})
            print(f"  OK — embedding shape: {emb.shape}")
        except Exception as e:
            n_failed += 1
            print(f"  Failed: {e}")

        if len(pending) >= CHECKPOINT_EVERY:
            existing_df = save_checkpoint(existing_df, pending)
            print(f"  [checkpoint: {len(existing_df)} embeddings saved to embeddings.csv]")
            pending = []

    existing_df = save_checkpoint(existing_df, pending)
    print(f"\nDone. {len(existing_df)} total embeddings saved, {n_failed} failed this run → embeddings.csv")
