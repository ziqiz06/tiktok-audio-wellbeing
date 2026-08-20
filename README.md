# TikTok EMA — Audio Content Features and Momentary Well-Being

This project investigates whether the **audio content** of TikTok videos watched in the 2 hours before an Ecological Momentary Assessment (EMA) predicts momentary well-being outcomes. It combines Digital Data Donation (DDP) behavioral logs, video downloads, and two types of audio feature extraction — handcrafted (Librosa) and deep learned (Audio Spectrogram Transformer) — then fits Ridge regression models to compare predictive performance.

> **Data availability:** The raw participant data (DDP/EMA logs and everything derived from them) is not included in this repository — it identifies real study participants and is restricted under the study's IRB protocol. This repo contains the full pipeline code, documentation of the expected data schema (`data/README.md`), and the result plots from the pilot run. See "Data Files" below.

---

## Research Questions

1. Do session-level behavioral features (views, likes, app usage) predict EMA outcomes?
2. Do handcrafted audio features add signal beyond behavioral counts?
3. Do deep audio embeddings (AST) add signal beyond both?

**Outcome variables available in the data:**
- `happy` — momentary happiness (0–100 VAS) — **the only outcome evaluated so far; see Findings below**
- `smu_intention_pc` — social media use intention (person-centered) — defined, not yet modeled
- `smu_experience_happy` — happiness during social media use — defined, not yet modeled
- `life_satisfied` — life satisfaction (0–100 VAS) — defined, not yet modeled

---

## Findings (Pilot, N=150 bursts / 120 participants)

The notebook is parameterized to evaluate one outcome at a time; the saved run below is for `happy`. Out-of-fold predictions via Leave-One-Group-Out CV (grouped by participant, 120 folds):

| Model | Features | R² (OOF) | RMSE (OOF) |
|-------|----------|----------|------------|
| A: Behavioral | 42 | −0.133 | 29.81 |
| B: Behavioral + handcrafted audio | 306 | −0.077 | 29.06 |
| C: Behavioral + handcrafted audio + AST (PCA-30) | 336 | −0.079 | 29.08 |

**Bottom line:** none of the three models beat a mean-outcome baseline (all R² < 0) at this pilot scale — expected with ~150 noisy bursts and 300+ correlated predictors. Adding handcrafted audio features modestly reduced RMSE relative to behavioral counts alone; the AST deep embeddings added no further improvement over handcrafted audio (B and C are effectively tied). Read this as "audio content is not yet ruled out as a predictor," not as a positive result — the other three outcomes haven't been run yet, and a larger sample is needed before drawing real conclusions.

---

## Pipeline Overview

```
DDP logs (ddp_table_ema_120.csv, ddp_view_url_120.csv)   [not included, see data/README.md]
    │
    ├─ pilot_sample.py          → pilot_sample_events.csv   (150 bursts, ≤5 URLs each)
    │
    ├─ download_videos.py       → videos/                   (yt-dlp download)
    │
    ├─ section5.1/
    │   └─ audio_feature_extraction.py  → section5.1/df_final.csv   (Librosa features)
    │
    ├─ embeddingvector.py       → embeddings.csv            (AST 768-dim embeddings)
    │
    ├─ build_features.py        → video_features.csv        (join Librosa + AST)
    │                           → events_with_features.csv  (pilot events + features)
    │
    ├─ build_final_model_df.py  → session_features.csv      (burst-level aggregates)
    │                           → final_model_df.csv        (merge with EMA outcomes)
    │
    └─ ridge_model.ipynb        → model results + plots
```

---

## Scripts

| File | Description |
|------|-------------|
| `pilot_sample.py` | Samples 150 EMA bursts (one per participant first, then top-up), selects up to 5 TikTok URLs per burst. Anchors sampling on EMA file to guarantee join coverage. |
| `download_videos.py` | Downloads sampled TikTok videos via `yt-dlp` into `videos/`. |
| `section5.1/audio_feature_extraction.py` | Extracts handcrafted audio features per video using Librosa (MFCCs, chroma, spectral features, tempo, beat count, non-silence ratio). |
| `embeddingvector.py` | Runs each video through a pretrained Audio Spectrogram Transformer (AudioSet-pretrained) and extracts a 768-dim embedding from the pre-classification layer. |
| `build_features.py` | Joins Librosa and AST features into `video_features.csv`; merges onto pilot events to produce `events_with_features.csv`. |
| `build_final_model_df.py` | Aggregates video-level features to the burst level (mean + std across sampled videos), then merges with behavioral counts and EMA outcomes into `final_model_df.csv`. |
| `ridge_model.ipynb` | Fits and evaluates three Ridge regression models using Leave-One-Group-Out CV, grouped by participant. Produces comparison plots and coefficient visualizations. |

---

## Audio Features

### Handcrafted (Librosa)

Extracted at 22,050 Hz. Frame-level features are summarized per video as **mean**, **standard deviation**, and **volatility** (std of frame-to-frame differences).

| Feature | Variables |
|---------|-----------|
| RMS Energy | `rms_1_{mean,std,volatility}` |
| Zero-Crossing Rate | `zcr_1_{mean,std,volatility}` |
| Spectral Centroid | `centroid_1_{mean,std,volatility}` |
| Spectral Rolloff | `rolloff_1_{mean,std,volatility}` |
| Spectral Bandwidth | `bandwidth_1_{mean,std,volatility}` |
| Spectral Contrast | `contrast_{1–7}_{mean,std,volatility}` |
| MFCCs (13 coefficients) | `mfcc_{1–13}_{mean,std,volatility}` |
| Chroma (12 pitch classes) | `chroma_{1–12}_{mean,std,volatility}` |
| Tonnetz (6 harmonic dims) | `tonnetz_{1–6}_{mean,std,volatility}` |
| Tempo | `tempo` |
| Beat Count | `beat_count` |
| Non-Silence Ratio | `non_silence_ratio` |

See `section5.1/audio_feature.md` for full documentation.

### Deep Embeddings (AST)

The [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) (pretrained on AudioSet) produces a 768-dim embedding per video, extracted via a forward hook on the pre-classification layer. Embeddings are averaged across sampled videos within a burst, then reduced with PCA (n=30) before modeling.

`third_party/ast/` vendors just the `ASTModel` class needed for this (BSD-3-Clause) — not the full upstream repo. See `third_party/ast/README.md`.

---

## Models

Three Ridge regression models are compared using Leave-One-Group-Out cross-validation, grouped by participant (N=150 bursts across 120 participants):

| Model | Features | Count |
|-------|----------|-------|
| A | Behavioral (DDP counts) | 42 |
| B | Behavioral + handcrafted audio | 306 |
| C | Behavioral + handcrafted audio + AST (PCA-30) | 336 |

Preprocessing: mean imputation → StandardScaler → RidgeCV (α ∈ {0.01, 0.1, 1, 10, 100, 1000}).

> **Note:** This is a pilot run (N=150 bursts, 120 participants). R² estimates have high variance at this sample size and should be treated as directional only — see "Findings" above.

Result plots (`coefficients_model_A.png`, `coefficients_model_B.png`, `model_comparison.png`, `pca_explained_variance.png`, `predicted_vs_actual.png`) are aggregate model outputs and are included.

---

## Data Files

Raw and per-participant files are **not included** — see `data/README.md` for the documented schema. This table shows what each stage of the pipeline produces so the flow is auditable even without the data itself.

| File | Description |
|------|-------------|
| `data/ddp_table_ema_120.csv` | EMA responses + behavioral burst summaries (120-min window) |
| `data/ddp_view_url_120.csv` | Raw view events with TikTok URLs |
| `pilot_sample_events.csv` | Sampled burst-level view events (output of `pilot_sample.py`) |
| `embeddings.csv` | AST 768-dim embeddings per video URL |
| `section5.1/df_final.csv` | Librosa features per video, merged with pilot events |
| `video_features.csv` | Joined Librosa + AST features per video URL |
| `events_with_features.csv` | Pilot events with all video-level features attached |
| `session_features.csv` | Burst-level feature aggregates |
| `final_model_df.csv` | Final modeling dataset (150 rows × 2055 columns) |

---

## Dependencies

```
python >= 3.9
pandas, numpy, scikit-learn
librosa
torch, torchaudio, soundfile
yt-dlp
ffmpeg  (system install, used by embeddingvector.py)
```

The AST model weights (AudioSet pretrained) must be placed in `third_party/ast/pretrained_models/` — see `third_party/ast/pretrained_models/README.md` (downloaded automatically on first run if missing).

---

## Reproducing the Pipeline

```bash
# 1. Sample pilot bursts
python pilot_sample.py

# 2. Download videos
python download_videos.py

# 3. Extract handcrafted audio features
python section5.1/audio_feature_extraction.py

# 4. Extract AST embeddings
python embeddingvector.py

# 5. Join features
python build_features.py

# 6. Build modeling dataset
python build_final_model_df.py

# 7. Run models
jupyter notebook ridge_model.ipynb
```
