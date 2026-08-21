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

## Findings (Provisional — N=150 bursts / 120 participants)

> These numbers are from the **old 150-burst pilot**, re-run only to confirm the updated pipeline (PCA-reduced audio block, see "Sample-Size Planning" below) still works correctly — they are not the target-scale run. `power_analysis.py` shows 0% power at N=150 to detect even a small effect, so don't read anything into the ranking between B and C below; it will very likely change once the pipeline is actually re-run at the ~736–794-burst scale the data supports.

The notebook is parameterized to evaluate one outcome at a time; the run below is for `happy`. Out-of-fold predictions via Leave-One-Group-Out CV (grouped by participant, 120 folds):

| Model | Features (post-PCA) | R² (OOF) | RMSE (OOF) |
|-------|----------|----------|------------|
| A: Behavioral | 42 | −0.133 | 29.81 |
| B: Behavioral + handcrafted audio (PCA-10) | 52 | −0.074 | 29.02 |
| C: Behavioral + handcrafted audio (PCA-10) + AST (PCA-30) | 82 | −0.101 | 29.39 |

**Bottom line:** none of the three models beat a mean-outcome baseline (all R² < 0) — expected, given 0% power at this N. This table exists to prove the pipeline runs end-to-end after the audio-PCA change, not to draw a conclusion about whether audio content predicts well-being. That question needs the larger-N rerun described below.

---

## Pipeline Overview

```
DDP logs (ddp_table_ema_120.csv, ddp_view_url_120.csv)   [not included, see data/README.md]
    │
    ├─ pilot_sample.py          → pilot_sample_events.csv   (~736 bursts, full candidate pool, ≤5 URLs each)
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
| `pilot_sample.py` | Samples the full candidate pool of EMA bursts (~736 with ≥5 URLs, one per participant first then top-up), selects up to 5 TikTok URLs per burst. Anchors sampling on EMA file to guarantee join coverage. See "Sample-Size Planning" for why this isn't capped at an arbitrary smaller number anymore. |
| `power_analysis.py` | Standalone script — computes the sample size needed to reliably detect audio/AST effects via Cohen's f² and the noncentral F-distribution. Run before investing in a full pipeline rerun. |
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

These per-video features are aggregated to the burst level (mean + std across sampled videos), then reduced with PCA (n=10) before modeling — see "Sample-Size Planning" below for why.

See `section5.1/audio_feature.md` for full documentation.

### Deep Embeddings (AST)

The [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) (pretrained on AudioSet) produces a 768-dim embedding per video, extracted via a forward hook on the pre-classification layer. Embeddings are averaged across sampled videos within a burst, then reduced with PCA (n=30) before modeling.

`third_party/ast/` vendors just the `ASTModel` class needed for this (BSD-3-Clause) — not the full upstream repo. See `third_party/ast/README.md`.

---

## Models

Three Ridge regression models are compared using Leave-One-Group-Out cross-validation, grouped by participant:

| Model | Features | Count (post-PCA) |
|-------|----------|-------|
| A | Behavioral (DDP counts) | 42 |
| B | Behavioral + handcrafted audio (PCA-10) | 52 |
| C | Behavioral + handcrafted audio (PCA-10) + AST (PCA-30) | 82 |

Preprocessing: mean imputation → StandardScaler → PCA (audio and AST blocks only) → RidgeCV (α ∈ {0.01, 0.1, 1, 10, 100, 1000}).

> **Note:** Both content blocks (handcrafted audio and AST) are PCA-reduced for the same reason — see "Sample-Size Planning" below. Behavioral features aren't reduced; there are few enough of them relative to N that it isn't needed.

Result plots (`coefficients_model_A.png`, `coefficients_model_B.png`, `model_comparison.png`, `pca_explained_variance.png`, `predicted_vs_actual.png`) are aggregate model outputs and are included.

---

## Sample-Size Planning

The original 150-burst pilot was underpowered to answer its own research questions: a power analysis (`power_analysis.py`, Cohen's f² for an R² increment via the noncentral F-distribution) shows **0% power at N=150** to detect even a small effect (f²=0.02) of audio features on top of behavioral counts — meaning a negative result at that scale can't be distinguished from "the pipeline just isn't powered to see it."

Two independent levers came out of that analysis:
1. **More data, without new recruitment:** the existing DDP export already contains ~736–794 usable EMA bursts (124 participants) — the 150-burst pilot only used ~19% of what's already collected. `pilot_sample.py` now samples the full available pool instead of an arbitrary 150.
2. **PCA-reduce the handcrafted audio block**, the same treatment AST embeddings already got. Leaving it at 264 raw dimensions would need N≈3,579 for adequate power even after scaling up the burst count — reducing it to 10 components (matching the AST block's approach) brings that down to N≈875, within reach of the ~736–794 available bursts.

Even with both changes, this only gets you to ~75% power for a *small* effect — full power is reached only for medium-or-larger effects. That's the honest ceiling given the data currently on hand; getting more DDP-donating participants would still improve it further.

`power_analysis.py` is meant to be re-run whenever the candidate burst pool changes (e.g. if the lab collects more DDP data) — it's the way to check the target N before investing in another full pipeline rerun, rather than rerunning and hoping.

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
# 0. (Optional but recommended) Check what sample size is actually worth
#    targeting before investing in a full rerun
python power_analysis.py

# 1. Sample bursts (full candidate pool by default -- see Sample-Size Planning)
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
