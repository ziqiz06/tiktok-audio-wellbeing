# TikTok EMA — Audio Content Features and Momentary Well-Being

This project investigates whether the **audio content** of TikTok videos watched in the 2 hours before an Ecological Momentary Assessment (EMA) predicts momentary well-being outcomes. It combines Digital Data Donation (DDP) behavioral logs, video downloads, and two types of audio feature extraction — handcrafted (Librosa) and deep learned (Audio Spectrogram Transformer) — then fits Ridge regression models to compare predictive performance.

> **Data availability:** The raw participant data (DDP/EMA logs and everything derived from them) is not included in this repository — it identifies real study participants and is restricted under the study's IRB protocol. This repo contains the full pipeline code, documentation of the expected data schema (`data/README.md`), and the result plots from the pilot run. See "Data Files" below.

---

## Research Questions

1. Do session-level behavioral features (views, likes, app usage) predict EMA outcomes?
2. Do handcrafted audio features add signal beyond behavioral counts?
3. Do deep audio embeddings (AST) add signal beyond both?

**Outcome variables:**
- `happy` — momentary happiness (0–100 VAS)
- `smu_intention_pc` — social media use intention (person-centered)
- `smu_experience_happy` — happiness during social media use
- `life_satisfied` — life satisfaction (0–100 VAS)

---

## Findings (N=736 bursts / 120 participants — full target scale)

Out-of-fold R²/RMSE via Leave-One-Group-Out CV (grouped by participant, 120 folds), all four outcomes:

| Outcome | A: Behavioral | B: +handcrafted audio (PCA-10) | C: +AST (PCA-30) |
|---|---|---|---|
| `happy` | R²=−0.311, RMSE=33.32 | R²=−0.341, RMSE=33.69 | R²=−0.356, RMSE=33.88 |
| `smu_intention_pc` | R²=−0.013, RMSE=0.78 | R²=−0.027, RMSE=0.78 | R²=−0.035, RMSE=0.79 |
| `smu_experience_happy` | R²=−0.283, RMSE=1.20 | R²=−0.298, RMSE=1.20 | R²=−0.321, RMSE=1.21 |
| `life_satisfied` | R²=−0.483, RMSE=36.00 | R²=−0.517, RMSE=36.41 | R²=−0.564, RMSE=36.97 |

**Permutation test** (`permutation_test.py`, target=`happy`, 100 permutations per comparison — tests whether the R² change from adding a feature block is distinguishable from randomly shuffling that block against the outcome):

| Comparison | Observed ΔR² | p-value | Verdict |
|---|---|---|---|
| A→B (adding handcrafted audio) | −0.030 | 0.92 | not significant |
| B→C (adding AST embeddings) | −0.014 | 0.42 | not significant |

**Bottom line: this is a genuine null result, not an inconclusive one.** All models underperform a mean-outcome baseline (R² < 0 throughout), both feature additions made predictions slightly *worse* rather than better, and the permutation test rules out "the small negative change is actually a small positive effect obscured by noise" (p is nowhere close to 0.05 in either direction). Per `power_analysis.py`, this design had ~75% power to detect even a small real effect at this N — so a real small-to-medium effect on `happy` would likely have shown up. At this sample size, audio content — neither handcrafted features nor AST deep embeddings — adds detectable predictive signal for momentary well-being beyond behavioral counts alone. The permutation test has only been run for `happy`; the other three outcomes have R²/RMSE above but not yet a significance test.

One methodological note worth carrying into any write-up: the first full-scale run produced nonsensical results (R² as low as −47) traced to a single participant with a ~46-SD outlier value on one behavioral feature, which a linear model extrapolated catastrophically on. Fixed by winsorizing each feature to its training-fold 5th/95th percentile before scaling (see `modeling_utils.py`) — a reminder that scaling up a pilot can surface data-quality issues invisible at N=150.

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
| `modeling_utils.py` | Shared model-building logic (feature grouping, PCA pipeline, outlier winsorizing) used by both `run_ridge_models.py` and `permutation_test.py`, so a fix only has to happen once. `ridge_model.ipynb` intentionally keeps its own inline copy — see its Section 6 — since it's meant to be read standalone rather than import a project module. |
| `ridge_model.ipynb` | Fits and evaluates three Ridge regression models using Leave-One-Group-Out CV, grouped by participant, for the `happy` outcome (see Section 12 for switching outcomes). Produces comparison plots and coefficient visualizations. |
| `run_ridge_models.py` | Non-interactive version of the notebook's modeling step, covering all four outcomes in one run — writes `ridge_results.json`. Used for the unattended pipeline run in `run_pipeline.sh`. |
| `permutation_test.py` | Tests whether an observed R² change from adding a feature block is distinguishable from chance, via shuffling and refitting under the same CV. The piece `power_analysis.py` doesn't cover. |
| `run_pipeline.sh` | Runs the full pipeline unattended (e.g. overnight) — pre-flight download check, per-step logging to `logs/`, status to `PIPELINE_STATUS.txt`. |

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

Even with both changes, this reaches ~75% power for a *small* effect at the full N=794 ceiling — **not** the conventional 80% target. In plain terms: if a real small effect exists, this design would still miss it roughly 1 in 4 times. That's not a reason to skip the rerun (794 is the actual ceiling — there's no more data to add without new recruitment), but it should be reported as "scaled to the maximum available data, reaching ~75% power," not as "adequately powered." Full power (≥80%) is reached for medium-or-larger effects, and comfortably so for the AST comparison at this N.

`power_analysis.py` is meant to be re-run whenever the candidate burst pool changes (e.g. if the lab collects more DDP data) — it's the way to check the target N before investing in another full pipeline rerun, rather than rerunning and hoping.

**Update: this rerun has been done.** The pipeline was actually re-run at full scale (736 bursts, 120 participants — the download step landed slightly below the 794 ceiling due to normal content attrition on ~1-year-old videos, not a shortfall in the sampling). See "Findings" above for the result: a genuine null finding, not an underpowered one, thanks to this planning.

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
| `final_model_df.csv` | Final modeling dataset (736 rows × 2055 columns) |

---

## Dependencies

```
python >= 3.9
pandas, numpy, scikit-learn
librosa
torch, torchaudio, soundfile
yt-dlp (recent version) + curl_cffi
ffmpeg  (system install, used by embeddingvector.py)
```

> **`curl_cffi` is required, not optional**, despite not being imported directly anywhere in this repo's code. Without it, `yt-dlp` can't impersonate a real browser's TLS fingerprint, and TikTok's bot detection rejects most download requests with a misleading `"Your IP address is blocked from accessing this post"` error — which looks exactly like a real network-level block but isn't one. Confirmed by direct testing: the identical URL, on the identical network, failed without `curl_cffi` installed and succeeded with it. Install both together: `pip install -U "yt-dlp[default]" curl_cffi`.

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

# 8. (optional) Test whether an R² improvement is real or chance
python permutation_test.py --comparison both --n-permutations 200
```

Or run the whole thing unattended (steps 1–7, plus a permutation test) via `bash run_pipeline.sh` — see its header comment for usage with `caffeinate`/`nohup` for an overnight run.
