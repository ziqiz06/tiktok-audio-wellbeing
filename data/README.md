# Data (not included)

The raw Digital Data Donation (DDP) and EMA files this pipeline runs on are **not included in this repository** — they contain real study participants' identifiers, GPS coordinates, IP addresses, demographics, and mental-health baseline scores, and are restricted under the study's IRB protocol.

All pipeline inputs and outputs live in this directory (gitignored). This README documents the two expected input files so the pipeline scripts (`pipeline/sample_bursts.py`, `pipeline/build_model_dataset.py`) are reproducible against equivalent data.

## `ddp_table_ema_120.csv`

One row per EMA (Ecological Momentary Assessment) occasion, joined with a 120-minute pre-EMA behavioral summary. Key column groups:

- **Survey metadata** — `Start Date`, `End Date`, `Participant ID`, `Day of Survey`, `Occasion within Day`, `Finished`
- **EMA outcomes** — `happy`, `angry`, `inspired`, `lonely`, `distressed`, `life_satisfied`, `smu_intention*`, `smu_experience_*` (0–100 VAS or Likert)
- **Demographics / baseline traits** — `Age`, `Sex`, `Ethnicity simplified`, baseline anxiety/depression/FOMO/PSMU scores, etc.
- **Behavioral burst summaries (`ddp_*`)** — per-burst and per-person counts, durations, and match-timestamps for TikTok actions: `like`, `login`, `search`, `share`, `view`, `comment`, `post`, `follow`, `favorite`
- **App-usage minutes (`app_*_burst` / `app_*_person` / `app_*_personday`)** — screen time by app category, from device usage logs

Join key: `(PID, Start Date)`.

## `ddp_view_url_120.csv`

Raw view-event log: one row per TikTok video viewed in a burst window.

| Column | Description |
|---|---|
| `url` | TikTok video share URL |
| `time_utc` / `time` | Event timestamp |
| `Start Date` | Burst window this event belongs to (join key back to the EMA table) |
| `PID` | Participant ID |
| `activity` | Event type, e.g. `view`, `like` |
| `shared_content`, `shared_method`, `like_received_from_post` | Optional metadata, mostly null for `view` events |

## Derived files (also excluded, same reason)

Every file below is derived from the two inputs above and carries the same participant-linkage restriction.

| File | Produced by | Contents |
|---|---|---|
| `sampled_events.csv` | `pipeline/sample_bursts.py` | Sampled EMA bursts, ≤5 viewed-video URLs each |
| `videos/` | `pipeline/download_videos.py` | Downloaded TikTok videos |
| `download_failures.csv` | `pipeline/download_videos.py` | Append-only log of failed downloads (used to skip them on resume) |
| `audio_features.csv` | `pipeline/extract_audio_features.py` | Librosa features per video |
| `embeddings.csv` | `pipeline/extract_ast_embeddings.py` | 768-dim AST embedding per video |
| `video_features.csv` | `pipeline/build_features.py` | Librosa + AST features joined per video URL |
| `events_with_features.csv` | `pipeline/build_features.py` | Sampled events with video features attached |
| `session_features.csv` | `pipeline/build_model_dataset.py` | Burst-level feature aggregates (mean + std across videos) |
| `final_model_df.csv` | `pipeline/build_model_dataset.py` | Final modeling table (736 bursts × 2,055 columns) |
