# Data (not included)

The raw Digital Data Donation (DDP) and EMA files this pipeline runs on are **not included in this repository** — they contain real study participants' identifiers, GPS coordinates, IP addresses, demographics, and mental-health baseline scores, and are restricted under the study's IRB protocol.

This directory documents the two expected input files so the pipeline scripts (`pilot_sample.py`, `build_final_model_df.py`) are reproducible against equivalent data.

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

## Downstream files (also excluded, same reason)

`pilot_sample_events.csv`, `embeddings.csv`, `video_features.csv`, `events_with_features.csv`, `session_features.csv`, `final_model_df.csv`, and `section5.1/df_final.csv` are all derived from the two files above and carry the same participant-linkage restriction — see the root `README.md` "Data Files" table for what each one contains and the script that produces it.
