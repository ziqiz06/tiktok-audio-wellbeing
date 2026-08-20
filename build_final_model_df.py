"""
Builds session_features.csv and final_model_df.csv from the pilot sample.

Design notes:
- Behavioral burst counts (views, likes, searches, etc.) come from
  ddp_table_ema_120.csv — these are complete, researcher-computed summaries
  of the full 120-min pre-EMA window. Do NOT recompute them from the sampled
  events file, which only contains up to 3 view-event rows per burst.

- Content/audio features (handcrafted + AST) are aggregated from
  events_with_features.csv, which is a PILOT SAMPLE of up to 3 videos per
  burst. These are approximations, not full burst representations.

- AST embeddings (768 dims) are saved raw in session_features.csv.
  For modeling, use PCA to reduce dimensionality before fitting.
"""
import os
import numpy as np
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))

# ── Load inputs ───────────────────────────────────────────────────────────────
events = pd.read_csv(os.path.join(base, "events_with_features.csv"))
ema    = pd.read_csv(os.path.join(base, "data", "ddp_table_ema_120.csv"))

print(f"events_with_features.csv : {events.shape}")
print(f"ddp_table_ema_120.csv    : {ema.shape}")

# ── Identify content feature column groups ────────────────────────────────────
ast_cols   = [c for c in events.columns if c.startswith("ast_")]
audio_cols = [c for c in events.columns if any(
    c.startswith(p) for p in [
        "tempo", "beat_count", "non_silence",
        "rms_", "zcr_", "centroid_", "rolloff_", "bandwidth_",
        "mfcc_", "contrast_", "chroma_", "tonnetz_",
    ]
)]

# ── Aggregate content features per EMA burst ─────────────────────────────────
# NOTE: only view events are in events_with_features.csv (pilot sample, up to
# 3 videos per burst). So we aggregate directly — no activity filter needed.
records = []

for (pid, burst), grp in events.groupby(["PID", "Start Date"]):
    row = {
        "PID": pid,
        "Start Date": burst,
        # How many pilot videos contributed to the content features this burst
        "pilot_n_videos_with_features": grp[ast_cols[0]].notna().sum() if ast_cols else np.nan,
        "pilot_n_videos_sampled": len(grp),
    }

    # Handcrafted audio: mean + std across sampled videos in burst
    for col in audio_cols:
        vals = grp[col].dropna()
        row[f"{col}_sess_mean"] = vals.mean() if len(vals) else np.nan
        row[f"{col}_sess_std"]  = vals.std()  if len(vals) else np.nan

    # AST embeddings: mean + std across sampled videos (raw, use PCA later)
    emb_matrix = grp[ast_cols].dropna(how="all")
    if len(emb_matrix):
        means = emb_matrix.mean(axis=0)
        stds  = emb_matrix.std(axis=0)
        for col in ast_cols:
            row[f"{col}_mean"] = means[col]
            row[f"{col}_std"]  = stds[col]
    else:
        for col in ast_cols:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"]  = np.nan

    records.append(row)

session_df = pd.DataFrame(records)
session_df.to_csv(os.path.join(base, "session_features.csv"), index=False)
print(f"\nsession_features.csv     : {session_df.shape}")
print(f"  Unique PIDs            : {session_df['PID'].nunique()}")
print(f"  Bursts with >=1 video with AST features: "
      f"{(session_df['pilot_n_videos_with_features'] > 0).sum()}")

# ── Normalize Start Date for joining ─────────────────────────────────────────
def parse_date(series):
    return pd.to_datetime(series, format="%m/%d/%Y %I:%M%p", errors="coerce")

session_df["_date_norm"] = parse_date(session_df["Start Date"])
ema["_date_norm"]        = parse_date(ema["Start Date"])

sess_nulls = session_df["_date_norm"].isna().sum()
ema_nulls  = ema["_date_norm"].isna().sum()
if sess_nulls: print(f"\n  WARNING: {sess_nulls} session Start Dates failed to parse")
if ema_nulls:  print(f"  WARNING: {ema_nulls} EMA Start Dates failed to parse")

# ── Diagnostics: unmatched pairs ─────────────────────────────────────────────
sess_keys = set(zip(session_df["PID"].astype(str), session_df["_date_norm"].astype(str)))
ema_keys  = set(zip(ema["PID"].astype(str), ema["_date_norm"].astype(str)))

only_sess = sess_keys - ema_keys
only_ema  = ema_keys  - sess_keys

print(f"\n  (PID, date) in session only (no EMA match) : {len(only_sess)}")
for pair in list(only_sess)[:5]:
    print(f"    {pair}")

print(f"  (PID, date) in EMA only (no session match) : {len(only_ema)}")
for pair in list(only_ema)[:5]:
    print(f"    {pair}")

# ── Merge: content features (session) + behavioral counts + outcomes (EMA) ───
# Behavioral counts come entirely from the EMA file — they cover the full burst,
# not just the 3 sampled videos.
final_df = session_df.merge(
    ema, on=["PID", "_date_norm"], how="inner", suffixes=("", "_ema")
)
final_df = final_df.drop(columns=["_date_norm"])
if "Start Date_ema" in final_df.columns:
    final_df = final_df.drop(columns=["Start Date_ema"])

final_df.to_csv(os.path.join(base, "final_model_df.csv"), index=False)
print(f"\nfinal_model_df.csv       : {final_df.shape}")
print(f"  Unique PIDs            : {final_df['PID'].nunique()}")

# ── Missing values for key outcomes ──────────────────────────────────────────
outcomes = ["happy", "smu_intention_pc", "smu_experience_happy", "life_satisfied"]
print("\nMissing values in outcome columns:")
for col in outcomes:
    if col in final_df.columns:
        n = final_df[col].isna().sum()
        print(f"  {col:30s}: {n} / {len(final_df)}")
    else:
        print(f"  {col:30s}: column not found")

# ── Funnel summary ────────────────────────────────────────────────────────────
pilot      = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
emb_df     = pd.read_csv(os.path.join(base, "embeddings.csv"))
craft_df   = pd.read_csv(os.path.join(base, "section5.1", "df_final.csv"))
videos_dir = os.path.join(base, "videos")
downloaded = [f for f in os.listdir(videos_dir) if not f.endswith(".wav")]
ev_raw     = pd.read_csv(os.path.join(base, "data", "ddp_view_url_120.csv"))

feat_col = ast_cols[0] if ast_cols else None
rows_with_feat = events.dropna(subset=[feat_col]) if feat_col else events.iloc[0:0]

print("\n=== FUNNEL SUMMARY ===")
print(f"  All events in ddp_view_url_120    : {len(ev_raw):>6}")
print(f"  Sampled bursts (pilot)            : {pilot.groupby(['PID','Start Date']).ngroups:>6}")
print(f"  Selected URLs (pilot)             : {pilot['url'].nunique():>6}")
print(f"  Downloaded video files            : {len(downloaded):>6}")
print(f"  AST embeddings extracted          : {len(emb_df):>6}")
print(f"  Handcrafted features extracted    : {craft_df['url'].nunique():>6}")
print(f"  Event rows with features          : {len(rows_with_feat):>6}  ({rows_with_feat['url'].nunique()} unique URLs)")
print(f"  Session rows (bursts aggregated)  : {len(session_df):>6}")
print(f"  EMA rows available                : {len(ema):>6}")
print(f"  Final matched rows                : {len(final_df):>6}")
