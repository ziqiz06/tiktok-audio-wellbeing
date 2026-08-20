import ast
import os
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. Build video-level feature table ───────────────────────────────────────

emb_df = pd.read_csv(os.path.join(base, "embeddings.csv"))
emb_df["embedding"] = emb_df["embedding"].apply(ast.literal_eval)
ast_cols = [f"ast_{i+1}" for i in range(768)]
emb_expanded = pd.DataFrame(emb_df["embedding"].tolist(), columns=ast_cols, index=emb_df.index)
emb_df = pd.concat([emb_df.drop(columns=["embedding"]), emb_expanded], axis=1)

craft_df = pd.read_csv(os.path.join(base, "section5.1", "df_final.csv"))
craft_cols = [c for c in craft_df.columns if c not in ("PID", "Start Date", "time_utc")]
craft_video = craft_df[craft_cols].drop_duplicates(subset=["url"])

video_features = craft_video.merge(emb_df, on="url", how="inner")
video_features.to_csv(os.path.join(base, "video_features.csv"), index=False)
print(f"video_features.csv       →  {video_features.shape[0]} urls × {video_features.shape[1]} features")

# ── 2. Event-level table: pilot events + content features ────────────────────

pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
events_rich = pilot.merge(video_features, on="url", how="left")
events_rich.to_csv(os.path.join(base, "events_with_features.csv"), index=False)
print(f"events_with_features.csv →  {events_rich.shape[0]} events × {events_rich.shape[1]} columns")
