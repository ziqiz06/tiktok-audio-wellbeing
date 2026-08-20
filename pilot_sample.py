"""
Step 1 & 2: Sample at the EMA-burst level, maximize participant diversity.
Produces pilot_sample_events.csv.

IMPORTANT: Sample from EMA bursts first (ddp_table_ema_120.csv), then pull
matching view events. This guarantees every sampled burst has an EMA row to
merge with later. Sampling from the view file's Start Dates does NOT work
because only ~43% of view-file bursts have a matching EMA row.
"""
import os
import numpy as np
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))

# ── Load ──────────────────────────────────────────────────────────────────────
ev  = pd.read_csv(os.path.join(base, "data", "ddp_view_url_120.csv"))
ema = pd.read_csv(os.path.join(base, "data", "ddp_table_ema_120.csv"))

# Normalize join keys
for df in (ev, ema):
    df["PID"]        = df["PID"].astype(str).str.strip()
    df["Start Date"] = df["Start Date"].astype(str).str.strip()

print("=== STEP 1: Raw event file ===")
print(f"  Total rows        : {len(ev)}")
print(f"  Unique PIDs       : {ev['PID'].nunique()}")
print(f"  Unique URLs       : {ev['url'].nunique()}")

# ── Filter to view events with a URL ─────────────────────────────────────────
views = ev[ev["activity"] == "view"].dropna(subset=["url"]).copy()
views = views[["PID", "url", "Start Date", "time_utc"]].drop_duplicates()

# ── Build candidate bursts: EMA bursts that ALSO have view events ─────────────
# This is the key fix: anchor on EMA Start Dates, not view-file Start Dates.
ema_keys  = set(zip(ema["PID"], ema["Start Date"]))
view_keys = views.groupby(["PID", "Start Date"])["url"].nunique().reset_index()
view_keys.columns = ["PID", "Start Date", "n_urls"]

# Keep only bursts that exist in EMA AND have view events
view_keys["in_ema"] = view_keys.apply(
    lambda r: (r["PID"], r["Start Date"]) in ema_keys, axis=1
)
candidate_bursts = view_keys[view_keys["in_ema"]].copy()

print(f"\n  View bursts with >=1 URL      : {len(view_keys)}")
print(f"  Of those, also in EMA file    : {len(candidate_bursts)}")
print(f"  Unique PIDs in candidates     : {candidate_bursts['PID'].nunique()}")

# ── STEP 2: Sample 150 bursts, spread across PIDs ────────────────────────────
TARGET_BURSTS = 150
URLS_PER_BURST = 5

# Prefer bursts with >=5 URLs
good_bursts = candidate_bursts[candidate_bursts["n_urls"] >= URLS_PER_BURST].copy()

# One burst per PID first, then fill
pids = good_bursts["PID"].unique()
rng  = np.random.default_rng(seed=42)
rng.shuffle(pids)

round1 = []
for pid in pids:
    pid_bursts = good_bursts[good_bursts["PID"] == pid]
    chosen = pid_bursts.sample(n=1, random_state=42)
    round1.append(chosen)
    if len(round1) >= TARGET_BURSTS:
        break

sampled_bursts = pd.concat(round1).reset_index(drop=True)

# Top up if needed
if len(sampled_bursts) < TARGET_BURSTS:
    already = set(zip(sampled_bursts["PID"], sampled_bursts["Start Date"]))
    remaining = good_bursts[
        ~good_bursts.apply(lambda r: (r["PID"], r["Start Date"]) in already, axis=1)
    ]
    n_extra = TARGET_BURSTS - len(sampled_bursts)
    extra = remaining.sample(n=min(n_extra, len(remaining)), random_state=42)
    sampled_bursts = pd.concat([sampled_bursts, extra]).reset_index(drop=True)

print(f"\n=== STEP 2: Sampled bursts ===")
print(f"  Sampled bursts  : {len(sampled_bursts)}")
print(f"  Unique PIDs     : {sampled_bursts['PID'].nunique()}")

# ── For each sampled burst, keep up to URLS_PER_BURST view URLs ──────────────
selected_rows = []
for _, burst_row in sampled_bursts.iterrows():
    pid   = burst_row["PID"]
    burst = burst_row["Start Date"]
    burst_views = views[(views["PID"] == pid) & (views["Start Date"] == burst)]
    # Deduplicate by URL within this burst, take up to 3
    burst_urls = burst_views.drop_duplicates(subset=["url"]).head(URLS_PER_BURST)
    selected_rows.append(burst_urls)

pilot = pd.concat(selected_rows).reset_index(drop=True)

print(f"  Selected event rows : {len(pilot)}")
print(f"  Unique URLs selected: {pilot['url'].nunique()}")
print(f"\nPIDs in pilot:")
for pid in sorted(pilot['PID'].unique()):
    n = pilot[pilot['PID']==pid]['url'].nunique()
    print(f"  {pid}  →  {n} URLs")

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(base, "pilot_sample_events.csv")
pilot.to_csv(out, index=False)
print(f"\nSaved → pilot_sample_events.csv  ({pilot.shape})")
