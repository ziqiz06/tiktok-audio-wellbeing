"""
Permutation test: is the R^2 improvement from adding a feature block
(handcrafted audio, or AST embeddings) bigger than you'd expect by chance?

This is the piece power_analysis.py doesn't give you. Power analysis tells
you whether your sample size is *big enough to detect* a real effect if one
exists -- it says nothing about whether the specific improvement you saw in
a given run is real or noise. This script answers that second question
directly, for whatever run you just did.

Method (standard permutation test for "does this feature block add signal
beyond the baseline"):
  1. Fit baseline model and extended model (baseline + new block) on the
     REAL data via Leave-One-Group-Out CV (grouped by PID), same as
     ridge_model.ipynb. Record observed R^2 for each and the observed
     improvement (delta = R2_extended - R2_baseline).
  2. Repeat N_PERMUTATIONS times: shuffle the new block's rows (breaking the
     true correspondence between that participant's audio/AST features and
     their outcome + behavioral features, while leaving the block's own
     internal feature covariance intact -- so PCA within each permutation
     is still meaningful, just decoupled from y). Refit ONLY the extended
     model (baseline is unaffected by permuting a block it doesn't contain)
     under the same CV. Record the permuted R^2 and delta.
  3. p-value = fraction of permuted deltas >= the observed delta.
     A small p-value means the real improvement is unlikely to have
     happened by chance; a large one means it's indistinguishable from
     shuffling the audio/AST features randomly.

Pipeline-building logic (including outlier winsorizing, added after a real
bug at N=736 -- see modeling_utils.py) lives in modeling_utils.py, shared
with run_ridge_models.py.

Runtime: each permutation refits the extended model across every LOGO fold
(one fit per participant). At N~150 this is seconds per permutation; at the
target N~736-794 scale, budget more -- start with a small N_PERMUTATIONS
to gauge per-permutation time before committing to a long run. Exact
p-value granularity is limited to 1/N_PERMUTATIONS, so don't run so few
permutations that you can't distinguish e.g. p=0.01 from p=0.05.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from modeling_utils import AUDIO_PCA, N_PCA, group_features, make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_oof_r2(pipe, X, y, groups):
    y_pred = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups)
    return r2_score(y, y_pred)


def run_permutation_test(df, target, comparison, n_permutations, seed=42):
    """comparison: 'AB' (does handcrafted audio add signal beyond behavior)
    or 'BC' (does AST add signal beyond behavior+audio)."""
    df_model = df.dropna(subset=[target]).copy()
    y = df_model[target].values
    groups = df_model["PID"].values
    n = len(df_model)

    behav_cols, audio_cols, ast_cols = group_features(df_model, target)

    if comparison == "AB":
        baseline_blocks = [("behav", behav_cols, None)]
        baseline_cols = behav_cols
        new_cols = audio_cols
        new_block_name = "audio"
        new_n_pca = AUDIO_PCA
    elif comparison == "BC":
        baseline_blocks = [("behav", behav_cols, None), ("audio", audio_cols, AUDIO_PCA)]
        baseline_cols = behav_cols + audio_cols
        new_cols = ast_cols
        new_block_name = "ast"
        new_n_pca = N_PCA
    else:
        raise ValueError("comparison must be 'AB' or 'BC'")

    # Baseline model -- fit once, unaffected by permuting the new block
    pipe_baseline = make_pipeline(baseline_blocks, n)
    r2_baseline = fit_oof_r2(pipe_baseline, df_model[baseline_cols], y, groups)

    # Observed extended model (real, unpermuted)
    extended_blocks = baseline_blocks + [(new_block_name, new_cols, new_n_pca)]
    extended_cols = baseline_cols + new_cols
    pipe_extended = make_pipeline(extended_blocks, n)
    r2_observed = fit_oof_r2(pipe_extended, df_model[extended_cols], y, groups)
    observed_delta = r2_observed - r2_baseline

    print(f"Comparison: {comparison}  (baseline {len(baseline_cols)} raw cols -> "
          f"+{new_block_name} {len(new_cols)} raw cols, PCA-{new_n_pca})")
    print(f"  R2 baseline : {r2_baseline:.4f}")
    print(f"  R2 extended : {r2_observed:.4f}")
    print(f"  Observed delta R2 : {observed_delta:.4f}")
    print(f"  Running {n_permutations} permutations...")

    rng = np.random.default_rng(seed)
    null_deltas = np.empty(n_permutations)
    X_extended = df_model[extended_cols].copy()

    for i in range(n_permutations):
        X_perm = X_extended.copy()
        perm_idx = rng.permutation(n)
        X_perm[new_cols] = X_extended[new_cols].to_numpy()[perm_idx]

        pipe_i = make_pipeline(extended_blocks, n)
        r2_perm = fit_oof_r2(pipe_i, X_perm, y, groups)
        null_deltas[i] = r2_perm - r2_baseline

        if (i + 1) % max(1, n_permutations // 10) == 0:
            print(f"    {i + 1}/{n_permutations} permutations done")

    p_value = (np.sum(null_deltas >= observed_delta) + 1) / (n_permutations + 1)  # +1: avoid p=0

    print(f"\n  Null delta R2: mean={null_deltas.mean():.4f}  std={null_deltas.std():.4f}")
    print(f"  p-value = {p_value:.4f}  (fraction of {n_permutations} permutations with "
          f"delta R2 >= observed {observed_delta:.4f}, +1 smoothing)")
    if p_value < 0.05:
        print(f"  -> Observed improvement is unlikely under the null (p < 0.05): "
              f"the {new_block_name} block's contribution looks like real signal, not noise.")
    else:
        print(f"  -> Cannot distinguish the observed improvement from chance at alpha=0.05: "
              f"the {new_block_name} block's contribution is not statistically supported by this run.")

    return {
        "comparison": comparison,
        "r2_baseline": r2_baseline,
        "r2_observed": r2_observed,
        "observed_delta": observed_delta,
        "null_deltas": null_deltas,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="happy",
                         choices=["happy", "smu_intention_pc", "smu_experience_happy", "life_satisfied"])
    parser.add_argument("--comparison", default="both", choices=["AB", "BC", "both"])
    parser.add_argument("--n-permutations", type=int, default=200,
                         help="More = finer p-value resolution but longer runtime. "
                              "Start small to gauge per-permutation time at your current N.")
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "final_model_df.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.replace("[]", np.nan)
    print(f"Loaded {args.data}: {df.shape[0]} rows x {df.shape[1]} columns, target={args.target}\n")

    comparisons = ["AB", "BC"] if args.comparison == "both" else [args.comparison]
    results = []
    for c in comparisons:
        results.append(run_permutation_test(df, args.target, c, args.n_permutations))
        print()

    summary = pd.DataFrame([{
        "comparison": r["comparison"],
        "R2_baseline": round(r["r2_baseline"], 4),
        "R2_extended": round(r["r2_observed"], 4),
        "delta_R2": round(r["observed_delta"], 4),
        "p_value": round(r["p_value"], 4),
    } for r in results])
    print("=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
