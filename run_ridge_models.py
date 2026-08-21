"""
Non-interactive version of ridge_model.ipynb's modeling step, for unattended
/ overnight runs -- ridge_model.ipynb itself needs a human to open Jupyter
and hit "Run All" (or `nbconvert`, which isn't installed here), neither of
which happens automatically in a background pipeline run.

Runs all three models (A/B/C) for all four outcome variables (the notebook
only ever saved a run for `happy`; this is also what fills that gap) and
writes results to ridge_results.json + prints a summary table. The
plots/coefficient-inspection parts of the notebook are still notebook-only --
this script is for getting real numbers unattended, not for reproducing
every visualization.

Pipeline-building logic lives in modeling_utils.py, shared with
permutation_test.py, so a fix (like the outlier winsorizing added after a
real bug at N=736) only needs to happen in one place.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from modeling_utils import AUDIO_PCA, N_PCA, group_features, make_pipeline

OUTCOMES = ["happy", "smu_intention_pc", "smu_experience_happy", "life_satisfied"]


def evaluate(pipe, X, y, groups):
    y_pred = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups)
    return {
        "r2": r2_score(y, y_pred),
        "rmse": root_mean_squared_error(y, y_pred),
    }


def run_for_target(df, target):
    df_model = df.dropna(subset=[target]).copy()
    if len(df_model) < 10:
        return {"target": target, "error": f"only {len(df_model)} rows with non-missing target, skipping"}

    y = df_model[target].values
    groups = df_model["PID"].values
    n = len(df_model)
    n_groups = len(np.unique(groups))

    behav_cols, audio_cols, ast_cols = group_features(df_model, target)

    blocks_A = [("behav", behav_cols, None)]
    blocks_B = [("behav", behav_cols, None), ("audio", audio_cols, AUDIO_PCA)]
    blocks_C = [("behav", behav_cols, None), ("audio", audio_cols, AUDIO_PCA), ("ast", ast_cols, N_PCA)]

    res_A = evaluate(make_pipeline(blocks_A, n), df_model[behav_cols], y, groups)
    res_B = evaluate(make_pipeline(blocks_B, n), df_model[behav_cols + audio_cols], y, groups)
    res_C = evaluate(make_pipeline(blocks_C, n), df_model[behav_cols + audio_cols + ast_cols], y, groups)

    return {
        "target": target,
        "n_rows": n,
        "n_groups": n_groups,
        "model_A": res_A,
        "model_B": res_B,
        "model_C": res_C,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="final_model_df.csv")
    parser.add_argument("--out", default="ridge_results.json")
    args = parser.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.data)
    df = df.replace("[]", np.nan)
    print(f"Loaded {args.data}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Outcomes to run: {OUTCOMES}\n")

    all_results = []
    for target in OUTCOMES:
        print(f"=== {target} ===")
        t_start = time.time()
        result = run_for_target(df, target)
        result["seconds"] = round(time.time() - t_start, 1)
        all_results.append(result)

        if "error" in result:
            print(f"  {result['error']}")
        else:
            print(f"  N={result['n_rows']} rows, {result['n_groups']} groups (participants)")
            for m in ["model_A", "model_B", "model_C"]:
                r = result[m]
                print(f"  {m}: R2={r['r2']:.4f}  RMSE={r['rmse']:.4f}")
        print(f"  ({result['seconds']}s)\n")

    with open(args.out, "w") as f:
        json.dump({
            "data_file": args.data,
            "n_rows_loaded": int(df.shape[0]),
            "audio_pca": AUDIO_PCA,
            "ast_pca": N_PCA,
            "results": all_results,
            "total_seconds": round(time.time() - t0, 1),
        }, f, indent=2)

    print(f"Saved {args.out}. Total time: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
