"""
Power analysis: how many bursts (N) do we need to reliably detect whether
audio features add predictive signal beyond behavioral counts alone?

Method: Cohen's f^2 effect size for an R^2 increment in nested regression,
using the noncentral F-distribution directly to compute power at a given N,
then searching for the smallest N that reaches a target power.

Run this BEFORE scaling up the data pipeline -- it tells you the target N,
so you know whether a rerun can realistically reach a conclusive answer.

Two real comparisons from this project's actual model design (see README):
  A -> B : Behavioral (42) -> + handcrafted audio, RAW/unreduced (+264)
  B -> C : Behavioral+audio (306) -> + AST embeddings, PCA-reduced (+30)
These have very different predictor counts, so they need separate power
numbers -- lumping them into one generic "audio block" placeholder hides
that the two additions are not comparable in cost.
"""

import numpy as np
from scipy.stats import f as f_dist, ncf

# ---------------------------------------------------------------------------
# Effect size assumptions
# ---------------------------------------------------------------------------
# Cohen's f^2 = (R2_full - R2_reduced) / (1 - R2_full)
# Current pilot R^2 values are negative (noise-dominated -- not a real
# effect size to extrapolate from), so plan around Cohen's conventional
# benchmarks instead of the pilot's own noisy number:
#   small  f2 = 0.02   <- realistic range for subtle behavioral/psych effects
#   medium f2 = 0.15
#   large  f2 = 0.35
EFFECT_SIZES = [("small", 0.02), ("medium", 0.15), ("large", 0.35)]

# ---------------------------------------------------------------------------
# Real comparisons from ridge_model.ipynb (Model A=42, B=306, C=336 features)
# ---------------------------------------------------------------------------
SCENARIOS = [
    ("A -> B: behavioral -> +handcrafted audio (raw, unreduced)",
     dict(k_baseline=42, n_new_predictors=264)),
    ("B -> C: behavioral+audio -> +AST embeddings (PCA-30)",
     dict(k_baseline=306, n_new_predictors=30)),
    ("A -> B (if audio block were PCA-reduced to 10 dims, hypothetical)",
     dict(k_baseline=42, n_new_predictors=10)),
]

ALPHA = 0.05
TARGET_POWER = 0.80     # standard target: 80% chance of detecting a true effect
N_MAX_SEARCH = 20000    # upper bound for the search -- raw-audio scenario may need a big range


def power_for_n(n, df_num, k_baseline, f2, alpha=ALPHA):
    """Power to detect an R^2 increment of Cohen's f2 at sample size n."""
    k_total = k_baseline + df_num
    df_denom = n - k_total - 1
    if df_denom <= 0:
        return 0.0
    ncp = f2 * df_denom  # noncentrality parameter (Cohen's formula)
    f_crit = f_dist.ppf(1 - alpha, df_num, df_denom)
    return 1 - ncf.cdf(f_crit, df_num, df_denom, ncp)


def required_n(df_num, k_baseline, f2, target_power=TARGET_POWER, alpha=ALPHA, n_max=N_MAX_SEARCH):
    for n in range(k_baseline + df_num + 5, n_max):
        if power_for_n(n, df_num, k_baseline, f2, alpha) >= target_power:
            return n
    return None  # not reachable within n_max -- effect too small to detect practically


if __name__ == "__main__":
    for scenario_label, params in SCENARIOS:
        k_baseline = params["k_baseline"]
        n_new = params["n_new_predictors"]
        print("=" * 78)
        print(scenario_label)
        print(f"  baseline predictors = {k_baseline}, new predictors = {n_new}, "
              f"alpha={ALPHA}, target power={TARGET_POWER:.0%}")
        print()

        for label, f2 in EFFECT_SIZES:
            n_needed = required_n(n_new, k_baseline, f2)
            if n_needed:
                print(f"  {label:>6} effect (f2={f2}): N needed = {n_needed}")
            else:
                print(f"  {label:>6} effect (f2={f2}): not reachable within N={N_MAX_SEARCH} -- "
                      f"effect too subtle to detect practically at this feature/predictor ratio")

        print(f"\n  Power at specific sample sizes (small effect, f2=0.02):")
        for n in [150, 300, 500, 1000, 2000]:
            p = power_for_n(n, n_new, k_baseline, 0.02)
            print(f"    N={n:>5}: power = {p:.1%}")
        print()
