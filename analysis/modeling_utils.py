"""
Shared model-building logic for run_ridge_models.py and permutation_test.py.

Factored out after this logic was duplicated across three places (this file's
predecessor state, permutation_test.py, and ridge_model.ipynb) and drifted --
AUDIO_PCA/N_PCA got out of sync between them more than once. ridge_model.ipynb
intentionally keeps its own inline copy despite that risk: it's meant to be
read top-to-bottom as a standalone notebook, not to import a project module.
If you change anything here, check ridge_model.ipynb's Section 6/13 matches.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AUDIO_PCA = 10
N_PCA = 30
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each column to its [lower_pct, upper_pct] percentile range,
    fit on training data only (so no leakage across CV folds).

    Added after a real bug: at N=736, one participant's ddp_density_post_person
    was 1.50 against a population range of 0-0.13 (a ~46 SD outlier, likely a
    near-zero-denominator artifact in the raw DDP export). Ridge has no
    protection against linear extrapolation on a held-out point that far
    outside the training distribution -- this one participant's 14 held-out
    rows got predicted around -1050 for a variable bounded [0,100], which
    alone was enough to send pooled R^2 from ~-0.1 (expected, matches the
    smaller pilot) to ~-29 (nonsensical). Winsorizing at (5,95) fixed it;
    tighter (1,99) still left a second, less extreme outlier causing the
    same problem for a different participant.
    """
    def __init__(self, lower_pct=5, upper_pct=95):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower_pct, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Winsorizing clips values in place -- it doesn't add, drop, or
        # reorder columns, so feature names pass through unchanged. Needed
        # for ColumnTransformer.get_feature_names_out() to work when this
        # step is in the pipeline (used for labeling coefficient plots).
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(len(self.lower_))], dtype=object)


def make_block_pipeline(cols, n_pca, n_samples):
    steps = [
        ("impute", SimpleImputer(strategy="mean")),
        ("winsorize", Winsorizer()),
        ("scale", StandardScaler()),
    ]
    if n_pca is not None:
        safe_n = min(n_pca, len(cols), n_samples - 1)
        steps.append(("pca", PCA(n_components=safe_n)))
    return Pipeline(steps)


def make_pipeline(blocks, n_samples):
    """blocks: list of (name, cols, n_pca_or_None)."""
    ct = ColumnTransformer([
        (name, make_block_pipeline(cols, n_pca, n_samples), cols)
        for name, cols, n_pca in blocks
    ])
    return Pipeline([
        ("preprocess", ct),
        ("ridge", RidgeCV(alphas=ALPHAS, scoring="r2")),
    ])


def group_features(df, target):
    exclude = {"PID", "Start Date", target, "pilot_n_videos_sampled", "pilot_n_videos_with_features"}
    all_cols = set(df.columns) - exclude

    behav_cols = sorted([c for c in all_cols if any(c.startswith(p) or p in c for p in ["ddp_", "app_Browser"])])
    ast_cols = sorted([c for c in all_cols if c.startswith("ast_") and c.endswith("_mean")])
    audio_patterns = ["mfcc", "chroma", "tempo", "rms_", "zcr_", "tonnetz",
                       "centroid", "rolloff", "bandwidth", "contrast", "beat_count", "non_silence"]
    audio_cols = sorted([c for c in all_cols if any(p in c for p in audio_patterns) and not c.startswith("ast_")])

    def keep_numeric(cols):
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    return keep_numeric(behav_cols), keep_numeric(audio_cols), keep_numeric(ast_cols)
