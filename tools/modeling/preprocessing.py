import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from ..utils import binary_cols


# class NullDropper(TransformerMixin, BaseEstimator):
#     def __init__(self, *, drop_features=False, feature_thresh=0.5):
#         self.drop_features = drop_features
#         self.feature_thresh = feature_thresh

#     def fit(self, X, y=None):
#         if isinstance(X, pd.DataFrame):
#             self._df_columns = X.columns
#             self._df_index = X.index
#             self._orig_df = True
#         self.missing_mask_ = pd.DataFrame(X).isna().to_numpy()
#         return self

#     @property
#     def null_observations_(self):
#         check_is_fitted(self)
#         return self.missing_mask_[:, ~self.null_features_].any(axis=1)

#     @property
#     def null_features_(self):
#         check_is_fitted(self)
#         return self.missing_mask_.mean(axis=0) > self.feature_thresh

#     def transform(self, X):
#         check_is_fitted(self)
#         X = np.asarray(X)
#         self.dropped_observations_ = X[self.null_observations_]
#         self.dropped_features_ = X[:, self.null_features_]
#         return X[~self.null_observations_][:, ~self.null_features_].copy()

#     def inverse_transform(self, X):
#         check_is_fitted(self)
#         shape = self.missing_mask_.shape[0], X.shape[1]
#         reconst = np.zeros(shape, dtype=X.dtype)
#         reconst[self.null_observations_] = self.dropped_observations_[:, ~self.null_features_]
#         reconst[~self.null_observations_] = X
#         indices = np.arange(0, self.null_features_.size)[self.null_features_]
#         for i, feat in zip(indices, self.dropped_features_.T):
#             reconst = np.insert(reconst, i, feat, axis=1)
#         return reconst

def infer_feature_names(
    X_array: np.ndarray,
    X_frame: pd.DataFrame,
    dummies: bool = False,
    dummy_kws: dict = None,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Infer feature names for `X_array` from `X_frame`.

    When preprocessing data using Pandas and Scikit-Learn Pipelines, the
    feature names are easily lost. This function infers feature names
    from the nearest feature in the upstream DataFrame.

    It assumes that the ndarray and DataFrame have the same shape (after
    optional one-hot encoding), and that their rows (not columns) are aligned.
    Both structures are min-max scaled to [0, 1] and have missing values filled
    with 0 before distances are calculated.

    Parameters
    ----------
    X_array : ndarray
        Array with unknown feature names.
    X_frame : DataFrame
        Upstream DataFrame from which `X_array` was derived.
    dummies : bool, optional
        Use pandas.get_dummies to encode categoricals, by default False.
    dummy_kws: dict, optional
        Keyword arguments for pandas.get_dummies, by default None.
    metric: str or callable, optional
        Distance metric to use, by default 'euclidean'. If a string,
        distance function can be 'braycurtis', 'canberra', 'chebyshev',
        'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'. For more details,
        see the documentation for scipy.spatial.distance.cdist.

    Returns
    -------
    DataFrame
        `X_array` with labeled features.

    Raises
    ------
    ValueError
        `X_array` and `X_frame` must have the same shape.
    """
    if dummies:
        if dummy_kws is None:
            dummy_kws = dict()
        X_frame = pd.get_dummies(X_frame, **dummy_kws)
    if X_array.shape != X_frame.shape:
        raise ValueError(
            f"`X_array` and `X_frame` must have same shape. got {X_array.shape} and {X_frame.shape}"
        )
    downstream = pd.DataFrame(X_array).transform(minmax_scale).fillna(0.0)
    upstream = X_frame.transform(minmax_scale).fillna(0.0)
    dist_matrix = distance.cdist(upstream.values.T, downstream.values.T, metric=metric).T
    features = pd.DataFrame(dist_matrix, columns=X_frame.columns).idxmin()
    features = features.sort_values().index
    return pd.DataFrame(X_array, columns=features, index=X_frame.index)

def binary_features(X: np.ndarray, as_mask=False):
    df = pd.DataFrame(X)
    mask = (df.nunique() == 2).to_numpy()
    return mask if as_mask else df.columns[mask].to_numpy()
