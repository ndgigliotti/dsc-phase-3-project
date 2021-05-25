import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import minmax_scale
from ..utils import broad_corr


# class NullTrimmer(TransformerMixin, BaseEstimator):
#     def __init__(self, *, trim_features=True, feature_thresh=0.5, copy=True):
#         self.trim_features = trim_features
#         self.feature_thresh = feature_thresh
#         self.copy = copy

#     def fit(self, X, y=None):
#         self.missing_mask_ = np.isnan(X)
#         return self

#     @property
#     def incomplete_rows(self):
#         check_is_fitted(self)
#         return self.missing_mask_.any(axis=1)

#     def transform(self, X, copy=None):
#         check_is_fitted(self)

#         copy = copy if copy is not None else self.copy
#         if copy:
#             X = X.copy()

#         self.trimmed_rows_ = X[self.incomplete_rows]
#         return X[~self.incomplete_rows]

#     def inverse_transform(self, X, copy=None):
#         check_is_fitted(self)
#         copy = copy if copy is not None else self.copy
#         if copy:
#             X = X.copy()
#         X[self.incomplete_rows_] = self.trimmed_rows_
#         return X


def infer_feature_names(
    X_array: np.ndarray, X_frame: pd.DataFrame, dummies: bool = False
) -> pd.DataFrame:
    """EXPERIMENTAL: Infer feature names for `X_array` from `X_frame`.

    When preprocessing data using Pandas and Scikit-Learn Pipelines, the
    feature names are easily lost. This function infers feature names
    using correlations with the unprocessed DataFrame.

    It assumes that the ndarray and DataFrame have the same shape (after
    optional one-hot encoding), that their rows (not columns) are aligned,
    that only linear transformations have been applied, and that there
    are no extremely high correlations between features. This function may fail
    if any transformation profoundly affecting correlation has been applied.

    Parameters
    ----------
    X_array : ndarray
        Array with unknown feature names.
    X_frame : DataFrame
        Upstream DataFrame from which `X_array` was derived.
    dummies : bool, optional
        One-hot encode categoricals in `X_frame`, by default False.

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
        X_frame = pd.get_dummies(X_frame)
    if X_array.shape != X_frame.shape:
        raise ValueError(
            f"`X_array` and `X_frame` must have same shape. got {X_array.shape} and {X_frame.shape}"
        )
    result = pd.DataFrame(X_array, index=X_frame.index)
    column_map = broad_corr(result, X_frame).idxmax()
    column_map = pd.Series(column_map.index, index=column_map.values)
    result.columns = result.columns.map(column_map)
    return result

def infer_feature_names2(
    X_array: np.ndarray, X_frame: pd.DataFrame, dummies: bool = False, metric="hamming") -> pd.DataFrame:
    """EXPERIMENTAL: Infer feature names for `X_array` from `X_frame`.

    When preprocessing data using Pandas and Scikit-Learn Pipelines, the
    feature names are easily lost. This function infers feature names
    using hamming distances (proportion of disagreeing components) with
    the upstream DataFrame.

    It assumes that the ndarray and DataFrame have the same shape (after
    optional one-hot encoding), that their rows (not columns) are aligned,
    that only linear transformations have been applied.

    Parameters
    ----------
    X_array : ndarray
        Array with unknown feature names.
    X_frame : DataFrame
        Upstream DataFrame from which `X_array` was derived.
    dummies : bool, optional
        One-hot encode categoricals in `X_frame`, by default False.

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
        X_frame = pd.get_dummies(X_frame)
    if X_array.shape != X_frame.shape:
        raise ValueError(
            f"`X_array` and `X_frame` must have same shape. got {X_array.shape} and {X_frame.shape}"
        )
    X_array = minmax_scale(X_array)
    X_frame = pd.DataFrame(minmax_scale(X_frame.values), columns=X_frame.columns, index=X_frame.index)
    dist_matrix = distance.cdist(X_frame.values.T, X_array.T, metric=metric)
    return pd.DataFrame(dist_matrix.T, columns=X_frame.columns)