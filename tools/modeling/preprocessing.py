import numpy as np
import pandas as pd
from typing import Callable
from functools import partial
from scipy.spatial import distance
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import (
    minmax_scale,
    scale,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.utils import as_float_array
from .. import outliers

# The following partial objects are shorthand callables
# for constructing commonly used estimators.

log_transformer = partial(
    FunctionTransformer,
    func=np.log,
    inverse_func=np.exp,
)

log10_transformer = partial(
    FunctionTransformer,
    func=np.log10,
    inverse_func=partial(np.power, 10),
)


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Simple quantile-based Winsorizer."""

    def __init__(self, inner: float = None) -> None:
            self.inner = inner
    @property
    def limits(self):
        return (1 - np.array([self.inner] * 2)) / 2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = as_float_array(X, force_all_finite="allow-nan")
        return winsorize(X, limits=self.limits, axis=0, nan_policy="propagate")


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Transformer wrapper for pd.get_dummies."""

    def __init__(
        self,
        prefix=None,
        prefix_sep="_",
        dummy_na=False,
        columns=None,
        sparse=False,
        drop_first=False,
    ):
        for key, value in locals().items():
            if key == "self":
                continue
            else:
                setattr(self, key, value)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dummies = pd.get_dummies(X, **self.get_params()).astype(np.float64)
        self.feature_names_ = dummies.columns.to_numpy()
        return dummies


def binary_features(X: np.ndarray, as_mask: bool = False) -> np.array:
    """Returns column indices of binary features.

    Parameters
    ----------
    X : np.ndarray
        Array to get binary feature indices from.
    as_mask : bool, optional
        Return boolean mask instead of indices, by default False.

    Returns
    -------
    np.ndarray
        Flat array of feature indices (or booleans).
    """
    df = pd.DataFrame(X)
    mask = (df.nunique() == 2).to_numpy()
    return mask if as_mask else df.columns[mask].to_numpy()