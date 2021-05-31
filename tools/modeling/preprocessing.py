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

tukey_winsorizer = partial(
    FunctionTransformer,
    func=outliers.tukey_winsorize,
    kw_args=dict(show_report=False),
)

standard_winsorizer = partial(
    FunctionTransformer,
    func=outliers.z_winsorize,
    kw_args=dict(show_report=False),
)

mean_imputer = partial(SimpleImputer, strategy="mean")
median_imputer = partial(SimpleImputer, strategy="median")
mode_imputer = partial(SimpleImputer, strategy="most_frequent")


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
    dist_matrix = distance.cdist(
        upstream.values.T, downstream.values.T, metric=metric
    ).T
    features = pd.DataFrame(dist_matrix, columns=X_frame.columns).idxmin()
    features = features.sort_values().index
    return pd.DataFrame(X_array, columns=features, index=X_frame.index)


class NameFinder(BaseEstimator, TransformerMixin):
    """Estimator which infers feature names using upstream DataFrame.

    When preprocessing data using Pandas and Scikit-Learn Pipelines, the
    feature names are easily lost. This function infers feature names
    by finding the nearest feature in the upstream DataFrame.

    It assumes that the ndarray and DataFrame have the same shape (after
    optional one-hot encoding), and that their rows (not columns) are aligned.
    Both structures are scaled and have missing values filled before distances
    are calculated.

    Parameters
    ----------
    X_frame : DataFrame
        Upstream DataFrame with labeled features.
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
        `X` with labeled features.

    Raises
    ------
    ValueError
        `X` and `X_frame` must have the same shape.
    """

    def __init__(
        self,
        X_frame: pd.DataFrame,
        dummies: bool = False,
        dummy_kws: dict = None,
        metric: str = "euclidean",
    ) -> None:
        self.X_frame = X_frame
        self.dummies = dummies
        self.dummy_kws = dummy_kws if dummy_kws else dict()
        self.metric = metric

    def fit(self, X, y=None):
        if self.dummies:
            upstream = pd.get_dummies(self.X_frame, **self.dummy_kws)
        else:
            upstream = self.X_frame.copy()

        if X.shape != upstream.shape:
            ValueError(
                "Expected `X` and `X_frame` to have same shape,"
                f" got {X.shape} and {upstream.shape}"
            )

        upstream = upstream.transform(scale)
        upstream.fillna(upstream.mean(), inplace=True)
        downstream = pd.DataFrame(X).transform(scale)
        downstream.fillna(downstream.mean(), inplace=True)

        self.distances_ = distance.cdist(
            upstream.values.T,
            downstream.values.T,
            metric=self.metric,
        ).T

        feature_names = pd.DataFrame(self.distances_, columns=upstream.columns).idxmin()
        self.feature_names_ = feature_names.sort_values().index.to_numpy()
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_, index=self.X_frame.index)

    def __repr__(self):
        params = pd.Series(self.get_params())
        str_type = params.map(type) == str
        params[str_type] = params[str_type].map(lambda x: f"'{x}'")
        params["X_frame"] = f"{self.X_frame.__class__.__name__}{self.X_frame.shape}"
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class SymmetricalWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, inner=0.9) -> None:
        self.inner = inner

    @property
    def outer(self):
        return 1 - self.inner

    @property
    def limits(self):
        return np.array([self.outer] * 2) / 2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return winsorize(X, limits=self.limits, axis=0, nan_policy="propagate")


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


def categorical_pipe(
    categories: str = "auto",
    drop: str = None,
    sparse: bool = True,
    dtype: Callable = np.float64,
    handle_unknown: str = "error",
) -> Pipeline:
    """Returns a Pipeline for mode-imputation and one-hot encoding.

    See the documentation for sklearn.impute.SimpleImputer,
    sklearn.preprocessing.OneHotEncoder, and sklearn.pipeline.Pipeline
    for more information.

    Parameters
    ----------
    categories : str or list of array-like
        Categories (unique values) per feature. Defaults to 'auto'.
    drop : str or array-like of shape (n_features,)
        Methodology to use to drop one of the categories per feature.
        Possible values: None, 'first', 'if_binary', array-like.
    sparse : bool
        Return sparse matrix. Defaults to True.
    dtype : number type
        Desired dtype of output. Defaults to np.float64.
    handle_unknown : str
        Whether to 'raise' an error or 'ignore' if an unknown categorical feature
        is present during transform (default is 'raise').

    Returns
    -------
    Pipeline
        SimpleImputer followed by OneHotEncoder.
    """
    pipe = Pipeline(
        [
            ("cat_imputer", mode_imputer()),
            ("cat_encoder", OneHotEncoder(**locals())),
        ]
    )
    return pipe


def _clone(estimator: BaseEstimator, safe: bool = True, **params) -> BaseEstimator:
    """Constructs an unfitted estimator with the same (updated) parameters.

    Wrapper for sklearn.base.clone which allows the clone to be updated
    with **params. See the documentation for sklearn.base.clone for more
    details.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be cloned.
    safe : bool
        If False, fall back to a deep copy on objects that are not
        estimators. True by default.
    **params
        Parameters to be passed to the clone's `set_params` method.

    Returns
    -------
    BaseEstimator
        Cloned estimator with (optionally) updated parameters.
    """
    new = clone(estimator, safe=safe)
    if params:
        new.set_params(**params)
    return new


def clone_factory(estimator: BaseEstimator, safe: bool = True) -> Callable:
    """Returns callable for generating unfitted clones of `estimator`.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be cloned.
    safe : bool
        If False, fall back to a deep copy on objects that are not
        estimators. True by default.

    Returns
    -------
    Callable
        Clone factory.
    """
    return partial(_clone, estimator, safe=safe)