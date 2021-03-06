import inspect
from typing import Callable, List

import numpy as np
import pandas as pd
from pandas._typing import FrameOrSeries

NULL = frozenset([np.nan, pd.NA, None])


def numeric_cols(data: pd.DataFrame) -> list:
    """Returns a list of all numeric column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the numeric column names.
    """
    return data.select_dtypes("number").columns.to_list()


def true_numeric_cols(data: pd.DataFrame, min_unique=3) -> list:
    """Returns numeric columns with at least `min_unique` unique values.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        Numeric column names.
    """
    num = data.select_dtypes("number")
    return num.columns[min_unique <= num.nunique()].to_list()


def cat_cols(data: pd.DataFrame, min_cats: int = None, max_cats: int = None) -> list:
    """Returns a list of categorical column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.
    min_cats : int, optional
        Minimum number of categories, by default None.
    max_cats : int, optional
        Maximum number of categories, by default None.

    Returns
    -------
    list
        Categorical column names.
    """
    cats = data.select_dtypes("category")
    cat_counts = cats.nunique()
    if min_cats is None:
        min_cats = cat_counts.min()
    if max_cats is None:
        max_cats = cat_counts.max()
    keep = (min_cats <= cat_counts) & (cat_counts <= max_cats)
    return cats.columns[keep].to_list()


def multicat_cols(data: pd.DataFrame) -> list:
    """Returns column names of categoricals with 3+ categories.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        Categorical (3+) column names.
    """
    cats = data.select_dtypes("category")
    return cats.columns[3 <= cats.nunique()].to_list()


def noncat_cols(data: pd.DataFrame) -> list:
    """Returns a list of all non-categorical column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the non-categorical column names.
    """
    return data.columns.drop(cat_cols(data)).to_list()


def binary_cols(data: pd.DataFrame) -> list:
    """Returns a list of columns with exactly 2 unique values.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the binary column names.
    """
    return data.columns[data.nunique() == 2].to_list()


def get_defaults(callable: Callable) -> dict:
    """Returns dict of parameters with their default values, if any.

    Parameters
    ----------
    callable : Callable
        Callable to look up parameters for.

    Returns
    -------
    dict
        Parameters with default values, if any.

    Raises
    ------
    TypeError
        `callable` must be Callable.
    """
    if not isinstance(callable, Callable):
        raise TypeError(f"`callable` must be Callable, not {type(callable)}")
    params = pd.Series(inspect.signature(callable).parameters)
    defaults = params.map(lambda x: x.default)
    return defaults.to_dict()


def pandas_heatmap(
    frame: pd.DataFrame,
    subset=None,
    na_rep="",
    precision=3,
    cmap="vlag",
    low=0,
    high=0,
    vmin=None,
    vmax=None,
    axis=None,
):
    """Style DataFrame as a heatmap."""
    table = frame.style.background_gradient(
        subset=subset, cmap=cmap, low=low, high=high, vmin=vmin, vmax=vmax, axis=axis
    )
    table.set_na_rep(na_rep)
    table.set_precision(precision)
    # table.highlight_null("white")
    return table


def filter_pipe(
    data: FrameOrSeries,
    like: List[str] = None,
    regex: List[str] = None,
    axis: int = None,
) -> FrameOrSeries:
    """Subset the DataFrame or Series labels with more than one filter at once.

    Parameters
    ----------
    data: DataFrame or Series
        DataFrame or Series to filter labels on.
    like : list of str
        Keep labels from axis for which "like in label == True".
    regex : list of str
        Keep labels from axis for which re.search(regex, label) == True.
    axis : {0 or ???index???, 1 or ???columns???, None}, default None
        The axis to filter on, expressed either as an index (int)
        or axis name (str). By default this is the info axis,
        'index' for Series, 'columns' for DataFrame.

    Returns
    -------
    Dataframe or Series
        Subset of `data`.
    """
    if like and regex:
        raise ValueError("Cannot pass both `like` and `regex`")
    elif like:
        if isinstance(like, str):
            like = [like]
        for exp in like:
            data = data.filter(like=exp, axis=axis)
    elif regex:
        if isinstance(regex, str):
            regex = [regex]
        for exp in like:
            data = data.filter(regex=exp, axis=axis)
    else:
        raise ValueError("Must pass either `like` or `regex` but not both")
    return data


def to_title(snake_case: str):
    """Format snake case string as title."""
    return snake_case.replace("_", " ").strip().title()


def title_mode(data: pd.DataFrame):
    """Return copy of `data` with strings formatted as titles."""
    result = data.copy()
    result.update(result.select_dtypes("object").applymap(to_title))
    for label, column in result.select_dtypes("category").items():
        result[label] = column.cat.rename_categories(to_title)
    if result.columns.dtype == "object":
        result.columns = result.columns.map(to_title)
    if result.index.dtype == "object":
        result.index = result.index.map(to_title)
    return result


def cartesian(*xi: np.ndarray) -> np.ndarray:
    """Return Cartesian product of 1d arrays.

    Returns
    -------
    ndarray
        Cartesian product.
    """
    return np.array(np.meshgrid(*xi)).T.reshape(-1, len(xi))


def broad_corr(frame: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Get correlations between features of one frame with those of another.

    Parameters
    ----------
    frame : DataFrame
        First DataFrame.
    other : DataFrame
        Second DataFrame.

    Returns
    -------
    DataFrame
        Pearson correlations.
    """
    return other.apply(lambda x: frame.corrwith(x))


def swap_index(data: pd.Series) -> pd.Series:
    """Swap index and values.

    Parameters
    ----------
    data : Series
        Series for swapping index and values.

    Returns
    -------
    Series
        Swapped Series.
    """
    return pd.Series(data.index, index=data.values, name=data.name, copy=True)
