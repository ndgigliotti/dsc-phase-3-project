from functools import singledispatch

import numpy as np
import pandas as pd
from pandas.api.types import is_integer, is_integer_dtype, is_float_dtype
from sklearn.preprocessing import StandardScaler

import tools.utils as utils

_rng = np.random.default_rng(42)


def get_iqr(data: pd.Series) -> float:
    """Returns IQR of `data`.

    Args:
        data (pd.Series): Series for calculating IQR.

    Returns:
        [float]: IQR
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    return q3 - q1


def iqr_fences(data: pd.Series) -> tuple:
    """Returns lower and upper Tukey fences.

    Args:
        data (pd.Series): Series for calculating fences.

    Returns:
        tuple: (lower, upper)
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def _display_report(outliers, verb):
    if isinstance(outliers, pd.Series):
        outliers = outliers.to_frame()
    report = outliers.sum()
    n_modified = outliers.any(axis=1).sum()
    report["total_obs"] = n_modified
    report = report.to_frame(f"n_{verb}")
    report[f"pct_{verb}"] = (report.squeeze() / outliers.shape[0]) * 100
    display(report)


@singledispatch
def iqr_outliers(data: pd.Series) -> pd.Series:
    """Returns boolean mask of IQR-fence outliers.

    Args:
        data (pd.Series): Series or DataFrame for finding outliers.

    Returns:
        pd.Series: Series or DataFrame mask.
    """
    lower, upper = iqr_fences(data)
    return (data < lower) | (data > upper)


@iqr_outliers.register
def _(data: pd.DataFrame) -> pd.DataFrame:
    """Function for DataFrames"""
    return data.apply(iqr_outliers)


def _jitter(shape, dist, dtype=np.float64, positive=True):
    if is_float_dtype(dtype):
        if positive:
            jitter = _rng.uniform(0, dist, shape).astype(dtype)
        else:
            jitter = _rng.uniform(dist * -1, dist, shape).astype(dtype)
    elif is_integer_dtype(dtype):
        dist = round(dist)
        if positive:
            jitter = _rng.integers(dist, size=shape, dtype=dtype, endpoint=True)
        else:
            jitter = _rng.integers(
                dist * -1, high=dist, size=shape, dtype=dtype, endpoint=True
            )
    else:
        raise ValueError(f"`dtype` must be either int or float dtype, got {dtype}")
    return jitter


def _jitter_like(data: pd.Series, dist: float, positive=True):
    return _jitter(data.shape, dist, dtype=data.dtype, positive=positive)


def _jitter_clipped(
    clipped: pd.Series, lower_outs: pd.Series, upper_outs: pd.Series, dist: float
):
    lower_jitter = _jitter_like(clipped[lower_outs], dist)
    upper_jitter = _jitter_like(clipped[upper_outs], dist)
    jittered = clipped.copy()
    jittered[lower_outs] += lower_jitter
    jittered[upper_outs] -= upper_jitter
    return jittered


@singledispatch
def iqr_winsorize(data: pd.Series, silent=False):
    """Reset outliers to outermost inlying values.

    Args:
        data (pd.Series): Series or DataFrame for clipping.
        silent (bool, optional): Do not display report. Defaults to False.

    Returns:
        [pd.Series]: Series or DataFrame
    """
    outliers = iqr_outliers(data)
    min_in, max_in = data[~outliers].agg(["min", "max"])
    if not silent:
        _display_report(outliers, "clipped")
    return data.clip(lower=min_in, upper=max_in)


@iqr_winsorize.register
def _(data: pd.DataFrame, silent=False) -> pd.DataFrame:
    """Function for DataFrames"""
    clipped = data.apply(iqr_winsorize, silent=True)
    if not silent:
        _display_report(iqr_outliers(data), "clipped")
    return clipped


@singledispatch
def iqr_clip(data: pd.Series, jitter=False, silent=False):
    """Move outliers to the Tukey fences.
    Args:
        data (pd.Series): Series or DataFrame for clipping.
        jitter (bool, optional): Add uniform noise. Defaults to False.
        silent (bool, optional): Do not display report. Defaults to False.
    Returns:
        [pd.Series]: Series or DataFrame
    """
    lower, upper = iqr_fences(data)
    clipped = data.clip(lower=lower, upper=upper)
    if is_integer_dtype(data):
        clipped = clipped.round().astype(data.dtype)
    if not silent:
        outliers = (data < lower) | (data > upper)
        _display_report(outliers, "clipped")
    if jitter:
        dist = get_iqr(data) / 5
        if is_integer_dtype(data):
            dist = round(dist)
        clipped = _jitter_clipped(clipped, data < lower, data > upper, dist)
    return clipped


@iqr_clip.register
def _(data: pd.DataFrame, jitter=False, silent=False) -> pd.DataFrame:
    """Function for DataFrames"""
    clipped = data.apply(iqr_clip, jitter=jitter, silent=True)
    if not silent:
        _display_report(iqr_outliers(data), "clipped")
    return clipped


@singledispatch
def iqr_drop(data: pd.DataFrame, silent=False) -> pd.DataFrame:
    """Drop IQR-fence outliers from `data`.

    Args:
        data (pd.DataFrame): Series or DataFrame for removing outliers.
        silent (bool, optional): Do not display report. Defaults to False.

    Returns:
        pd.DataFrame: Copy of `data` with outliers dropped.
    """
    outliers = iqr_outliers(data)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers.any(axis=1)].copy()


@iqr_drop.register
def _(data: pd.Series, silent=False) -> pd.Series:
    """Function for Series"""
    outliers = iqr_outliers(data)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers].copy()


@singledispatch
def z_outliers(data: pd.DataFrame, thresh=3) -> pd.DataFrame:
    """Returns boolean mask of z-score outliers.

    Args:
        data (pd.DataFrame): Series or DataFrame for finding outliers.
        thresh (int, optional): Z-score threshold for outliers. Defaults to 3.

    Returns:
        pd.DataFrame: Series or DataFrame mask
    """
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh=3) -> pd.Series:
    """Function for Series"""
    return z_outliers(data.to_frame(), thresh=thresh).squeeze()


@singledispatch
def z_clip(data: pd.DataFrame, thresh=3, silent=False) -> pd.DataFrame:
    """Move z-score outliers to z-score `thresh`.

    Args:
        data (pd.DataFrame): Series or DataFrame for clipping outliers.
        thresh (int, optional): Z-score threshold for outliers. Defaults to 3.
        silent (bool, optional): Do not display report. Defaults to False.

    Returns:
        pd.DataFrame: Copy of Series or DataFrame with outliers clipped.
    """
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    clipped = np.clip(z_data, -1 * thresh, thresh)
    clipped = ss.inverse_transform(clipped)
    clipped = pd.DataFrame(clipped, index=data.index, columns=data.columns)
    if not silent:
        _display_report(z_outliers(data), "clipped")
    return clipped


@z_clip.register
def _(data: pd.Series, thresh=3, silent=False) -> pd.Series:
    """Function for Series"""
    clipped = z_clip(data.to_frame(), thresh=thresh, silent=True).squeeze()
    if not silent:
        _display_report(z_outliers(data), "clipped")
    return clipped


@singledispatch
def z_drop(data: pd.DataFrame, thresh=3, silent=False) -> pd.DataFrame:
    """Drop z-score outliers from `data`.

    Args:
        data (pd.DataFrame): Series or DataFrame for removing outliers.
        thresh (int, optional): Z-score threshold for outliers. Defaults to 3.
        silent (bool, optional): Do not display report. Defaults to False.

    Returns:
        pd.DataFrame: Copy of Series or DataFrame with outliers dropped.
    """
    outliers = z_outliers(data, thresh=thresh)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers.any(axis=1)].copy()


@z_drop.register
def _(data: pd.Series, thresh=3, silent=False) -> pd.Series:
    """Function for Series"""
    outliers = z_outliers(data, thresh=thresh)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers].copy()
