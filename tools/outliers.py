from functools import singledispatch

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import scale


@singledispatch
def _display_report(outliers: pd.DataFrame, verb: str) -> None:
    """Display report of modified observations.

    Parameters
    ----------
    outliers : Series or DataFrame
        Boolean mask which marks outliers as True.
    verb : str
        Outlier adjustment verb (past tense), e.g. 'trimmed'.
    """
    report = outliers.sum()
    n_modified = outliers.any(axis=1).sum()
    report["total_obs"] = n_modified
    report = report.to_frame(f"n_{verb}")
    report[f"pct_{verb}"] = (report.squeeze() / outliers.shape[0]) * 100
    display(report)


@_display_report.register
def _(outliers: pd.Series, verb: str) -> None:
    """Process Series"""
    # simply convert Series to DataFrame
    _display_report(outliers.to_frame(), verb)


@singledispatch
def winsorize(data: pd.Series, outliers: pd.Series) -> pd.Series:
    """Reset outliers to outermost inlying values.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    outliers : Series or DataFrame
        Boolean mask of outliers.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    min_in, max_in = data[~outliers].agg(["min", "max"])
    data = data.clip(lower=min_in, upper=max_in)
    _display_report(outliers, "winsorized")
    return data


@winsorize.register
def _(data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    bounds = data.mask(outliers).agg(["min", "max"]).T
    data = data.clip(lower=bounds["min"], upper=bounds["max"], axis=1)
    _display_report(outliers, "winsorized")
    return data


@singledispatch
def trim(data: pd.Series, outliers: pd.Series) -> pd.Series:
    """Remove outliers from data.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    outliers : Series or DataFrame
        Boolean mask of outliers.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    data = data.loc[~outliers].copy()
    _display_report(outliers, "trimmed")
    return data


@trim.register
def _(data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    data = data.loc[~outliers.any(axis=1)].copy()
    _display_report(outliers, "trimmed")
    return data


def tukey_fences(data: pd.Series) -> tuple:
    """Get the lower and upper Tukey fences.

    Tukey's fences are located at 1.5 * IQR from the IQR on either side.

    Parameters
    ----------
    data : Series
        Distribution for calculating fences.

    Returns
    -------
    lower : float
        Lower Tukey fence.
    upper : float
        Upper Tukey fence.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


@singledispatch
def tukey_outliers(data: pd.Series) -> pd.Series:
    """Returns boolean mask of Tukey-fence outliers.

    Inliers are anything between Tukey's fences (inclusive).
    Tukey's fences are located at 1.5 * IQR from the IQR on either side.
    Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    lower, upper = tukey_fences(data)
    return (data < lower) | (data > upper)


@tukey_outliers.register
def _(data: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrames"""
    # simply map Series function across DataFrame
    return data.apply(tukey_outliers)


@singledispatch
def z_outliers(data: pd.DataFrame, thresh: int = 3) -> pd.DataFrame:
    """Returns boolean mask of z-score outliers.

    Inliers are anything with an absolute z-score less than
    or equal to `thresh`. Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    z_data = scale(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh: int = 3) -> pd.Series:
    """Process Series"""
    # convert to DataFrame and then squeeze back into Series
    return z_outliers(data.to_frame(), thresh=thresh).squeeze()


@singledispatch
def quantile_outliers(
    data: pd.Series,
    lower: float = 0.05,
    upper: float = 0.95,
    interpolation: str = "linear",
) -> pd.Series:
    """Returns boolean mask of quantile-determined outliers.

    Inliers are anything between the `lower` and `upper` quantiles,
    inclusive. Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.
    lower : float, optional
        Lower quantile boundary, by default 0.05.
    upper : float, optional
        Upper quantile boundary, by default 0.95.
    interpolation : str, optional
        Method to use when quantile lies between two data points,
        by default "linear". See the Pandas documentation for
        Series.quantile().

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    lower, upper = data.quantile([lower, upper], interpolation=interpolation)
    inliers = data.between(lower, upper, inclusive=True) | data.isna()
    return ~inliers


@quantile_outliers.register
def _(
    data: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Process DataFrames"""
    kwargs = dict(lower=lower, upper=upper, interpolation=interpolation)
    # simply map Series function across DataFrame
    return data.apply(quantile_outliers, **kwargs)


def tukey_winsorize(data: pd.DataFrame) -> pd.DataFrame:
    """Reset outliers to outermost values within Tukey fences.

    For DataFrames, outliers are Winsorized independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = tukey_outliers(data)
    return winsorize(data, outliers)


def tukey_trim(data: pd.DataFrame) -> pd.DataFrame:
    """Remove observations beyond the Tukey fences.

    For DataFrames, outliers are found independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = tukey_outliers(data)
    return trim(data, outliers)


def z_winsorize(data: pd.DataFrame, thresh: int = 3) -> pd.DataFrame:
    """Reset outliers to outermost values within z-score threshold.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = z_outliers(data, thresh=thresh)
    return winsorize(data, outliers)


def z_trim(data: pd.DataFrame, thresh: int = 3) -> pd.DataFrame:
    """Remove observations beyond the z-score threshold.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = z_outliers(data, thresh=thresh)
    return trim(data, outliers)


def quantile_winsorize(
    data: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Reset outliers to outermost values within quantile bounds.

    The `lower` and `upper` quantiles mark the boundaries for inliers
    (inclusive). For DataFrames, outliers are determined independently for
    each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    lower : float, optional
        Lower quantile boundary, by default 0.05.
    upper : float, optional
        Upper quantile boundary, by default 0.95.
    interpolation : str, optional
        Method to use when quantile lies between two data points,
        by default "linear". See the Pandas documentation for
        Series.quantile().

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    kwargs = dict(lower=lower, upper=upper, interpolation=interpolation)
    outliers = quantile_outliers(data, **kwargs)
    return winsorize(data, outliers)


def quantile_trim(
    data: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Remove observations beyond the `lower` and `upper` quantiles.

    The `lower` and `upper` quantiles mark the boundaries for inliers
    (inclusive). For DataFrames, outliers are determined independently for
    each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    lower : float, optional
        Lower quantile boundary, by default 0.05.
    upper : float, optional
        Upper quantile boundary, by default 0.95.
    interpolation : str, optional
        Method to use when quantile lies between two data points,
        by default "linear". See the Pandas documentation for
        Series.quantile().

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    kwargs = dict(lower=lower, upper=upper, interpolation=interpolation)
    outliers = quantile_outliers(data, **kwargs)
    return trim(data, outliers)