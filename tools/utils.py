import datetime
from time import perf_counter
from collections.abc import Mapping
import numpy as np
import pandas as pd

NULL = frozenset([np.nan, pd.NA, None])
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"


def numeric_cols(data: pd.DataFrame) -> list:
    """Returns a list of all numeric column names.

    Args:
        data (pd.DataFrame): DataFrame to get column names from.

    Returns:
        list: All numeric column names.
    """
    numeric = data.dtypes.map(pd.api.types.is_numeric_dtype)
    return data.columns[numeric].to_list()


def noncat_cols(data: pd.DataFrame) -> list:
    categorical = data.dtypes.map(pd.api.types.is_categorical_dtype)
    return data.columns[~categorical].to_list()


def cat_cols(data: pd.DataFrame) -> list:
    categorical = data.dtypes.map(pd.api.types.is_categorical_dtype)
    return data.columns[categorical].to_list()


def transform(data: pd.DataFrame, pipe: list):
    tr = data.to_numpy()
    for func in pipe:
        tr = func(tr)
    display([x.__name__ for x in pipe])
    return pd.DataFrame(tr, index=data.index, columns=data.columns)


def filter_pipe(
    data: pd.DataFrame, like: list = None, regex: list = None, axis: int = None
) -> pd.DataFrame:
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


def get_groups(groupby: pd.core.groupby.DataFrameGroupBy):
    return {x: groupby.get_group(x) for x in groupby.groups}


def elapsed(start_time):
    return datetime.timedelta(seconds=perf_counter() - start_time)

def to_title(pylabel):
    return pylabel.replace("_", " ").strip().title()

def cartesian(*xi):
    return np.array(np.meshgrid(*xi)).T.reshape(-1, len(xi))
# def map_list_likes(data: pd.Series, mapper: dict):
#     """Apply `mapper` to elements of elements of `data`.

#     Args:
#         data (pd.Series): Series containing only list-like elements.
#         mapper (dict): Dict-like or callable to apply to elements of elements of `data`.
#     """

#     def transform(list_):
#         if isinstance(mapper, Mapping):
#             return [mapper[x] if x not in NULL else x for x in list_]
#         else:
#             return [mapper(x) if x not in NULL else x for x in list_]

#     return data.map(transform, na_action="ignore")


# def datetime_from_name(name):
#     name = os.path.basename(name)
#     root, _ = os.path.splitext(name)
#     fmt = DATETIME_FORMAT if root.count("-") == 4 else DATE_FORMAT
#     return datetime.datetime.strptime(root, fmt)


# def datetime_to_name(when, ext=None):
#     if isinstance(when, datetime.datetime):
#         name = when.isoformat(timespec="seconds").replace(":", "-")
#     elif isinstance(when, datetime.date):
#         name = when.isoformat()
#     else:
#         raise ValueError("'when' must be datetime.datetime or datetime.date")
#     if ext:
#         if not ext.startswith("."):
#             ext = "." + ext
#         name += ext
#     return name


# def date_from_name(name):
#     return datetime_from_name(name).date()


# def date_to_name(when, ext=None):
#     return datetime_to_name(when, ext=ext)


# def now_name(ext=None):
#     return datetime_to_name(datetime.datetime.now(), ext=ext)


# def today_name(ext=None):
#     return date_to_name(datetime.date.today(), ext=ext)