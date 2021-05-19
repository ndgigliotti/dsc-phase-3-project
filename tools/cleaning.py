import re
import json
import unidecode
from IPython.display import display, HTML
from string import punctuation
import pandas as pd
import numpy as np
import tools.utils as utils

RE_PUNCT = re.compile(f"[{re.escape(punctuation)}]")
RE_WHITESPACE = re.compile(r"\s+")


def nan_info(data: pd.DataFrame):
    df = data.isna().sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)


def dup_info(data: pd.DataFrame):
    df = data.duplicated().sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)


def nan_rows(data: pd.DataFrame):
    return data[data.isna().any(axis=1)]


def dup_rows(data: pd.DataFrame, **kwargs):
    return data[data.duplicated(**kwargs)]


def who_is_nan(data: pd.DataFrame, col: str, name_col: str):
    return nan_rows(data)[data[col].isna()][name_col]


def process_strings(strings: pd.Series) -> pd.Series:
    df = strings.str.lower()
    df = df.str.replace(RE_PUNCT, "").str.replace(RE_WHITESPACE, " ")
    df = df.map(unidecode.unidecode, na_action="ignore")
    return df


def detect_json_list(x):
    return isinstance(x, str) and bool(re.fullmatch(r"\[.*\]", x))


def coerce_list_likes(data):
    if not isinstance(data, pd.Series):
        raise TypeError("`data` must be pd.Series")
    json_strs = data.map(detect_json_list, na_action="ignore")
    clean = data.copy()
    clean[json_strs] = clean.loc[json_strs].map(json.loads)
    list_like = clean.map(pd.api.types.is_list_like)
    clean[~list_like] = clean.loc[~list_like].map(lambda x: [x], na_action="ignore")
    clean = clean.map(list, na_action="ignore")
    return clean


def info(data: pd.DataFrame, round_pct: int = 2) -> pd.DataFrame:
    """Get counts of NaNs, uniques, and duplicates.

    Parameters
    ----------
    data : pd.DataFrame
        [description]
    round_pct : int, optional
        [description], by default 2

    Returns
    -------
    pd.DataFrame
        [description]
    """    
    n_rows = data.shape[0]
    nan = data.isna().sum().to_frame("nan")
    dup = pd.DataFrame(
        index=data.columns, data=data.duplicated().sum(), columns=["dup"]
    )
    uniq = data.nunique().to_frame("uniq")
    info = pd.concat([nan, dup, uniq], axis=1)
    pcts = (info / n_rows) * 100
    pcts.columns = pcts.columns.map(lambda x: f"{x}_%")
    pcts = pcts.round(round_pct)
    info = pd.concat([info, pcts], axis=1)
    order = ["nan", "nan_%", "uniq", "uniq_%", "dup", "dup_%"]
    info = info.loc[:, order]
    info.sort_values("nan", ascending=False, inplace=True)
    return info


def show_uniques(data: pd.DataFrame, cut: int = 10, columns: list = None) -> None:
    """Display the unique values for each column of `data`.

    Parameters
    ----------
    data : DataFrame
        DataFrame for viewing unique values.
    cut : int, optional
        Show only columns with this many or fewer uniques, by default 10.
    columns : list, optional
        Columns to show, by default None. Ignores `cut` if specified.
    """
    if columns:
        data = data.loc[:, columns]
    elif cut:
        data = data.loc[:, data.nunique() <= cut]
    cols = [pd.Series(y.dropna().unique(), name=x) for x, y in data.iteritems()]
    table = pd.concat(cols, axis=1).to_html(index=False, na_rep="", notebook=True)
    display(HTML(table))