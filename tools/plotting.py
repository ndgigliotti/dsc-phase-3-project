import re
from functools import singledispatch
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import minmax_scale
from matplotlib import ticker

import tools.utils as utils
import tools.outliers as outliers

HEATMAP_STYLE = MappingProxyType(
    {
        "square": True,
        "annot": True,
        "fmt": ".2f",
        "cbar": False,
        "center": 0,
        "cmap": sns.color_palette("coolwarm", n_colors=100, desat=0.6),
        "linewidths": 0.1,
        "linecolor": "k",
        "annot_kws": MappingProxyType({"fontsize": 8})
    }
)

_rng = np.random.default_rng(31)


def _format_big_number(num, dec):
    abb = ""
    if num != 0:
        mag = np.log10(np.abs(num))
        if mag >= 12:
            num = num / 10 ** 12
            abb = "T"
        elif mag >= 9:
            num = num / 10 ** 9
            abb = "B"
        elif mag >= 6:
            num = num / 10 ** 6
            abb = "M"
        elif mag >= 3:
            num = num / 10 ** 3
            abb = "K"
        num = round(num, dec)
    return f"{num:,.{dec}f}{abb}"


def big_number_formatter(dec=0):
    @ticker.FuncFormatter
    def formatter(num, pos):
        return _format_big_number(num, dec)

    return formatter


def big_money_formatter(dec=0):
    @ticker.FuncFormatter
    def formatter(num, pos):
        return f"${_format_big_number(num, dec)}"

    return formatter


def figsize_like(data: pd.DataFrame, scale: float = 0.85) -> np.ndarray:
    """Calculate figure size based on the shape of data.

    Args:
        data (pd.DataFrame): Ndarray, Series, or Dataframe for figsize.
        scale (float, optional): Scale multiplier for figsize. Defaults to 0.85.

    Returns:
        [np.ndarray]: array([width, height]).
    """
    return (np.array(data.shape)[::-1] * scale)


def add_tukey_marks(
    data: pd.Series,
    ax: plt.Axes,
    iqr_color: str = "r",
    fence_color: str = "k",
    fence_style: str = "--",
    show_quarts: bool = False,
) -> plt.Axes:
    """Add IQR box and fences to a histogram-like plot.

    Args:
        data (pd.Series): Data for calculating IQR and fences.
        ax (plt.Axes): Axes to annotate.
        iqr_color (str, optional): Color of shaded IQR box. Defaults to "r".
        fence_color (str, optional): Fence line color. Defaults to "k".
        fence_style (str, optional): Fence line style. Defaults to "--".
        show_quarts (bool, optional): Annotate Q1 and Q3. Defaults to False.

    Returns:
        plt.Axes: Annotated Axes object.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ax.axvspan(q1, q3, color=iqr_color, alpha=0.2)
    iqr_mp = q1 + ((q3 - q1) / 2)
    lower, upper = outliers.iqr_fences(data)
    ax.axvline(lower, c=fence_color, ls=fence_style)
    ax.axvline(upper, c=fence_color, ls=fence_style)
    text_yval = ax.get_ylim()[1]
    text_yval *= 1.01
    ax.text(iqr_mp, text_yval, "IQR", ha="center")
    if show_quarts:
        ax.text(q1, text_yval, "Q1", ha="center")
        ax.text(q3, text_yval, "Q3", ha="center")
    ax.text(upper, text_yval, "Fence", ha="center")
    ax.text(lower, text_yval, "Fence", ha="center")
    return ax

def add_quantile_marks(
    data: np.ndarray,
    quantiles: list,
    ax: plt.Axes,
    line_color: str = "k",
    line_style: str = "--",
    percent_fmt: bool= True,
) -> plt.Axes:
    quant_labels = quantiles
    quantiles = np.quantile(data, quant_labels)
    text_yval = ax.get_ylim()[1]
    text_yval *= 1.01
    quant_labels = np.asarray(quant_labels) * 100
    quant_labels = quant_labels.round().astype(np.int64)
    for quant, label in zip(quantiles, quant_labels):
        ax.axvline(quant, c=line_color, ls=line_style)
        label = f"{label}%" if percent_fmt else str(label)
        ax.text(quant, text_yval, label, ha="center")
    return ax

@singledispatch
def rotate_ticks(ax: plt.Axes, deg: float, axis: str = "x"):
    get_labels = getattr(ax, f"get_{axis}ticklabels")
    for label in get_labels():
        label.set_rotation(deg)


@rotate_ticks.register
def _(ax: np.ndarray, deg: float, axis: str = "x"):
    axs = ax
    for ax in axs:
        rotate_ticks(ax, deg=deg, axis=axis)


def pair_corr_heatmap(
    data, ignore=None, annot=True, high_corr=None, scale=0.5, ax=None, **kwargs
):
    if not ignore:
        ignore = []
    corr_df = data.drop(columns=ignore).corr()
    title = "Correlations Between Features"
    if ax is None:
        figsize = figsize_like(corr_df, scale)
        fig, ax = plt.subplots(figsize=figsize)
    if high_corr is not None:
        if annot:
            annot = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    mask = np.triu(np.ones_like(corr_df, dtype="int64"), k=0)
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    style.update({"annot": annot})
    ax = sns.heatmap(
        data=corr_df,
        mask=mask,
        ax=ax,
        **style,
    )
    ax.set_title(title, pad=10)
    return ax


def calc_subplots_size(nplots: int, ncols: int, sp_height: int) -> tuple:
    """Calculate number of rows and figsize for subplots.

    Args:
        nplots (int): Number of subplots.
        ncols (int): Number of columns in figure.
        sp_height (int): Height of each subplot.

    Returns:
        [tuple]: Tuple containing:
                nrows (int): Number of rows in figure.
                figsize (tuple): (width, height)

    """
    nrows = round(nplots / ncols)
    figsize = (ncols * sp_height, nrows * sp_height)
    return nrows, figsize


def multi_dist(data: pd.DataFrame, ncols=3, sp_height=5, **kwargs) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    nrows, figsize = calc_subplots_size(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        ax = sns.histplot(data=data, x=column, ax=ax, **kwargs)
        ax.set_title(f"Distribution of `{column}`")
    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    return axs


def multi_scatter(
    data: pd.DataFrame,
    target: str,
    ncols=3,
    sp_height=5,
    reflexive=False,
    yformatter=None,
    **kwargs,
) -> np.ndarray:
    data = data.select_dtypes(include="number")
    target_data = data.loc[:, target]
    if not reflexive:
        data.drop(columns=target, inplace=True)
    nrows, figsize = calc_subplots_size(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        ax = sns.scatterplot(x=data[column], y=target_data, ax=ax, **kwargs)
        ax.set_ylabel(target, labelpad=10)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    return axs


def linearity_scatters(
    data: pd.DataFrame, target: str, ncols=3, sp_height=5, yformatter=None, **kwargs
) -> plt.Figure:
    data = data.loc[:, utils.numeric_cols(data)]
    corr_df = data.corrwith(data[target]).round(2)
    nrows, figsize = calc_subplots_size(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        ax = sns.scatterplot(data=data, x=column, y=target, ax=ax, **kwargs)
        text = f"r={corr_df[column]:.2f}"
        ax.text(
            0.975,
            1.025,
            text,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    fig.tight_layout()
    return fig


def multi_joint(
    data: pd.DataFrame, target: str, reflexive: bool = False, **kwargs
) -> np.ndarray:
    data = data.select_dtypes(include="number")
    grids = []
    columns = data.columns if reflexive else data.columns.drop(target)
    for column in columns:
        g = sns.jointplot(data=data, x=column, y=target, **kwargs)
        g.fig.suptitle(f"{column} vs. {target}")
        g.fig.subplots_adjust(top=0.9)
        grids.append(g)
    return np.array(grids)


def annot_bars(
    ax: plt.Axes,
    dist: float = 0.15,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
    **kwargs,
) -> plt.Axes:
    """Annotate a bar graph with the bar values.

    Args:
        ax (plt.Axes): Axes object to annotate.
        dist (float, optional): Distance from ends as fraction of max bar. Defaults to 0.15.
        color (str, optional): Text color. Defaults to "k".
        compact (bool, optional): Annotate inside the bars. Defaults to False.
        orient (str, optional): Bar orientation. Defaults to "h".
        format_spec (str, optional): Format string for annotations. Defaults to "{x:.2f}".
        fontsize (int, optional): Font size. Defaults to 12.
        alpha (float, optional): Opacity of text. Defaults to 0.5.
        drop_last (int, optional): Number of bars to ignore on tail end. Defaults to 0.

    Raises:
        ValueError: `orient` only accepts 'h' or 'v'.

    Returns:
        plt.Axes: Annotated axes object.
    """
    if not compact:
        dist = -dist

    xb = np.array(ax.get_xbound()) * (1 + abs(2 * dist))
    ax.set_xbound(*xb)

    max_bar = np.abs([b.get_width() for b in ax.patches]).max()
    dist = dist * max_bar
    for bar in ax.patches[: -drop_last or len(ax.patches)]:
        if orient.lower() == "h":
            x = bar.get_width()
            x = x + dist if x < 0 else x - dist
            y = bar.get_y() + bar.get_height() / 2
        elif orient.lower() == "v":
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            y = y + dist if y < 0 else y - dist
        else:
            raise ValueError("`orient` must be 'h' or 'v'")

        text = format_spec.format(x=bar.get_width())
        ax.annotate(
            text,
            (x, y),
            ha="center",
            va="center",
            c=color,
            fontsize=fontsize,
            alpha=alpha,
            **kwargs,
        )
    return ax


def heated_barplot(
    data: pd.Series,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    figsize: tuple = (6, 8),
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a sharply divided barplot ranking positive and negative values.

    Args:
        data (pd.Series): Data to plot.
        heat (str): Name of color palette to be passed to Seaborn.
        heat_desat (float, optional): Saturation of color palette. Defaults to 0.6.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        plt.Axes: Axes for the plot.
    """
    data.index = data.index.astype(str)
    data.sort_values(ascending=False, inplace=True)
    heat = pd.Series(
        sns.color_palette(heat, desat=heat_desat, n_colors=201),
        index=pd.RangeIndex(-100, 101),
    )
    pal_vals = np.around(minmax_scale(data, feature_range=(-100, 100))).astype(np.int64)
    palette = heat.loc[pal_vals]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(
        x=data.values, y=data.index, palette=palette, orient="h", ax=ax, **kwargs
    )
    ax.axvline(0.0, color="k", lw=1, ls="-", alpha=0.33)
    return ax


def diagnostics(
    model,
    height=5,
    xformatter=big_number_formatter(2),
    yformatter=big_number_formatter(2),
):
    fig, (qq, hs) = plt.subplots(ncols=2, figsize=(height * 2, height))
    sm.graphics.qqplot(model.resid, fit=True, line="45", ax=qq)
    qq.set_title("Normality of Residuals")
    hs = sns.scatterplot(x=model.predict(), y=model.resid, s=5)
    hs.set_ylabel("Residuals", labelpad=10)
    hs.set_xlabel("Predicted Values", labelpad=10)
    # hs.yaxis.set_major_formatter(yformatter)
    # hs.xaxis.set_major_formatter(xformatter)
    for label in hs.get_xticklabels():
        label.set_rotation(45)
    hs.set_title("Homoscedasticity Check")
    fig.tight_layout()
    return fig


def cat_palette(
    name: str, keys: list, shuffle: bool = False, offset: int = 0, **kwargs
) -> dict:
    """Create a color palette dictionary for a categorical variable.

    Args:
        name (str): Color palette name to be passed to Seaborn.
        keys (list): Keys for mapping to colors.
        shuffle (bool, optional): Shuffle the palette. Defaults to False.
        offset (int, optional): Number of initial colors to skip over. Defaults to 0.

    Returns:
        dict: Categorical-style color mapping.
    """
    n_colors = len(keys) + offset
    pal = sns.color_palette(name, n_colors=n_colors, **kwargs)[offset:]
    if shuffle:
        _rng.shuffle(pal)
    return dict(zip(keys, pal))


def derive_coeff_labels(coeff_df):
    re_cat = r"C\(\w+\)\[T\.([\w\s]+)\]"
    label = coeff_df.index.to_series(name="label")
    cat_label = label.filter(regex=re_cat, axis=0)
    label.update(cat_label.str.extract(re_cat).squeeze())
    return coeff_df.assign(label=label)


def simple_barplot(
    data, x, y, sort="asc", orient="v", estimator=np.mean, scale=0.5, ax=None, **kwargs
):
    if ax is None:
        width = data[x].nunique()
        width *= scale
        height = width / 2
        figsize = (width, height) if orient == "v" else (height, width)
        figsize = np.array(figsize).round().astype(np.int64)
        fig, ax = plt.subplots(figsize=figsize)
    if sort:
        if sort.lower() in ("asc", "desc"):
            asc = sort.lower() == "asc"
        else:
            raise ValueError("`sort` must be 'asc', 'desc', or None")
        order = data.groupby(x)[y].agg(estimator)
        order = order.sort_values(ascending=asc).index.to_list()
    else:
        order = None

    titles = {
        "y": utils.to_title(y),
        "x": utils.to_title(x),
        "est": utils.to_title(estimator.__name__),
    }

    if orient.lower() == "h":
        x, y = y, x
    elif orient.lower() != "v":
        raise ValueError("`orient` must be 'v' or 'h'")
    ax = sns.barplot(
        data=data,
        x=x,
        y=y,
        estimator=estimator,
        orient=orient,
        order=order,
        ax=ax,
        **kwargs,
    )

    ax.set_title("{est} {y} by {x}".format(**titles), pad=10)
    ax.set_xlabel(titles["x" if orient.lower() == "v" else "y"], labelpad=10)
    ax.set_ylabel(titles["y" if orient.lower() == "v" else "x"], labelpad=15)
    return ax


# def cat_regressor_barplots(
#     main_df,
#     coeff_df,
#     exog,
#     endog,
#     sp_height=5,
#     plot_corr=True,
#     palette=None,
#     saturation=0.75,
#     annot_kws=None,
#     corr_kws=None,
#     estimator=np.median,
# ):
#     if "label" not in coeff_df.columns:
#         coeff_df = derive_coeff_labels(coeff_df)
#     if plot_corr:
#         _, figsize = calc_subplots_size(3, 3, sp_height)
#         fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
#     else:
#         _, figsize = calc_subplots_size(2, 2, sp_height)
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

#     uniq_exog = main_df[exog].sort_values().unique()
#     if not palette:
#         palette = cat_palette(None, uniq_exog)
#     if isinstance(palette, str):
#         palette = cat_palette(palette, uniq_exog)
#     coeff_df = coeff_df.filter(like=exog, axis=0)
#     coeff_df = coeff_df.assign(label=coeff_df.label.astype(uniq_exog.dtype))
#     coeff_df.sort_values("label", inplace=True)

#     ax1 = sns.barplot(
#         data=coeff_df,
#         x="label",
#         y="coeff",
#         palette=palette,
#         saturation=saturation,
#         order=coeff_df.label,
#         ax=ax1,
#     )
#     ax2 = sns.barplot(
#         data=main_df,
#         x=exog,
#         y=endog,
#         estimator=estimator,
#         palette=palette,
#         saturation=saturation,
#         ax=ax2,
#     )

#     ax1.set_ylabel(f"Effect on {endog.title()}", labelpad=10)
#     ax2.set_ylabel(endog.title(), labelpad=10)
#     ax1.set_title(f"Projected Effects of {exog.title()} on {endog.title()}", pad=10)
#     est_name = estimator.__name__.title()
#     ax2.set_title(f"{est_name} {endog.title()} by {exog.title()}", pad=10)
#     for ax in (ax1, ax2):
#         ax.set_xlabel(exog.title())

#     if plot_corr:
#         if not corr_kws:
#             corr_kws = dict()
#         ax3 = heated_barplot(
#             pd.get_dummies(main_df[exog]).corrwith(main_df[endog]),
#             saturation=saturation,
#             ax=ax3,
#             **corr_kws,
#         )
#         default_annot_kws = {"color": "k", "dist": 0.2, "fontsize": 11}
#         if annot_kws:
#             default_annot_kws.update(annot_kws)
#         ax3 = annot_bars(ax3, **default_annot_kws)
#         ax3.set_title(f"Correlation: {exog.title()} and {endog.title()}")
#         ax3.set_xlabel("Correlation", labelpad=10)
#         ax3.set_ylabel(exog.title(), labelpad=10)
#     fig.tight_layout()
#     return fig

def cat_line_and_corr(
    main_df,
    exog,
    endog,
    sp_height=5,
    lw=3,
    ms=10,
    marker="o",
    palette=None,
    annot_kws=None,
    corr_kws=None,
    estimator=np.median,
):
    _, figsize = calc_subplots_size(2, 2, sp_height)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

    ax1 = sns.lineplot(
        data=main_df,
        x=exog,
        y=endog,
        estimator=estimator,
        palette=palette,
        lw=lw,
        ms=ms,
        marker=marker,
        ax=ax1,
    )

    ax1.set_ylabel(endog.title(), labelpad=10)
    ax1.set_xlabel(exog.title(), labelpad=10)
    est_name = estimator.__name__.title()
    ax1.set_title(f"{est_name} {endog.title()} by {exog.title()}", pad=10)

    if not corr_kws:
        corr_kws = dict()
    ax2 = heated_barplot(
        pd.get_dummies(main_df[exog]).corrwith(main_df[endog]),
        ax=ax2,
        **corr_kws,
    )
    default_annot_kws = {"color": "k", "dist": 0.2, "fontsize": 11}
    if annot_kws:
        default_annot_kws.update(annot_kws)
    ax2 = annot_bars(ax2, **default_annot_kws)
    ax2.set_title(f"Correlation: {exog.title()} and {endog.title()}")
    ax2.set_xlabel("Correlation", labelpad=10)
    ax2.set_ylabel(exog.title(), labelpad=10)
    fig.tight_layout()
    return fig

def cat_regressor_lineplots(
    main_df,
    coeff_df,
    exog,
    endog,
    sp_height=5,
    lw=3,
    ms=10,
    marker="o",
    plot_corr=True,
    palette=None,
    annot_kws=None,
    corr_kws=None,
    estimator=np.median,
):
    if "label" not in coeff_df.columns:
        coeff_df = derive_coeff_labels(coeff_df)
    if plot_corr:
        _, figsize = calc_subplots_size(3, 3, sp_height)
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
    else:
        _, figsize = calc_subplots_size(2, 2, sp_height)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

    coeff_df = coeff_df.filter(like=exog, axis=0)
    coeff_df = coeff_df.assign(label=coeff_df.label.astype(main_df[exog].dtype))
    coeff_df.sort_values("label", inplace=True)

    ax1 = sns.lineplot(
        data=coeff_df,
        x="label",
        y="coeff",
        palette=palette,
        lw=lw,
        ms=ms,
        marker=marker,
        ax=ax1,
    )
    ax2 = sns.lineplot(
        data=main_df,
        x=exog,
        y=endog,
        estimator=estimator,
        palette=palette,
        lw=lw,
        ms=ms,
        marker=marker,
        ax=ax2,
    )

    ax1.set_ylabel(f"Effect on {endog.title()}", labelpad=10)
    ax2.set_ylabel(endog.title(), labelpad=10)
    ax1.set_title(f"Average Effect of {exog.title()} on {endog.title()}", pad=10)
    est_name = estimator.__name__.title()
    ax2.set_title(f"{est_name} {endog.title()} by {exog.title()}", pad=10)
    for ax in (ax1, ax2):
        ax.set_xlabel(exog.title(), labelpad=10)

    if plot_corr:
        if not corr_kws:
            corr_kws = dict()
        ax3 = heated_barplot(
            pd.get_dummies(main_df[exog]).corrwith(main_df[endog]),
            ax=ax3,
            **corr_kws,
        )
        default_annot_kws = {"color": "k", "dist": 0.2, "fontsize": 11}
        if annot_kws:
            default_annot_kws.update(annot_kws)
        ax3 = annot_bars(ax3, **default_annot_kws)
        ax3.set_title(f"Correlation: {exog.title()} and {endog.title()}")
        ax3.set_xlabel("Correlation", labelpad=10)
        ax3.set_ylabel(exog.title(), labelpad=10)
    fig.tight_layout()
    return fig


def cat_corr_heatmap(
    data: pd.DataFrame,
    categorical: str,
    transpose:bool=False,
    high_corr:float=None,
    scale: float = 0.5,
    no_prefix: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a correlation heatmap of categorical vs. numeric features.

    Args:
        data (pd.DataFrame): Frame containing categorical and numeric data.
        categorical (str): Name or list of names of categorical features.
        high_corr (float): Threshold for high correlation. Defaults to None.
        scale (float, optional): Multiplier for determining figsize. Defaults to 0.5.
        no_prefix (bool, optional): If only one cat, do not prefix dummies. Defaults to True.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        plt.Axes: Axes of the plot.
    """
    if isinstance(categorical, str):
        ylabel = utils.to_title(categorical)
        categorical = [categorical]
        single_cat = True
    else:
        ylabel = "Categorical Features"
        single_cat = False
    title = "Correlation with Numeric Features"
    cat_df = data.filter(categorical, axis=1)
    if no_prefix and single_cat:
        dummies = pd.get_dummies(cat_df, prefix="", prefix_sep="")
    else:
        dummies = pd.get_dummies(cat_df)
    corr_df = dummies.apply(lambda x: data.corrwith(x))
    if not transpose:
        corr_df = corr_df.T
    if high_corr is not None:
        if "annot" not in kwargs or kwargs.get("annot"):
            kwargs["annot"] = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_like(corr_df, scale=scale))
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    ax = sns.heatmap(corr_df, ax=ax, **style)
    xlabel = "Numeric Features"
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=10)
    return ax