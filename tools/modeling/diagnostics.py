from types import MappingProxyType
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ..plotting import HEATMAP_STYLE
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_confusion_matrix,
    classification_report as sk_report,
)


def get_estimator_name(estimator: BaseEstimator):
    if isinstance(estimator, Pipeline):
        name = estimator[-1].__class__.__name__
    else:
        name = estimator.__class__.__name__
    return name


def high_correlations(data: pd.DataFrame, thresh: float = 0.75) -> pd.Series:
    """Get non-trivial feature correlations at or above `thresh`.

    Parameters
    ----------
    data : DataFrame
        Data for finding high correlations.
    thresh : float, optional
        High correlation threshold, by default 0.75.

    Returns
    -------
    Series
        High correlations.
    """
    corr_df = pd.get_dummies(data).corr()
    mask = np.tril(np.ones_like(corr_df, dtype=np.bool_))
    corr_df = corr_df.mask(mask).stack()
    high = corr_df >= thresh
    return corr_df[high]


def pandas_heatmap(
    frame: pd.DataFrame,
    na_rep="",
    precision=3,
    cmap="vlag",
    low=0,
    high=0,
    vmin=None,
    vmax=None,
    axis=None,
):
    table = frame.style.background_gradient(
        cmap=cmap, low=low, high=high, vmin=vmin, vmax=vmax, axis=axis
    )
    table.set_na_rep(na_rep)
    table.set_precision(precision)
    table.highlight_null("white")
    return table


def classification_report(y_test, y_pred, zero_division="warn", heatmap=False):
    report = pd.DataFrame(
        sk_report(y_test, y_pred, output_dict=True, zero_division=zero_division)
    )

    order = report.columns.to_list()[:2] + [
        "macro avg",
        "weighted avg",
        "accuracy",
    ]
    report = report.loc[:, order]

    support = report.loc["support"].iloc[:2]
    support /= report.loc["support", "macro avg"]
    report.loc["support"] = support

    report["bal accuracy"] = balanced_accuracy_score(y_test, y_pred)
    mask = np.array([[0, 1, 1, 1], [0, 1, 1, 1]]).T.astype(np.bool_)
    report[["accuracy", "bal accuracy"]] = report.filter(like="accuracy", axis=1).mask(
        mask
    )

    return pandas_heatmap(report, vmin=0, vmax=1) if heatmap else report


def compare_scores(estimator_1, estimator_2, X_test, y_test, prec=3, heatmap=True):
    scores_1 = classification_report(
        y_test, estimator_1.predict(X_test), precision=prec
    )
    scores_2 = classification_report(
        y_test, estimator_2.predict(X_test), precision=prec
    )
    result = scores_1.compare(scores_2, keep_equal=True, keep_shape=True)
    name_1 = get_estimator_name(estimator_1)
    name_2 = get_estimator_name(estimator_2)
    result.rename(columns=dict(self=name_1, other=name_2), inplace=True)
    result = result.T
    return pandas_heatmap(result) if heatmap else result


def classification_plots(estimator, X_test, y_test):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    plot_confusion_matrix(
        estimator, X_test, y_test, cmap="Blues", normalize="true", ax=ax1
    )
    plot_roc_curve(estimator, X_test, y_test, ax=ax2)
    plot_precision_recall_curve(estimator, X_test, y_test, ax=ax3)

    baseline_style = dict(lw=2, linestyle=":", color="r", alpha=1)
    ax2.plot([0, 1], [0, 1], **baseline_style)
    ax3.plot([0, 1], [y_test.mean()] * 2, **baseline_style)
    ax3.plot([0, 0], [y_test.mean(), 1], **baseline_style)

    y_score = estimator.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_score).round(2)
    ap_score = average_precision_score(y_test, y_score).round(2)

    ax2.set_title(f"Receiver Operating Characteristic Curve: AUC = {auc_score}")
    ax3.set_title(f"Precision-Recall Curve: AP = {ap_score}")
    ax2.get_legend().set_visible(False)
    ax3.get_legend().set_visible(False)
    fig.tight_layout()
    return fig


def standard_report(estimator, X_test, y_test, zero_division="warn"):
    table = classification_report(
        y_test, estimator.predict(X_test), zero_division=zero_division, heatmap=True
    )
    classification_plots(estimator, X_test, y_test)
    display(table)