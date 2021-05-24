from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


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


def class_report(estimator, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4));
    metrics.plot_confusion_matrix(
        estimator, X_test, y_test, cmap="Blues", normalize="true", ax=ax1
    )
    metrics.plot_roc_curve(estimator, X_test, y_test, ax=ax2)
    ax2.plot([0, 1], [0, 1], color="r", lw=2, linestyle=":", alpha=1)
    sk_report = metrics.classification_report(y_test, estimator.predict(X_test))
    fig.tight_layout()
    print(sk_report)