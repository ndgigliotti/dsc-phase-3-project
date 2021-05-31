import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from .. import utils


def tidy_results(cv_results: dict):
    cv_results = pd.DataFrame(cv_results)
    splits = cv_results.filter(regex=r"split[0-9]+_").columns
    cv_results.drop(columns=splits, inplace=True)
    return cv_results


def make_search_pipe(
    estimator_pipe: Pipeline,
    param_grid: dict,
    kind: str = "grid",
    step_name: str = "search",
    n_jobs: int = None,
    **kwargs,
) -> Pipeline:
    """Construct a parameter search pipeline from another pipeline.

    Creates a pipeline for conducting a parameter search on the
    final estimator of another pipeline. The search pipeline will be
    a clone of the original with the final estimator swapped out
    for a search estimator. The search estimator contains a clone
    of the original final estimator with the search parameters reset.

    Parameters
    ----------
    estimator_pipe : Pipeline
        Pipeline with final estimator to be used in search.
    param_grid : dict
        Parameter grid or distributions for search.
    kind : str, optional
        Search type: "grid" (default) or "randomized".
    step_name : str, optional
        Name of search step in new pipeline, by default "search".
    n_jobs: int, optional
        Number of jobs to run in parallel.
    **kwargs
        Additional keyword arguments for search estimator.

    Returns
    -------
    Pipeline
        Pipeline with search as the final estimator.
    """
    # Clone estimator from `estimator_pipe`
    estimator = clone(estimator_pipe[-1])

    # Reset parameters in `estimator` which are in `param_grid`
    defaults = pd.Series(utils.get_defaults(estimator.__class__))
    to_reset = defaults.loc[defaults.index.isin(param_grid)]
    estimator.set_params(**to_reset)

    # Create search estimator
    if kind.lower() == "grid":
        search = GridSearchCV(estimator, param_grid, n_jobs=n_jobs, **kwargs)
    elif kind.lower() == "randomized":
        search = RandomizedSearchCV(estimator, param_grid, n_jobs=n_jobs, **kwargs)

    # Construct search pipeline with search as last step
    search_pipe = list(clone(estimator_pipe[:-1]).named_steps.items())
    search_pipe.append((step_name, search))
    return Pipeline(search_pipe)
