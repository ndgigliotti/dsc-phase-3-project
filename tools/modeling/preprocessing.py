import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# class NullTrimmer(TransformerMixin, BaseEstimator):
#     def __init__(self, *, trim_features=True, feature_thresh=0.5, copy=True):
#         self.trim_features = trim_features
#         self.feature_thresh = feature_thresh
#         self.copy = copy

#     def fit(self, X, y=None):
#         self.missing_mask_ = np.isnan(X)
#         return self

#     @property
#     def incomplete_rows(self):
#         check_is_fitted(self)
#         return self.missing_mask_.any(axis=1)

#     def transform(self, X, copy=None):
#         check_is_fitted(self)

#         copy = copy if copy is not None else self.copy
#         if copy:
#             X = X.copy()

#         self.trimmed_rows_ = X[self.incomplete_rows]
#         return X[~self.incomplete_rows]

#     def inverse_transform(self, X, copy=None):
#         check_is_fitted(self)
#         copy = copy if copy is not None else self.copy
#         if copy:
#             X = X.copy()
#         X[self.incomplete_rows_] = self.trimmed_rows_
#         return X
