from typing import List

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

from .base import LogMixin, ColumnNameHandler


class BaseEstimator(BaseEstimator, RegressorMixin, LogMixin, ColumnNameHandler):
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def fit(self, x: pd.DataFrame, y=None) -> 'BaseEstimator':
        return self
    
    def predict(self, x: pd.DataFrame) -> List[float]:
        raise NotImplementedError
    
    @property
    def resid(self) -> np.ndarray:
        raise NotImplementedError
    

class RegReturnEstimator(BaseEstimator):
    def __init__(self, feature_cols: List[str]):
        self.estimator = None
        self.feature_cols = feature_cols if isinstance(feature_cols, List) else [feature_cols]
    
    def fit(self, x: DataFrame, y=None) -> BaseEstimator:
        self.estimator = sm.OLS(
            x[self.ret_col], 
            sm.add_constant(x[self.feature_cols])
        ).fit()
        return self
    
    def predict(self, x: DataFrame) -> List[float]:
        return self.estimator.predict(sm.add_constant(x[self.feature_cols])).tolist()
    
    @property
    def resid(self) -> np.ndarray:
        if self.estimator:
            return self.estimator.resid
        return
    

class MAMReturnEstimator(BaseEstimator):
    def __init__(self):
        self.__resid = None
    
    @property
    def resid(self) -> float:
        return self.__resid

    def fit(self, x: pd.DataFrame, y=None) -> 'MAMReturnEstimator':
        self.__resid = (x[self.ret_col] - x[self.mkt_rf_col] - x[self.rf_col])**2
        return self
    
    def predict(self, x: pd.DataFrame) -> List[float]:
        return (x[self.mkt_rf_col] + x[self.rf_col]).tolist()
