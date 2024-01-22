import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin


class LogMixin(object):
    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)


@dataclass
class ColumnNameHandler:
    ret_col: str = 'Ret'
    offset_col: str = 'Offset'
    ref_rf_col: str = 'Ret-RF'
    rf_col: str = 'RF'
    mkt_rf_col: str = 'Mkt-RF'
    smb_col: str = 'SMB'
    hml_col: str = 'HML'
    rmw_col: str = 'RMW'
    cma_col: str = 'CMA'
    date_col: str = 'Date'
    ticker_col: str = 'ticker'
    volume_col: str = 'Volume'
    event_date_col: str = 'event_date'
    event_id_col: str = 'event_id'

    # Predicted cols
    pred_ret_col: str = 'pred_ret_col'
    ar_col: str = 'ar'
    car_col: str = 'car'
    sar_col: str = 'sar'
    scar_col: str = 'scar'
    e_std_col: str = 'estimation_std'


class BasePandasRegressor(BaseEstimator, RegressorMixin, LogMixin, ColumnNameHandler):
    def fit(self, x: pd.DataFrame, y=None):
        return self
    def predict(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        raise NotImplementedError
    def fit_predict(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(x, y).predict(x, y)
    