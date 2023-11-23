from typing import Union, List

import pandas as pd
import numpy as np
import statsmodels.api as sm

from .base import BasePandasRegressor


class BaseEstimator(BasePandasRegressor):
    def __init__(self, feature_cols: Union[str, List[str]], estimation_days: int = 255, 
                 gap_days: int = 50, window_before: int = 10, window_after: int = 10, 
                 min_estimation_days: int = 100):
        self.feature_cols = feature_cols if isinstance(feature_cols, list) else [feature_cols]
        self.estimation_days = estimation_days
        self.gap_days = gap_days
        self.window_before = window_before
        self.window_after = window_after
        self.min_estimation_days = min_estimation_days
        self.model = None

    def fit(self, x: pd.DataFrame, y=None):
        if any(c not in x for c in self.feature_cols):
            raise ValueError(f"Input data does not contain all required columns: {', '.join(self.feature_cols)}")
        
        estimation_period = np.arange(
            -(self.window_before+self.gap_days+self.estimation_days), 
            -(self.window_before+self.gap_days), 
            1)
        train_df = x[x[self.offset_col].isin(estimation_period)].copy()
        if train_df.shape[0] < self.min_estimation_days:
            return self
        self.model = sm.OLS(
            train_df[self.ret_col], 
            sm.add_constant(train_df[self.feature_cols])
        ).fit()
        return self
    
    def predict(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.model is None:
            return None
        event_period = np.arange(-self.window_before, self.window_after + 1, 1)
        event_df = x[
            x[self.offset_col].isin(event_period)
        ].copy()
        # Predict normal returns and calculate other variables
        event_df.sort_values(self.offset_col, inplace=True)
        event_df.loc[:, self.pred_ret_col] = self.model.predict(sm.add_constant(event_df[self.feature_cols]))
        event_df.loc[:, self.ar_col] = event_df[self.ret_col] - event_df[self.pred_ret_col]
        event_df.loc[:, self.car_col] = event_df[self.ar_col].cumsum()
        event_df.loc[:, self.sar_col] = event_df[self.ar_col] / self.model.resid.std()
        event_df.loc[:, self.scar_col] = event_df[self.car_col] / (
            self.model.resid.std() * np.sqrt(event_df.reset_index(drop=True).index+1))
        return event_df


class CAPM(BaseEstimator):
    """
    CAPM estimator - based on regression = alpha + Beta*(Mkt-Rf)
    Parameters
    ----------
    BaseEstimator : _type_
        _description_
    """
    def __init__(self, **kwargs):
        super().__init__(feature_cols=[self.mkt_rf_col], **kwargs)


class FF3(BaseEstimator):
    """FF3 estimator - based on regression = alpha + Beta * (Mkt-Rf) + Beta2 * SMB + Beta3 * HML"""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_cols=[self.mkt_rf_col, self.smb_col, self.hml_col], **kwargs)