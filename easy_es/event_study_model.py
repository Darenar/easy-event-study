from typing import List

import pandas as pd

from .base import ColumnNameHandler
from .data_loader import load_daily_returns, load_daily_factors
from .utils import expand_date_columns
from .estimation_models import CAPM, FF3, BaseEstimator


class EventStudy(ColumnNameHandler):
    AUX_OFFSET: int = 10
    def __init__(self, estimation_days: int = 255, gap_days: int = 50, 
                 window_before: int = 10, window_after: int = 10, 
                 min_estimation_days: int = 100, estimator_type: str = 'capm'):
        self.estimation_days = estimation_days
        self.gap_days = gap_days
        self.window_before = window_before
        self.window_after = window_after
        self.min_estimation_days = min_estimation_days
        self.returns_df = None
        self.factors_df = None
        self.estimator_type = estimator_type
        self.estimator = self.__initialize_estimator()
    
    def __initialize_estimator(self) -> BaseEstimator:
        study_params = {
            'estimation_days': self.estimation_days,
            'gap_days': self.gap_days,
            'window_before': self.window_before,
            'window_after': self.window_after,
            'min_estimation_days': self.min_estimation_days
        }
        if self.estimator_type.lower() == 'capm':
            self.factors_df = load_daily_factors()
            return CAPM(**study_params)
        elif self.estimator_type.lower() == 'ff3':
            self.factors_df = load_daily_factors()
            return FF3(*study_params)
        else:
            raise NotImplementedError(f"Estimator type {self.estimator_type} is not implemented.")

    def _process_events(self, input_df: pd.DataFrame) -> pd.DataFrame:
        event_df = input_df[[self.ticker_col, self.event_date_col]].copy()
        event_df[self.event_date_col] = pd.to_datetime(event_df[self.event_date_col]).dt.date
        # Add event ID column
        event_df.loc[:, self.event_id_col] = event_df.reset_index(drop=True).index
        # Remove duplicates and convert to datetime
        event_df.drop_duplicates(inplace=True)
        event_df.loc[:, self.date_col] = event_df[self.event_date_col].copy()
        # Adjust day of the event if needed
        event_df = expand_date_columns(
            event_df, 
            date_col=self.date_col,
            before=self.window_before+self.gap_days + self.estimation_days + self.AUX_OFFSET,
            after=self.window_after + self.AUX_OFFSET)
        event_df[self.date_col] = event_df[self.date_col].dt.date
        return event_df
    
    def add_returns(self, list_of_tickers: List[str]=None, min_date: str=None, max_date: str=None, ret_df: pd.DataFrame=None):
        if ret_df is not None:
            ret_df.loc[:, self.date_col] = pd.to_datetime(ret_df[self.date_col]).dt.date
            self.returns_df = ret_df[[self.ret_col, self.date_col, self.ticker_col]]
            return 
        if list_of_tickers is None or min_date is None or max_date is None:
            raise ValueError('Either returns DF should be provided, or the loading parameters.')
        self.returns_df = load_daily_returns(
            list_of_tickers=list_of_tickers,
            min_date=min_date, 
            max_date=max_date
        )
        return

    def run_study(self, x: pd.DataFrame, y=None) -> 'EventStudy':
        event_df = self._process_events(x)
        if self.returns_df is None:
            print(f"""Start loading the returns""")
            self.add_returns(
                event_df[self.ticker_col].unique(), 
                min_date=event_df[self.date_col].min(),
                max_date=event_df[self.date_col].max()
            )

        feature_df = pd.merge(
            self.returns_df,
            self.factors_df, on=self.date_col)
        event_feature_df = pd.merge(
            event_df,
            feature_df,
            on=[self.ticker_col, self.date_col], how='outer'
        )
        event_feature_df.dropna(inplace=True)
        ## Start creating offset column. Note - it is done for trading and not calendar days.
        # Lets create a column with name OFFSet but fill at first with order number by event_id
        event_feature_df.sort_values([self.event_id_col, self.date_col], inplace=True)
        event_feature_df.loc[:, self.offset_col] = event_feature_df.groupby(self.event_id_col).cumcount()
        # Get mask associated with event itself
        event_mask = event_feature_df[self.event_date_col]==event_feature_df[self.date_col]
        # For each event_id - get its offset value
        event_to_offset_dict = event_feature_df[event_mask].set_index(self.event_id_col)[self.offset_col].to_dict()
        # Remove event day value to obtain valid offset
        event_feature_df.loc[:, self.offset_col] = event_feature_df[self.offset_col] - event_feature_df[self.event_id_col].map(event_to_offset_dict)
        event_feature_df.dropna(inplace=True)
        
        stats_df = event_feature_df.groupby(self.event_id_col).apply(
                lambda sub_df: self.estimator.fit_predict(sub_df))
        return stats_df
