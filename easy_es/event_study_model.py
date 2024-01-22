from typing import List, Union, Dict

import pandas as pd
from copy import deepcopy
from dateutil import parser
import numpy as np
from datetime import timedelta

from .base import ColumnNameHandler as CNH
from .data_loader import load_daily_returns, load_daily_factors
from .utils import expand_date_columns, run_in_parallel
from .estimation_models import RegReturnEstimator, MAMReturnEstimator, BaseEstimator


def apply_estimator(input_dict: Dict[str, Union[BaseEstimator, pd.DataFrame]]) -> pd.DataFrame:
    """
    Function to apply estimator for a given input.
    Separated from the class for a multiprocessing purposes

    Parameters
    ----------
    input_dict : Dict[str, Union[BaseEstimator, pd.DataFrame]]
        Dictionary with input arguments. Should contains keys: estimator, estimation_df, prediction_df

    Returns
    -------
    pd.DataFrame
        Dataframe with predicted returns
    """
    estimator = input_dict['estimator']
    estimation_df = input_dict['estimation_df']
    prediction_df = input_dict['prediction_df']

    estimator.fit(estimation_df)
    resid_std = estimator.resid.std()
    prediction_df.loc[:, CNH.pred_ret_col] = estimator.predict(prediction_df)
    # Get all abnormal columns
    prediction_df.sort_values(CNH.offset_col, inplace=True)
    prediction_df.loc[:, CNH.ar_col] = prediction_df[CNH.ret_col] - prediction_df[CNH.pred_ret_col]
    prediction_df.loc[:, CNH.car_col] = prediction_df[CNH.ar_col].cumsum()
    prediction_df.loc[:, CNH.sar_col] = prediction_df[CNH.ar_col] / resid_std
    prediction_df.loc[:, CNH.scar_col] = prediction_df[CNH.car_col] / (
        resid_std * np.sqrt(prediction_df.reset_index(drop=True).index+1))
    prediction_df.loc[:, CNH.e_std_col] = resid_std
    return prediction_df


class EventStudy(CNH):
    AUX_OFFSET: int = 10
    def __init__(self, estimation_days: int = 255, gap_days: int = 50, 
                 window_before: int = 10, window_after: int = 10, 
                 min_estimation_days: int = 100, estimator_type: str = 'capm',
                 n_cores: int = None):
        self.estimation_days = estimation_days
        self.gap_days = gap_days
        self.window_before = window_before
        self.window_after = window_after
        self.min_estimation_days = min_estimation_days
        self.returns_df = None
        self.estimator_type = estimator_type
        self.n_cores = n_cores
        self.estimator = self.__initialize_estimator()
        self.factors_df = self.add_factors()
    
    def __initialize_estimator(self):
        if self.estimator_type.lower() == 'capm':
            return RegReturnEstimator(feature_cols=self.mkt_rf_col)
        elif self.estimator_type.lower() == 'ff3':
            return RegReturnEstimator(feature_cols=[self.mkt_rf_col, self.smb_col, self.hml_col])
        elif self.estimator_type.lower() == 'ff5':
            return RegReturnEstimator(feature_cols=[self.mkt_rf_col, self.smb_col, self.hml_col, self.rmw_col, self.cma_col])
        elif self.estimator_type.lower() == 'mam':
            return MAMReturnEstimator()
        else:
            raise NotImplementedError(f"Estimator type {self.estimator_type} is not implemented.")
    
    def add_factors(self):
        if self.estimator_type == 'ff5':
            factors_df = load_daily_factors(five_factors=True)
        else:
            factors_df = load_daily_factors()
        # Need to downscale factor columns from percent to float
        factor_names = factors_df.columns.difference([self.date_col])
        factors_df.loc[:, factor_names] = factors_df[factor_names]/100
        return factors_df
        
    def process_events_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to process input events. In particular:
        1. Remove duplicates
        2. Convert dates into an appropriate datetime format
        3. Add auxiliary columns like offset for easier estimation/event window access
        Parameters
        ----------
        input_df : pd.DataFrame
            Input datatframe with ticker and event_date columns

        Returns
        -------
        pd.DataFrame
            Process event dataframe 
        """
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
    
    def add_returns(self, list_of_tickers: List[str]=None, 
                    min_date: str=None, max_date: str=None, 
                    ret_df: pd.DataFrame=None):
        """
        Function to load returns for specified tickers and time period.
        Parameters
        ----------
        list_of_tickers : List[str], optional
            List of tickers to load returns for, by default None
        min_date : str, optional
            Minimum date to load the returns from, by default None
        max_date : str, optional
            Maximum date to load the returns until, by default None
        ret_df : pd.DataFrame, optional
            Data frame of pre-loaded returns, by default None
        """
        # If pre-loaded returns are provided - use them
        if ret_df is not None:
            ret_df.loc[:, self.date_col] = pd.to_datetime(ret_df[self.date_col]).dt.date
            self.returns_df = ret_df[[self.ret_col, self.date_col, self.ticker_col, self.volume_col]]
            return 
        if list_of_tickers is None or min_date is None or max_date is None:
            raise ValueError('Either returns DF should be provided, or the loading parameters.')
        
        # Add offset to the dates to make sure all returns for all specified dates are returned
        if isinstance(min_date, str):
            min_date = parser.parse(min_date)
        if isinstance(max_date, str):
            max_date = parser.parse(max_date)
        min_date = min_date - timedelta(days=1)
        max_date = max_date + timedelta(days=1)
        self.returns_df = load_daily_returns(
            list_of_tickers=list_of_tickers,
            min_date=min_date, 
            max_date=max_date
        )
        return
    
    def __add_offset(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create an offset column. Offset - how many days is between an event and a particular date.
        Note - it should be done for trading and not calendar days. Thus, only after the inner-join with returns data

        Parameters
        ----------
        input_df : pd.DataFrame
            Event dataframe to process

        Returns
        -------
        _type_
            _description_
        """
        # Lets create a column with name offSet but fill at first with order number by event_id
        input_df.sort_values([self.event_id_col, self.date_col], inplace=True)
        input_df.loc[:, self.offset_col] = input_df.groupby(self.event_id_col).cumcount()
        # Get mask associated with event itself
        event_mask = input_df[self.event_date_col]==input_df[self.date_col]
        # For each event_id - get its offset value
        event_to_offset_dict = input_df[event_mask].set_index(self.event_id_col)[self.offset_col].to_dict()
        # Remove event day value to obtain valid offset
        input_df.loc[:, self.offset_col] = input_df[self.offset_col] - input_df[self.event_id_col].map(event_to_offset_dict)
        input_df.dropna(inplace=True)
        return input_df

    def run_study(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        event_df = self.process_events_df(x)

        # Adding Returns data
        if self.returns_df is None:
            print(f"""Start loading the returns""")
            self.add_returns(
                event_df[self.ticker_col].unique(), 
                min_date=event_df[self.date_col].min(),
                max_date=event_df[self.date_col].max()
            )
        event_df = pd.merge(
            event_df,
            self.returns_df,
            on=[self.ticker_col, self.date_col], how='outer'
        )
        event_df.dropna(inplace=True)

        # Add offset column
        event_df = self.__add_offset(event_df)
        
        # Add factors data
        event_df = pd.merge(event_df, self.factors_df, on=self.date_col)
        
        # Create estimation and prediction periods
        estimation_period = np.arange(
            -(self.window_before+self.gap_days+self.estimation_days), 
            -(self.window_before+self.gap_days), 
            1)
        prediction_period = np.arange(-self.window_before, self.window_after + 1, 1)

        # Create chunks by event_id
        groups = [
            {'estimator': deepcopy(self.estimator),
             'estimation_df':  group[group[self.offset_col].isin(estimation_period)],
             'prediction_df':  group[group[self.offset_col].isin(prediction_period)]}
             for _, group in event_df.groupby(self.event_id_col)
        ]
        # Filter out events with lower number of estimation observations
        groups = [g for g in groups if g['estimation_df'].shape[0] >= self.min_estimation_days]
        prediction_df = run_in_parallel(
            apply_estimator,
            groups, 
            n_cores=self.n_cores
        ) 
        prediction_df = pd.concat(prediction_df, ignore_index=True)
        return prediction_df
