from typing import List

import logging
import yfinance as yf
from tqdm import tqdm
import pandas as pd

from .base import ColumnNameHandler


def load_daily_returns(list_of_tickers: List[str], min_date: str, max_date: str) -> pd.DataFrame:
    """
    Function to load daily return for the specified list of tickers and dates from Yahoo Finance.

    Parameters
    ----------
    list_of_tickers : List[str]
        List of tickers to load
    min_date : str
        Date to load the returns from
    max_date : str
        Date to load the return untill

    Returns
    -------
    pd.DataFrame
        Dataframe with ticker, Ret and Date columns
    """
    returns_df = list()
    # Keep only one-token tickers (otherwise, will fall with error) and keep only unique
    list_of_tickers = set([t for t in list_of_tickers if len(t.split())==1])
    for t in tqdm(list_of_tickers):
        # Load stock data daily
        ticker_df = yf.download(t, min_date, max_date)
        # If no data available - skip the ticker
        if 'Adj Close' not in ticker_df or ticker_df.empty:
            logging.getLogger(__name__).info(f"Could not load return for ticker {t}. Skipping it.")
            continue
        
        # Calculate daily returns as a percentage change
        ticker_df.loc[:, ColumnNameHandler.ret_col] = ticker_df['Adj Close'].pct_change()
        # Remove Nan values
        ticker_df.dropna(inplace=True)
        ticker_df.reset_index(inplace=True)
        # Append ticker name as a new column
        ticker_df.loc[:, ColumnNameHandler.ticker_col] = t
        returns_df.append(ticker_df[[
            ColumnNameHandler.ticker_col, 
            ColumnNameHandler.date_col, 
            ColumnNameHandler.ret_col, 
            ColumnNameHandler.volume_col
            ]].copy())
    if not returns_df:
        return None
    # Combine all returns together
    returns_df = pd.concat(returns_df, axis=0)
    returns_df.dropna(axis=1, inplace=True)
    returns_df.reset_index(inplace=True, drop=True)
    returns_df[ColumnNameHandler.date_col] = pd.to_datetime(returns_df[ColumnNameHandler.date_col]).dt.date
    return returns_df


def load_daily_factors(five_factors: bool = False) -> pd.DataFrame:
    """
    Function to load daily factors from Fama-Franch public library.

    Parameters
    ----------
    five_factors : bool, optional
        Whether to load five factors or three factors instead, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe with loaded factors per date
    """
    website_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    if five_factors:
        url_to_factors = f'{website_url}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    else:
        url_to_factors = f'{website_url}/F-F_Research_Data_Factors_daily_CSV.zip'
    # First 3 rows show meta data - thus, skip it
    ff_df = pd.read_csv(url_to_factors, skiprows=3)
    # Original index is Date - thus, in the file it will be unnamed. Change it.
    ff_df.rename({'Unnamed: 0': 'Date'}, axis=1, inplace=True)
    ff_df.loc[:, 'Date'] = pd.to_datetime(ff_df['Date'].astype(str), errors='coerce').dt.date
    # There could be a footer with metadata - make sure to drop it
    ff_df.dropna(axis=0, inplace=True)
    return ff_df
