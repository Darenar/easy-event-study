import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
import plotly.graph_objects as go

from .base import ColumnNameHandler


def expand_date_columns(input_df: pd.DataFrame, date_col: str, before=400, after=30) -> pd.DataFrame:
    input_df.loc[:, date_col] = pd.to_datetime(input_df[date_col])
    input_df.loc[:, date_col] = input_df[date_col].apply(
        lambda v: [v + dt.timedelta(days=int(i)) for i in np.arange(-before, after+1)])
    input_df = input_df.explode(date_col)
    input_df.reset_index(drop=True, inplace=True)
    return input_df


def calculate_car_stats(event_res_df: pd.DataFrame, critical_value: float = 0.95) -> pd.DataFrame:
    """
    Calculate to statistics for mean CAR effect with confidence levels over the event window
    Parameters
    ----------
    event_res_df : pd.DataFrame
        Result dataframe from the output of event-study. Should contain event_id, offset, and car columns.
    critical_value : float, optional
        Critical value to use in confidence intervals estimation, by default 0.95

    Returns
    -------
    pd.DataFrame
        Dataframe with estimated statistics
    """
    car_df = event_res_df.pivot(
        columns=ColumnNameHandler.offset_col, 
        index=ColumnNameHandler.event_id_col, 
        values=ColumnNameHandler.car_col)
    # Calculate mean and STD per each offset-day
    stat_car_df = pd.concat([
        car_df.mean().to_frame('mean'),
        # STD for each offset day as the deviation from mean CAR on the day, divided N * (N-1), where N - number of asset on the day
        np.sqrt(((car_df - car_df.mean())**2).sum() / (car_df.notna().sum() * (car_df.notna().sum() - 1))).to_frame('sd')
        ], axis=1)
    # Add T-stat results
    stat_car_df.loc[:, 't_stat'] = stat_car_df['mean'] / (stat_car_df['sd'])
    stat_car_df.loc[:, 'p_value'] = stat_car_df['t_stat'].apply(lambda v: norm.cdf(-abs(v)) * 2).round(3)
    # Add confidence levels
    conf_level = - norm.ppf((1-critical_value) / 2)
    stat_car_df.loc[:, 'upper_ci'] = stat_car_df['mean'] + conf_level * stat_car_df['sd']
    stat_car_df.loc[:, 'lower_ci'] = stat_car_df['mean'] - conf_level * stat_car_df['sd']
    return stat_car_df


def plot_mean_car(event_res_df: pd.DataFrame, critical_value: float = 0.95) -> go.Figure:
    """
    Function to plot the mean CAR effect with confidence levels over the event window
    Parameters
    ----------
    event_res_df : pd.DataFrame
        Result dataframe from the output of event-study. Should contain event_id, offset, and car columns.
    critical_value : float, optional
        Critical value to use in confidence intervals estimation, by default 0.95

    Returns
    -------
    go.Figure
        Plotly figure
    """
    plot_df = calculate_car_stats(event_res_df=event_res_df, critical_value=critical_value)
    # Add Mean Trace
    trace_one = go.Scatter(
        x=plot_df.index,
        y=plot_df['mean'],
        name='mean',
        showlegend=False,
        mode='lines',
        line=dict(color='rgb(0, 0, 255)')
    )
    # Add trace for the upper bound
    trace_fill_one = go.Scatter(
        x=plot_df.index,
        y=plot_df['upper_ci'],
        name='Upper CI',
        mode='lines',
        line=dict(width=1, color='rgba(0, 0, 255, 0.5)', dash='dash'),
        showlegend=False,
        
    )
    # Add trace for the lower bound
    trace_fill_two = go.Scatter(
        x=plot_df.index,
        y=plot_df['lower_ci'],
        fill='tonextx',
        name='Lower CI',
        mode='lines',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(width=1, color='rgba(0, 0,255, 0.5)', dash='dash'),
        showlegend=False,
    )
    # Update layout  
    layout = go.Layout(
        title=f"""<b>Cumulitative Abnormal Return: Mean & {round(critical_value*100)}% Confidence Limits</b><br>"""\
        f"""<i>Based on {event_res_df[ColumnNameHandler.event_id_col].nunique()} non-missing events""",
        template='plotly_white',
        xaxis_title='Day Relative to Event',
        yaxis_title='Return',
        yaxis_tickformat=',.1%'
    )
    fig = go.Figure(data=[
        trace_one, 
        trace_fill_one,
        trace_fill_two
        ], layout=layout,
    )
    fig.add_vline(
        x=0, 
        line_width=1, 
        line_color='rgba(0, 0,0, 0.5)')
    fig.add_hline(
        y=0, 
        line_width=1, 
        line_color='rgba(0, 0,0, 0.5)')
    return fig
