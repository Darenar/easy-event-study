from typing import Any, Iterable, Callable
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
import plotly.graph_objects as go
from joblib import Parallel, delayed

from .base import ColumnNameHandler


def expand_date_columns(input_df: pd.DataFrame, date_col: str, before=400, after=30) -> pd.DataFrame:
    """
    Expand each date with n days before and n days after

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe
    date_col : str
        Date column
    before : int, optional
        Number of days to add before the date, by default 400
    after : _type_, optional
        Number of days to add after the date, by default 30

    Returns
    -------
    pd.DataFrame
        Dataframe with exploded dates
    """
    input_df.loc[:, date_col] = pd.to_datetime(input_df[date_col]).dt.date
    expand_date_mapping = {
        d: [d + dt.timedelta(days=int(i)) for i in np.arange(-before, after+1)] 
        for d in input_df[date_col].unique()
    }
    input_df.loc[:, date_col] = input_df[date_col].map(expand_date_mapping)
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
        car_df.median().to_frame('median'),
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


def plot_mean_car(event_res_df: pd.DataFrame, critical_value: float = 0.95, 
                  color_rgb: str = '0, 0, 255', bootstrap_df: pd.DataFrame = None) -> go.Figure:
    """
    Function to plot the mean CAR effect with confidence levels over the event window
    Parameters
    ----------
    event_res_df : pd.DataFrame
        Result dataframe from the output of event-study. Should contain event_id, offset, and car columns.
    critical_value : float, optional
        Critical value to use in confidence intervals estimation, by default 0.95
    color_rgb: str
        String specifying rgb color to be used in the plot. Default, blue (0,0,255)
    bootstrap_df: pd.DataFrame, optional
        Dataframe with bootstrap results to be used in confidence level identification

    Returns
    -------
    go.Figure
        Plotly figure
    """
    plot_df = calculate_car_stats(event_res_df=event_res_df, critical_value=critical_value)
    
    single_tail_critical_value = (1-critical_value)/2
    conf_level = np.round(- norm.ppf(single_tail_critical_value), 2)
    upper_conf_name = f'Mean + {conf_level}SE'
    lower_conf_name = f'Mean - {conf_level}SE'

    if bootstrap_df is not None:
        single_tail_critical_value = (1-critical_value)/2
        confidence_df = bootstrap_df.T.quantile([single_tail_critical_value, 1-single_tail_critical_value]).T
        confidence_df.columns = ['lower_ci', 'upper_ci']
        plot_df.drop(['lower_ci', 'upper_ci'], axis=1, inplace=True)
        plot_df = plot_df.join(confidence_df)
        upper_conf_name = f'Q:{round(1-single_tail_critical_value, 3)} by bootstrap'
        lower_conf_name = f'Q:{round(single_tail_critical_value, 3)} by bootstrap'

    # Add Mean Trace
    trace_one = go.Scatter(
        x=plot_df.index,
        y=plot_df['mean'],
        name='Mean',
        showlegend=True,
        mode='lines',
        line=dict(color=f'rgb({color_rgb})')
    )
    # Add trace for the upper bound
    trace_fill_one = go.Scatter(
        x=plot_df.index,
        y=plot_df['upper_ci'],
        name=upper_conf_name,
        mode='lines',
        line=dict(width=1, color=f'rgba({color_rgb}, 0.5)', dash='dash'),
        showlegend=True,
        
    )
    # Add trace for the lower bound
    trace_fill_two = go.Scatter(
        x=plot_df.index,
        y=plot_df['lower_ci'],
        name=lower_conf_name,
        mode='lines',
        line=dict(width=1, color=f'rgba({color_rgb}, 0.5)', dash='dash'),
        showlegend=True,
    )
    # Add trace for the lower bound
    trace_fill_two_shade = go.Scatter(
        x=plot_df.index,
        y=plot_df['lower_ci'],
        fill='tonextx',
        name='ShadedArea',
        mode='lines',
        fillcolor=f'rgba({color_rgb},0.1)',
        line=dict(width=1, color=f'rgba({color_rgb}, 0.5)', dash='dash'),
        showlegend=False,
    )
    # Update layout  
    layout = go.Layout(
        title=f"""<b>Cumulitative Abnormal Return: Mean & {round(critical_value*100)}% Confidence Limits</b><br>"""\
        f"""<i>Based on {event_res_df[ColumnNameHandler.event_id_col].nunique()} non-missing events""",
        template='plotly_white',
        xaxis_title='Day Relative to Event',
        yaxis_title='Return',
        yaxis_tickformat=',.1%',
        legend=dict(
                orientation="h",
                xanchor = "center",
                x=0.5,
                y=-0.2
        )        
    )
    fig = go.Figure(data=[
        trace_one, 
        trace_fill_one,
        trace_fill_two_shade,
        trace_fill_two
        ], layout=layout
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


def plot_joint_mean_car(event_res_one: pd.DataFrame, event_res_two: pd.DataFrame, 
                        name_one: str, name_two: str,  critical_value: float = 0.95) -> go.Figure:
    """
    Function to plot two mean CAR effects with confidence levels ont the same plot.
    
    Parameters
    ----------
    event_res_one : pd.DataFrame
        First result dataframe from the output of event-study. Should contain event_id, offset, and car columns.
    event_res_two : pd.DataFrame
        Second result dataframe from the output of event-study. Should contain event_id, offset, and car columns.
    name_one : str
        Name of the first event study
    name_two : str
        Name of the second event study
    critical_value : float, optional
        Critical value to use in confidence intervals estimation, by default 0.95
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig_one = plot_mean_car(event_res_one)
    fig_two = plot_mean_car(event_res_two, color_rgb='0, 153, 0')
    for trace in fig_one.data:
        trace.legendgroup = 'group1'
        trace.legendgrouptitle=dict(text=f'<b>{name_one}</b> ({event_res_one[ColumnNameHandler.event_id_col].nunique()} events)')

    for trace in fig_two.data:
        trace.legendgroup = 'group2'
        trace.legendgrouptitle=dict(text=f'<b>{name_two}</b> ({event_res_two[ColumnNameHandler.event_id_col].nunique()} events)')

    for trace in fig_one.data:
        fig_two.add_trace(trace)

    fig_two.update_layout(
        title=f"""<b>Cumulitative Abnormal Return: Mean & {round(critical_value*100)}% Confidence Limits</b><br>""",
        legend=dict(
            tracegroupgap=50,
            orientation="h",
            xanchor = "center",
            x=0.55,
            y=-0.3,
            entrywidth=200,
            traceorder='grouped'
        )
    )
    return fig_two


def run_in_parallel(func: Callable, data: Iterable[Any], n_cores: int) -> Iterable[Any]:
    if n_cores == 1:
        return [func(x) for x in data]
    processed_data = Parallel(n_jobs=n_cores)(
        delayed(func)(x) for x in data
    )
    return processed_data
