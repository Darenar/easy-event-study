# Easy-Event-Study
Conduct financial event-study in just few lines of Python code. 
<p align="center">
<img src="imgs/example_one.png" alt="image" width="800" height="auto" alt='Example of mean CAR'>
</p>
<p align="center">
  <i>Example of the event study output</i>
</p>

## Usage
```
from easy_es import EventStudy
from easy_es.utils import plot_mean_car

# Event file should have 2 columns: ticker and event_date
events = pd.read_csv('events_df.csv')

event_study = EventStudy(
    estimation_days=255,
    gap_days=50,
    window_after=10,
    window_before=10,
    min_estimation_days=100,
    estimator_type='ff3'
)
event_study.add_returns(
    list_of_tickers=events_df['ticker'].unique(),
    min_date='2010-01-01',  # Select min date to load returns from
    max_date='2023-01-01'   # Select max date to load returns until
)

event_res_df = event_study.run_study(events_df)
# Plot mean effect with confidence levels
plot_mean_car(event_res_df=event_res_df, critical_value=0.95)
```

## Methodology
<p align="center">
<img src="imgs/event_study_methodology.png" alt="image" width="800" height="auto" alt='Methodology'>
</p>
<p align="center">
  <i>Methodology schema</i>
</p>
For each event, define:

1. *Estimation Period* - when the parameters for the selected counterfactual model will be estimated.
2. *Gap Period* - number of days to skip after estimation period
3. *Event window boundaries* - number of days before and after an event to use in the calculations

Fit a selected model to estimate **normal returns** during the *estimation period*, and then construct **abnormal returns** (*actual returns - normal return*) during the *event window*. 

Implemented models to estimate normal returns are:

1. CAPM - $Ret_{i, t} = \alpha + \beta * (MktRet_{t} - RF_{t})$, where 
    
    * $Ret_{i,t}$ - Return of company $i$ on day $t$. 
    
    * $MktRet_{t}$ - Market return on day $t$. 

    * $RF_{t}$ - Risk free rate on day $t$. 

2. Fama-French 3 factor model (aka FF3) - $Ret_{i, t} = \alpha + \beta * (MktRet_{t} - RF_{t}) + \gamma * SMB + \theta * HML$, where

    * $Ret_{i,t}$ - Return of company $i$ on day $t$
    
    * $MktRet_{t}$ - Market return on day $t$. 

    * $RF_{t}$ - Risk free rate on day $t$. 

    * $SMB_{t}$ - Small minus Big factor on day $t$. 

    * $HML_{t}$ - High minus Low factor on day $t$. 

## Financial Data Sources
All the financial data is loaded automatically inside the package. In particular:

1. All daily factors are downloaded for the official Fama-Franch data library.
2. Prices are loaded using *yfinance* package.





