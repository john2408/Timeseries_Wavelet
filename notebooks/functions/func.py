import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from datetime import date,datetime,timedelta

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType, DoubleType, StructType, StructField, DateType, FloatType
from pyspark.sql.functions import *
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer

import mlflow
import mlflow.sklearn

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope


from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split

import os

def symm_mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    epsilon = np.finfo(np.float64).eps
    smape = 2*np.abs(y_pred - y_true) / np.maximum(np.abs(y_true)+np.abs(y_pred), epsilon)
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    return output_errors


# modified from https://blog.datadive.net/prediction-intervals-for-random-forests/
def pred_ints(model, X, interval_width):
    percentile = interval_width * 100
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x].reshape(1,-1))[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return np.asarray(err_down), np.asarray(err_up)


def config_objective(df, feature_cols, label_col, model, scoring, cv):
  
    """Configure the Hyperopt objective function

    Arguments:
    df: Pandas DataFrame:     The Pandas Dataframe on which to fit the model
    features_cols: List[str]: List of column names that represent the model features
    label_col: str            The label column name
    model: regressor          The model object that will be fit on the data, for instance a random forest
    scoring: str, scorer func Scoring method to use for selecting the best model
    cv: int, cv generator     The number of cross validation folds or cv generator
    """

    def objective(params):

        """The Hyperopt objective function"""

        model_conf = model(**params)

        scores = cross_val_score(model_conf, df[feature_cols], df[label_col], scoring=scoring, cv=cv)
        loss = -scores.mean()     # neg score as loss

        return {'loss': loss, 'params': params, 'status': STATUS_OK}  

    return objective

def config_objective_v2(df, feature_cols, label_col, model, scoring, cv):
  
    """Configure the Hyperopt objective function

    Arguments:
    df: Pandas DataFrame:     The Pandas Dataframe on which to fit the model
    features_cols: List[str]: List of column names that represent the model features
    label_col: str            The label column name
    model: regressor          The model object that will be fit on the data, for instance a random forest
    scoring: str, scorer func Scoring method to use for selecting the best model
    cv: int, cv generator     The number of cross validation folds or cv generator
    """

    def objective(params):

        """The Hyperopt objective function"""
        
        if model.__name__ not in ['XGBRegressor','RandomForestRegressor']:
            
            y = df.reset_index(drop=True)[label_col].copy()

            forecaster = model(**params)

            scores = evaluate(forecaster=forecaster, y=y, cv=cv)
            loss = -scores['test_MeanAbsolutePercentageError'].mean()    # neg score as loss
            
        else:
            
            model_conf = model(**params)

            scores = cross_val_score(model_conf, df[feature_cols], df[label_col], scoring=scoring, cv=cv)
            loss = -scores.mean()     # neg score as loss


        return {'loss': loss, 'params': params, 'status': STATUS_OK}  

    return objective

def fit_models_config(feature_cols, label_col, model, search_space, days_to_forecast, 
                      experiment_name=None, scoring='smape', 
                      cv=3, max_evals=100, 
                      bjective=config_objective, 
                      fit_best=True, ts=True):
  
    """Apply a scikit learn model to a group of data within a Spark DataFrame using a Pandas UDF

    Arguments:
    features_cols: List[str]:       List of column names that represent the model features
    label_col: str:                 Column to be predicted
    model: regressor :              (Regressor)Model to fit to the data
    search_space: Dict:             Grid search data structure containing the parameters to search
    days_to_forecast: int:          Number of days to forecast
    experiment_location: str:       Name of MLFlow experiment.If None, create a notebook experiment  
    scoring: str:                   Scoring method to use for validation
    cv: int:                        The number of cross validation folds 
    max_evals: int:                 Max Hyperopt evaluations to run
    objective: function:            The Hyperopt objective function
    fit_best: boolean:              If True, a model with the best parameters will be fit and logged in MLFlow
    ts: boolean:                    If True, the cross validation is a timeseries cross validation     
    """
    SEED = 123
    
    #if ts:
    #    cv = TimeSeriesSplit(n_splits=cv,test_size=days_to_forecast)
    if scoring == 'smape':
        scoring = make_scorer(symm_mean_absolute_percentage_error, greater_is_better=False)   # neg_smape as scoring    
        
    def fit_models(keys, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model; log the best model and its paramenters to 
        MLFlow"""

        # identify time series to forecast
        Key1_id = keys[0]
        Key2_id = keys[1]
        group_name = '{}_{}'.format(Key1_id,Key2_id)
        train_data = data
        train_data.sort_values(by='ds',inplace=True) #train_data.reset_index(drop=True, inplace=True)
            
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
            # Enable getting Experiment Details later by experiment = mlflow.get_experiment_by_name(experiment_name) 

        with mlflow.start_run() as run:

            # Configure and apply Hyperopt
            bayes_trials = Trials()
            objective_config = config_objective(train_data, feature_cols, 
                                                label_col, model, scoring=scoring, cv=cv)

            best_params = fmin(
                fn = objective_config, 
                space = search_space, 
                algo = tpe.suggest,
                max_evals = max_evals, 
                trials = bayes_trials, 
                rstate = np.random.default_rng(SEED))

            best_model_score = np.round(bayes_trials.best_trial['result']['loss'], 4)

            # Create model results output dataset
            model_results_df = pd.DataFrame([(group_name, best_model_score)], 
                                            columns= ['node', 'best_model_score'])

            # Log best model parameters and statistics to MLFlow
            mlflow.set_tag("ts_key", group_name)

            mlflow.set_tag("model_type", model.__name__)

            mlflow.log_metric("smape", best_model_score)

            mlflow.log_params(best_params)

            # Fit the best model on the full training dataset for the group
            if fit_best:

                # Configure and fit best model
                #best_params_as_int = {param_name: int(value) for param_name, value in best_params.items()}
                best_params['n_estimators'] = int(best_params['n_estimators'])
                # best_params['min_samples_split'] = int(best_params['min_samples_split'])
                # best_params['max_depth'] = int(best_params['max_depth'])
                # best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
                best_model_conf = model(**best_params)
                best_model_conf.fit(train_data[feature_cols], train_data[label_col])

                # Log the best model to MLFlow
                mlflow.sklearn.log_model(sk_model=best_model_conf, 
                                          artifact_path='tuned_model')

        return model_results_df

    return fit_models


def fit_multiple_models(feature_cols, 
                        label_col, 
                        models, 
                        search_space, 
                        days_to_forecast,
                        cv_objects,
                        experiment_name=None, 
                        scoring='smape',
                        max_evals=100, 
                        config_objective=config_objective_v2, 
                        fit_best=True, 
                        ts=True):
  
    """Apply a scikit learn model to a group of data within a Spark DataFrame using a Pandas UDF

    Arguments:
    features_cols: List[str]:       List of column names that represent the model features
    label_col: str:                 Column to be predicted
    models: regressor :             List of models to fit the timeseries
    search_space: Dict:             Grid search data structure containing the parameters to search
    days_to_forecast: int:          Number of days to forecast
    experiment_location: str:       Name of MLFlow experiment.If None, create a notebook experiment  
    scoring: str:                   Scoring method to use for validation
    cv: int:                        The number of cross validation folds 
    max_evals: int:                 Max Hyperopt evaluations to run
    objective: function:            The Hyperopt objective function
    fit_best: boolean:              If True, a model with the best parameters will be fit and logged in MLFlow
    ts: boolean:                    If True, the cross validation is a timeseries cross validation     
    """
    SEED = 123
    
    if scoring == 'smape':
        scoring = make_scorer(symm_mean_absolute_percentage_error, greater_is_better=False)   # neg_smape as scoring    
        
    def fit_models(keys, data: pd.DataFrame) -> None:
        """Fit the model; log the best model and its paramenters to 
        MLFlow"""

        # identify time series to forecast
        Key1_id = keys[0]
        Key2_id = keys[1]
        group_name = '{}_{}'.format(Key1_id,Key2_id)
        train_data = data
        train_data.sort_values(by='ds',inplace=True) #train_data.reset_index(drop=True, inplace=True)
            
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
            # Enable getting Experiment Details later by experiment = mlflow.get_experiment_by_name(experiment_name) 
        
        for model in models:
            
            if model.__name__ in ['XGBRegressor','RandomForestRegressor']:
                search_params = search_space['tree_algo']
                cv = cv_objects['tree_algo']
            else:
                search_params = search_space['ts_algo']
                cv = cv_objects['ts_algo']
                
            
            with mlflow.start_run() as run:

                # Configure and apply Hyperopt
                bayes_trials = Trials()
                objective_config = config_objective(train_data, feature_cols, 
                                                    label_col, model, scoring=scoring, cv=cv)

                best_params = fmin(
                    fn=objective_config, 
                    space=search_params, 
                    algo=tpe.suggest,
                    max_evals=max_evals, 
                    trials=bayes_trials, 
                    rstate=np.random.default_rng(SEED))

                best_model_score = np.round(bayes_trials.best_trial['result']['loss'], 4)

                # Create model results output dataset
                model_results_df = pd.DataFrame([(group_name, best_model_score)], 
                                                columns= ['node', 'best_model_score'])

                # Log best model parameters and statistics to MLFlow
                mlflow.set_tag("ts_key", group_name)

                mlflow.set_tag("model_type", model.__name__)

                mlflow.log_metric("smape", best_model_score)

                mlflow.log_params(best_params)

                # Fit the best model on the full training dataset for the group
                if fit_best:

                    # Configure and fit best model
                    #best_params_as_int = {param_name: int(value) for param_name, value in best_params.items()}
                    best_params['n_estimators'] = int(best_params['n_estimators'])
                    best_params['max_depth'] = int(best_params['max_depth'])
                    
                    # best_params['min_samples_split'] = int(best_params['min_samples_split'])
                    # best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
                    
                    best_model_conf = model(**best_params)
                    best_model_conf.fit(train_data[feature_cols], train_data[label_col])

                    # Log the best model to MLFlow
                    mlflow.sklearn.log_model(sk_model=best_model_conf, 
                                              artifact_path='tuned_model')

        return model_results_df

    return fit_models


def apply_models_config(features_cols, score="smape", experiment_id=None):
  
    """For each distinct group (values in groupBy statement), load the group's best model and 
    perform a prediction

    Arguments:
    features_cols: List[str]:       List of column names that represent the model features
    scoring: str                    Scoring method to use for selecting the best model
    experiment_id: str              The id of the experiment from which to select models. Note, if
                                  using the notebook experience (no external MLFlow experiment created)
                                  then the experiment id should equal to the notebook id(see top lines)

    """

    def apply_models(keys, data: pd.DataFrame) -> pd.DataFrame:

        """Load the relvent model for the selected group and generate a
        prediciton for the group"""

        # identify ts_key to forecast
        Key1_id = keys[0]
        Key2_id = keys[1]
        group_name = '{}_{}'.format(Key1_id,Key2_id)

        # Find the best model for this group with lowest scoring metric (smape)
        query = "tags.model_type = 'RandomForestRegressor' and attributes.status = 'FINISHED' and metrics.smape <= 2 and tags.ts_key ='{}'".format(group_name)
        best_run_df = mlflow.search_runs(experiment_id, filter_string=query,order_by=["metrics.smape"],max_results=1)

        best_model_run_id = best_run_df.run_id.values[0]

        # Load the best model via its run_id
        model_loc = os.path.join(os.getcwd(), f"mlruns/{experiment_id}/{best_model_run_id}/artifacts/tuned_model")
        loaded_model = mlflow.sklearn.load_model(model_loc)
        
        # make sure the data are in correct order while using pyspark
        data.sort_values(by='ds',inplace=True)
        # Perform forecast; combine features and predictions
        yhat = np.round(loaded_model.predict(data[features_cols]))
        yhat_lower, yhat_upper = pred_ints(loaded_model, data[features_cols].values, interval_width=0.95)
        preds_np = np.concatenate(
        (
          yhat.reshape(-1,1), 
          yhat_lower.reshape(-1,1), 
          yhat_upper.reshape(-1,1)
          ), axis=1
        )
        forecast_pd = pd.DataFrame(preds_np, columns=['yhat', 'yhat_lower', 'yhat_upper'])
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast_pd[col] = forecast_pd[col].clip(lower=0.0)
       
        # PREPARE RESULTS
        # ---------------------------------
        # merge forecast with history
        results_pd = pd.concat(
        [data['ds'], data['y'], forecast_pd],
        axis=1
        )

        # get ts_key from incoming data set
        results_pd['Key1'] = Key1_id
        results_pd['Key2'] = Key2_id
        results_pd['run_id'] = best_model_run_id

        return results_pd

    return apply_models

# modified from https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py

from matplotlib import pyplot as plt
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
from matplotlib.ticker import FuncFormatter

def generate_plot( model, forecast_pd, xlabel='ds', ylabel='y'):
    ax=None
    figsize=(10, 6)

    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    history_pd = forecast_pd[forecast_pd['y'] != np.NaN]
    fcst_t = forecast_pd['ds'].dt.to_pydatetime()

    ax.plot(history_pd['ds'].dt.to_pydatetime(), history_pd['y'], 'k.')
    ax.plot(fcst_t, forecast_pd['yhat'], linestyle='-', c='#0072B2', label='forecast')
    ax.fill_between(fcst_t, forecast_pd['yhat_lower'], forecast_pd['yhat_upper'], color='#0072B2', alpha=0.2)

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    return fig

def load_model(key1_id, key2_id, experiment_id, group_name):
    """
    Load model from mlruns
    """
    
    
    # Find the best model for this group with lowest scoring metric (smape)
    query = "tags.model_type = 'RandomForestRegressor' and attributes.status = 'FINISHED' and metrics.smape <= 2 and tags.ts_key ='{}'".format(group_name)
    best_run_df = mlflow.search_runs(experiment_id, filter_string=query,order_by=["metrics.smape"],max_results=1)

    best_model_run_id = best_run_df.run_id.values[0]
    

    # Load the best model via its run_id
    model_loc = os.path.join(os.getcwd(), f"mlruns/{experiment_id}/{best_model_run_id}/artifacts/tuned_model")
    model_cv_smape = best_run_df["metrics.smape"].values[0]
    model = mlflow.sklearn.load_model(model_loc)
    
    return model, best_run_df, model_cv_smape

def plot_forecast(forecast_pd, model_cv_smape, last_train_date, model, group_name, best_model_name ):
    """
    Forecast Plot with intervals. 
    """

    # construct a visualization of the forecast
    forecast_pd.sort_values(by='ds',inplace=True)
    forecast_pd['ds'] = pd.to_datetime(forecast_pd['ds'], errors='coerce')

    train = forecast_pd[forecast_pd['ds'].dt.date <= last_train_date]
    test = forecast_pd[forecast_pd['ds'].dt.date > last_train_date]
    test_smape = np.round(symm_mean_absolute_percentage_error(test['y'], test['yhat']),4)

    predict_fig = generate_plot(model, forecast_pd, xlabel='date', ylabel='inflow')
    plt.plot(test['ds'], test['y'], label='actual', c='#249156')
    plt.title('Best Model : {} - Validation data v. forecast of {} with smape(avg on fh) cv:{:.4f} vs. test:{:.4f}' \
              .format(best_model_name, group_name, model_cv_smape, test_smape))
    plt.legend();

    # adjust the x-axis to focus on a limited date range
    xlim = predict_fig.axes[0].get_xlim()
    new_xlim = (datetime.strptime('2020-9-15','%Y-%m-%d'), datetime.strptime('2022-03-17','%Y-%m-%d'))
    predict_fig.axes[0].set_xlim(new_xlim)
    # display the chart
    plt.show()