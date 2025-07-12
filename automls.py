import numpy as np
from tpot import TPOTRegressor
import h2o
from h2o.automl import H2OAutoML
from flaml import AutoML as flaml_AutoML
from supervised.automl import AutoML as mjar_AutoML
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
import time
from scipy.stats import pearsonr
import autosklearn.regression
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from pycaret.regression import *
from autogluon.tabular import TabularPredictor


def model_performance(y_test, y_pred):
    # ML perspective
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    performance_result = {'RMSE': math.sqrt(mse),
                          'NRMSE': (((math.sqrt(mse)) / abs(np.mean(y_test))) * 100),
                          'MAE': mae,
                          'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                          'R2': r2,
                          'std': y_pred.std(),
                          'rho': pearsonr(y_pred, y_test)[0],
                          'ref': y_test.std()
                          }

    return performance_result


def _tpot_(train,
           test,
           target,
           generation=3,
           population_size=5):
    tpot = TPOTRegressor(generations=generation,
                         population_size=population_size,
                         cv=5,
                         n_jobs=-1,
                         max_time_mins=10)

    start = time.time()

    tpot.fit(train.drop(target, axis=1), train[target])

    y_pred = tpot.predict(test.drop(target, axis=1))

    performance = model_performance(test[target], y_pred)

    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'TPOT'
    performance['Training time'] = running_time

    return performance


# Using H2O AutoML framework to derive the optimal ML configuration
def _h2o_(train,
          test,
          target,
          max_models=10):
    h2o.init()
    train_frame = h2o.H2OFrame(train)
    test_frame = h2o.H2OFrame(test)

    # Run AutoML for 10 base models
    aml = H2OAutoML(max_models=max_models)

    start = time.time()
    aml.train(x=train.drop(target, axis=1).columns.tolist(), y=target, training_frame=train_frame)

    y_pred = aml.leader.predict(test_frame).as_data_frame().loc[:, 'predict']
    y_test = test[target]

    performance = model_performance(y_test, y_pred)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    performance['Framework'] = 'H2O'
    performance['Training time'] = running_time

    # h2o leanderboard
    # lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
    return performance


# Using FLAML to train ML models
def _flaml_(train, test, target,
            max_iter=100,
            time_budget=300):
    automl = flaml_AutoML()

    start = time.time()

    automl.fit(X_train=train.drop(target, axis=1),
               y_train=train[target],
               task='regression',
               time_budget=time_budget,
               max_iter=max_iter)

    # Print the best model
    model = automl.model.estimator
    y_pred = model.predict(test.drop(target, axis=1))
    performance = model_performance(test[target], y_pred)

    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))  # minutes

    performance['Framework'] = 'FLAML'
    performance['Training time'] = running_time
    return performance


# Using mjar to train ML models
def _mjar_(train, test,
           target,
           total_time_limit=300,
           mode='Explain'):
    automl = mjar_AutoML(total_time_limit=total_time_limit, mode=mode)

    start = time.time()

    automl.fit(train.drop(target, axis=1), train[target])

    y_pred = automl.predict(test.drop(target, axis=1))

    performance = model_performance(test[target], y_pred)

    finish = time.time()

    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'mjar_supervised'
    performance['Training time'] = running_time
    return performance


def _autogluon_(train, test, target):
    start = time.time()
    predictor = TabularPredictor(label=target).fit(train)
    finish = time.time()

    running_time = ("%.2f" % ((finish - start) / 60))

    y_pred = predictor.predict(test)
    y_test = test[target]

    performance = model_performance(y_test, y_pred)
    performance['Framework'] = 'AutoGluon'
    performance['Training time'] = running_time
    return performance


def _pycaret_(train, test, target):
    start = time.time()
    s = setup(train,
              index=False,
              target=target,
              test_data=test)

    best = compare_models()
    finish = time.time()

    y_pred = predict_model(best, data=test).prediction_label
    performance = model_performance(test[target].values, y_pred.values)

    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'PyCaret'
    performance['Training time'] = running_time

    return performance


def _tabpfn_(train, test, target):
    start = time.time()

    regressor = TabPFNRegressor(ignore_pretraining_limits=True)
    regressor.fit(train.drop(target, axis=1), train[target])

    # Predict on the test set
    predictions = regressor.predict(test.drop(target, axis=1))

    # Evaluate the model
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    performance['Framework'] = 'TabPFN'
    performance['Training time'] = running_time
    return performance


# Ubuntu System
def _auto_sklearn_(train, test, target):
    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=300)

    start = time.time()
    automl.fit(train.drop(target, axis=1), train[target])
    finish = time.time()

    y_pred = automl.predict(test.drop(target, axis=1))
    performance = model_performance(test[target], y_pred)

    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'auto_sklearn'
    performance['Training time'] = running_time

    return performance


def _auto_pytorch_(train, test, target):
    api = TabularRegressionTask()
    start = time.time()
    api.search(
        X_train=train.drop(target, axis=1),
        y_train=train[target],
        X_test=test.drop(target, axis=1),
        y_test=test[target],
        optimize_metric='r2',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50)
    finish = time.time()

    y_pred = api.predict(test.drop(target, axis=1))
    performance = model_performance(test[target], y_pred)

    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'Auto_Pytorch'
    performance['Training time'] = running_time
    return performance
