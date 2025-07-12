# -*- coding: UTF-8 -*-
import pandas as pd
import data_aug as da
from automls import _tpot_, _mjar_, _flaml_, _h2o_, _pycaret_, _tabpfn_, _autogluon_, _auto_pytorch_, _auto_sklearn_
import warnings
import gc

warnings.filterwarnings('ignore')


# Data generation
def real_syn_data_generate(data,
                           target,
                           train_size,
                           random_state,
                           syn_method):
    # Use synthesized data
    real_train, synthetic_train, test, hp_log = da.synthesize_data_split(data, target,
                                                                         train_size=train_size,
                                                                         random_state=random_state,
                                                                         syn_method=syn_method)
    # data saved
    real_train.to_csv(f'./data_hp/real_train_{random_state}.csv', index=False)
    test.to_csv(f'./data_hp/test_{random_state}.csv', index=False)
    synthetic_train.to_csv(f'./data_hp/synthetic_train_{random_state}_{syn_method}.csv', index=False)
    hp_log.to_csv(f'./hp_log/hp_log_{random_state}_{syn_method}.csv', index=False)
    return real_train, test, synthetic_train


# Data saving
def data_saved(corr_threshold):
    real_data = pd.read_csv('data.csv', encoding='utf-8')
    random_states = [76, 19, 63, 36, 85, 78, 86, 99, 2, 12, 11, 37, 48, 31, 45, 68, 40, 39, 49, 77]

    for random_state in random_states:
        # ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']
        for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
            real_train, test, synthetic_train = real_syn_data_generate(data=real_data,
                                                                       target='Strength',
                                                                       random_state=random_state,
                                                                       syn_method=syn_method,
                                                                       train_size=0.7)
            for filter_method in ['Ransac', 'Lof', 'If']:
                real_train_corr = real_train.corr()
                features = real_train_corr[(abs(real_train_corr['Strength']) >= corr_threshold) &
                                           (real_train_corr['Strength'] < 1)].index.tolist()
                filter_synthetic_train = da.filter_synthesize_data(real_train, synthetic_train, features,
                                                                   filter_method=filter_method)
                filter_synthetic_train.to_csv(
                    f'./data_hp/filter_synthetic_train_{random_state}_{syn_method}_{filter_method}.csv', index=False)


# predict using real train and test
def predict(real_train,
            test,
            target,
            train_method='TabPFN'):
    if train_method == 'TPOT':
        return _tpot_(real_train, test, target)

    if train_method == 'H2O':
        return _h2o_(real_train, test, target)

    if train_method == 'FLAML':
        return _flaml_(real_train, test, target)

    if train_method == 'mjar_supervised':
        return _mjar_(real_train, test, target)

    if train_method == 'AutoGluon':
        return _autogluon_(real_train, test, target)

    if train_method == 'PyCaret':
        return _pycaret_(real_train, test, target)

    if train_method == 'TabPFN':
        return _tabpfn_(real_train, test, target)

    if train_method == 'auto-sklearn':
        return _auto_sklearn_(real_train, test, target)

    if train_method == 'AutoPyTorch':
        return _auto_pytorch_(real_train, test, target)


# predict using real train & initial train, test
def predict_syn(syn_train,
                test,
                target,
                train_method='TabPFN'):
    if train_method == 'TPOT':
        return _tpot_(syn_train, test, target)

    if train_method == 'H2O':
        return _h2o_(syn_train, test, target)

    if train_method == 'FLAML':
        return _flaml_(syn_train, test, target)

    if train_method == 'mjar_supervised':
        return _mjar_(syn_train, test, target)

    if train_method == 'AutoGluon':
        return _autogluon_(syn_train, test, target)

    if train_method == 'PyCaret':
        return _pycaret_(syn_train, test, target)

    if train_method == 'TabPFN':
        return _tabpfn_(syn_train, test, target)

    if train_method == 'auto-sklearn':
        return _auto_sklearn_(syn_train, test, target)

    if train_method == 'AutoPyTorch':
        return _auto_pytorch_(syn_train, test, target)


# using real train & finalized train, test
def predict_syn_filter(filter_syn_train,
                       test,
                       target,
                       train_method='TabPFN'):
    if train_method == 'TPOT':
        return _tpot_(filter_syn_train, test, target)

    if train_method == 'H2O':
        return _h2o_(filter_syn_train, test, target)

    if train_method == 'FLAML':
        return _flaml_(filter_syn_train, test, target)

    if train_method == 'mjar_supervised':
        return _mjar_(filter_syn_train, test, target)

    if train_method == 'AutoGluon':
        return _autogluon_(filter_syn_train, test, target)

    if train_method == 'PyCaret':
        return _pycaret_(filter_syn_train, test, target)

    if train_method == 'TabPFN':
        return _tabpfn_(filter_syn_train, test, target)

    if train_method == 'auto-sklearn':
        return _auto_sklearn_(filter_syn_train, test, target)

    if train_method == 'AutoPyTorch':
        return _auto_pytorch_(filter_syn_train, test, target)



# Experimental strategy
def train_experiments(train_method):
    random_states = [76, 19, 63, 36, 85, 78, 86, 99, 2, 12, 11, 37, 48, 31, 45, 68, 40, 39, 49, 77]

    performances = []
    for random_state in random_states:
        # real train and test
        real_train = pd.read_csv(f'./data_hp/real_train_{random_state}.csv', encoding='unicode_escape')
        test = pd.read_csv(f'./data_hp/test_{random_state}.csv', encoding='unicode_escape')

        # train using real train and test data
        performance = predict(real_train=real_train, test=test, target='Strength', train_method=train_method)

        # log
        performance['configuration'] = 'no_syn'
        performance['random state'] = random_state
        performance['model'] = train_method

        performances.append(performance)

        # ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']
        for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
            synthetic_train = pd.read_csv(
                f'./data_hp/synthetic_train_{random_state}_{syn_method}.csv', encoding='unicode_escape')

            # real_train + synthetic_train
            syn_train = pd.concat([real_train, synthetic_train], axis=0)

            performance_syn = predict_syn(syn_train=syn_train, test=test, target='Strength',
                                          train_method=train_method)

            del synthetic_train, syn_train
            gc.collect()

            # log
            performance_syn['random state'] = random_state
            performance_syn['model'] = train_method

            performance_syn['configuration'] = 'syn'
            performance_syn['configuration_syn_method'] = syn_method

            performances.append(performance_syn)

            for filter_method in ['Ransac', 'Lof', 'If']:
                filter_synthetic_train = pd.read_csv(
                    f'./data_hp/filter_synthetic_train_{random_state}_{syn_method}_{filter_method}.csv',
                    encoding='unicode_escape')

                # real_train + filter_synthetic_train
                filter_syn_train = pd.concat([real_train, filter_synthetic_train], axis=0)
                # ['TPOT', 'H20', 'FLAML', 'mjar_supervised', 'TabPFN']
                performance_syn_filter = predict_syn_filter(filter_syn_train=filter_syn_train,
                                                            test=test,
                                                            target='Strength',
                                                            train_method=train_method)

                del filter_synthetic_train, filter_syn_train
                gc.collect()

                # log
                performance_syn_filter['random state'] = random_state
                performance_syn_filter['model'] = train_method

                performance_syn_filter['configuration'] = 'syn_filter'
                performance_syn_filter['configuration_syn_method'] = syn_method

                performance_syn_filter['filter_method'] = filter_method

                performances.append(performance_syn_filter)
        del test
        gc.collect()
        break

    summary = pd.DataFrame(performances)
    summary.to_csv(f'./results_hp/{train_method}_summary.csv', index=False)


if __name__ == "__main__":
    # train_method = ['TPOT', 'H20', 'FLAML', 'mjar_supervised', 'TabPFN', 'PyCaret', 'AutoGluon', 'auto-sklearn', 'AutoPytorch']
    train_experiments(train_method='TabPFN')
